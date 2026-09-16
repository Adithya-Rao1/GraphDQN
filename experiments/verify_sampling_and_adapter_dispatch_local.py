"""Local (no peft/transformers download needed) checks for two pieces of
this session's work:

1. ppo/gnn_llm_actor_critic.py::sample_edit's new temperature/greedy params
   -- default args must reproduce the exact prior behavior (Categorical over
   raw logits), greedy must always pick the argmax, temperature must widen
   the empirical sampling distribution as it increases.
2. _GNNLLMBackbone._fused_features's mode-gated adapter_names= dispatch
   logic -- exercised against a lightweight fake tokenizer/LLM (not real
   peft, which doesn't import in this environment -- see
   verify_adapter_names_mode_switch.py for the real-peft version, which
   needs the remote GPU box), to confirm the *branching logic itself* is
   correct: adapter_names= is passed only when the fake model reports
   .training is False and has a set_adapter method.

Both run in this local (broken numpy/scipy/sklearn/transformers-for-peft)
environment.
"""
import torch
from torch.distributions import Categorical

from ppo.gnn_llm_actor_critic import _GNNLLMBackbone, sample_edit


def check_sample_edit():
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 0.5, -1.0, 0.1])

    # default args reproduce prior behavior exactly: Categorical(logits=edit_logits)
    torch.manual_seed(0)
    idx1, logp1, ent1 = sample_edit(logits)
    torch.manual_seed(0)
    dist_ref = Categorical(logits=logits)
    idx_ref = dist_ref.sample()
    assert idx1.item() == idx_ref.item(), "default sample_edit() call diverged from pre-change Categorical(logits=...) behavior"
    assert torch.allclose(logp1, dist_ref.log_prob(idx_ref))
    assert torch.allclose(ent1, dist_ref.entropy())
    print("[PASS] sample_edit() default args reproduce prior behavior exactly")

    # greedy always picks argmax, regardless of RNG state
    for seed in range(10):
        torch.manual_seed(seed)
        idx, _, _ = sample_edit(logits, greedy=True)
        assert idx.item() == int(torch.argmax(logits).item()), f"greedy=True did not pick argmax at seed={seed}"
    print("[PASS] sample_edit(greedy=True) always picks argmax across different RNG states")

    # temperature widens the sampling distribution: empirical entropy at
    # temperature=5.0 should exceed empirical entropy at temperature=0.2
    def empirical_entropy(temp, n=4000):
        torch.manual_seed(1)
        counts = torch.zeros(logits.numel())
        for _ in range(n):
            idx, _, _ = sample_edit(logits, temperature=temp)
            counts[idx.item()] += 1
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    low_temp_entropy = empirical_entropy(0.2)
    high_temp_entropy = empirical_entropy(5.0)
    assert high_temp_entropy > low_temp_entropy, (
        f"temperature did not widen the sampling distribution as expected: "
        f"low_temp_entropy={low_temp_entropy}, high_temp_entropy={high_temp_entropy}"
    )
    print(f"[PASS] temperature widens sampling distribution as expected "
          f"(low={low_temp_entropy:.3f}, high={high_temp_entropy:.3f})")


class _FakeLLMOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        from transformers import BatchEncoding
        n = len(texts)
        return BatchEncoding({
            "input_ids": torch.zeros(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        })


class _FakeLLM(torch.nn.Module):
    """Mimics just enough of a PeftModel's forward surface to exercise
    _fused_features's real branching logic -- records what kwargs it was
    actually called with, so the test can assert on them directly."""

    def __init__(self):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": 8})()
        self.last_call_kwargs = None

    def set_adapter(self, name):
        pass

    def forward(self, **kwargs):
        self.last_call_kwargs = kwargs
        n = kwargs["input_ids"].shape[0]
        return _FakeLLMOutput(last_hidden_state=torch.zeros(n, 3, 8))


class _FakeLLMNoAdapter(torch.nn.Module):
    """Same forward surface as _FakeLLM but with no set_adapter method at
    all -- mimics a plain (non-PEFT) frozen model."""

    def __init__(self):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": 8})()
        self.last_call_kwargs = None

    def forward(self, **kwargs):
        self.last_call_kwargs = kwargs
        n = kwargs["input_ids"].shape[0]
        return _FakeLLMOutput(last_hidden_state=torch.zeros(n, 3, 8))


def check_fused_features_mode_gating():
    backbone = _GNNLLMBackbone.__new__(_GNNLLMBackbone)  # bypass __init__ (no encoder/peft needed for this check)
    torch.nn.Module.__init__(backbone)  # still need nn.Module's own bookkeeping dicts for submodule assignment
    backbone.use_llm = True
    backbone.tokenizer = _FakeTokenizer()
    backbone.llm = _FakeLLM()
    backbone._adapter_name = "actor"

    # Plain functions, not nn.Module instances -- fusion/trunk aren't under
    # test here, only _fused_features's adapter_names= dispatch logic is.
    def _fake_fusion(graph_embedding, llm_hidden_states, llm_attention_mask):
        return llm_hidden_states[:, 0, :]
    backbone.fusion = _fake_fusion
    backbone.trunk = lambda x: x

    graph_embedding = torch.zeros(1, 4)

    class _FakeBatch:
        pass  # unused: we bypass self.encoder(data_batch) by monkeypatching below

    backbone.encoder = lambda data_batch: graph_embedding

    # -- eval mode: adapter_names= should be passed (has set_adapter, not training)
    backbone.llm.eval()
    backbone._fused_features(_FakeBatch(), ["a molecule"])
    assert "adapter_names" in backbone.llm.last_call_kwargs, \
        "adapter_names= was NOT passed in eval mode with set_adapter present -- mode-gating regressed"
    assert backbone.llm.last_call_kwargs["adapter_names"] == ["actor"]
    print("[PASS] eval mode + set_adapter present -> adapter_names= is passed")

    # -- train mode: adapter_names= should NOT be passed (would raise ValueError against real peft)
    backbone.llm.train()
    backbone._fused_features(_FakeBatch(), ["a molecule"])
    assert "adapter_names" not in backbone.llm.last_call_kwargs, \
        "adapter_names= WAS passed in train mode -- this would raise ValueError against real peft"
    print("[PASS] train mode -> adapter_names= is NOT passed (relies on prior set_adapter())")

    # -- eval mode, no set_adapter attribute at all (plain non-PEFT model): never passed
    backbone.llm = _FakeLLMNoAdapter()
    backbone.llm.eval()
    backbone._fused_features(_FakeBatch(), ["a molecule"])
    assert "adapter_names" not in backbone.llm.last_call_kwargs, \
        "adapter_names= was passed even though the model has no set_adapter -- would break non-PEFT models"
    print("[PASS] no set_adapter attribute -> adapter_names= is never passed")


if __name__ == "__main__":
    check_sample_edit()
    check_fused_features_mode_gating()
    print("\nAll local checks passed.")

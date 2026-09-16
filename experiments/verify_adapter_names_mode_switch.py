"""Remote-GPU-box verification for the adapter_names/train-eval mode fix in
ppo/gnn_llm_actor_critic.py::_GNNLLMBackbone._fused_features and
ppo/llm_finetune.py::run_llm_finetune_cycle.

peft does not import in the local dev environment (broken scipy/numpy ABI
chain, same as noted in CLAUDE.md), so this cannot be run locally -- run it
on the remote GPU box before trusting the fix under real concurrency.

Checks:
  (a) In .eval() mode, adapter_names=["actor"]*N vs ["critic"]*N on
      identical input produce DIFFERENT outputs (proves per-call adapter
      identity is actually respected, the original race-condition fix).
  (b) Calling .train() + set_adapter("actor") + a plain forward (no
      adapter_names) succeeds with no ValueError, and its output matches a
      second forward call made the same way (deterministic given the same
      adapter and eval-equivalent dropout seed is not required here --
      only that it does not raise).
  (c) Simulating the full guarded sequence run_llm_finetune_cycle uses
      (train() -> set_adapter -> forward/backward -> eval()) leaves the
      model correctly back in .eval() mode, and that adapter_names=
      dispatch works correctly again immediately after -- proving the
      invariant is actually restored, not just correct at the start.
"""
import torch

from ppo.gnn_llm_actor_critic import build_shared_llm_backbone, default_qwen2_lora_config

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tokenizer, peft_model = build_shared_llm_backbone(
        TEST_MODEL_NAME,
        {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    peft_model.to(device)

    # _GNNLLMBackbone.__init__ calls self.llm.eval() right after construction
    # -- build_shared_llm_backbone itself doesn't, so replicate that here to
    # match what PGMORLAgent/GNNLLMActor actually does before any forward pass.
    peft_model.eval()
    assert not peft_model.training, "expected eval mode after construction"
    print("[PASS] backbone starts in eval mode")

    text = ["Molecule CCO against target sequence MKV; prioritizing binding affinity."]
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # (a) adapter_names= dispatch differs by adapter, in eval mode
    with torch.no_grad():
        out_actor = peft_model(**encoded, adapter_names=["actor"], output_hidden_states=True)
        out_critic = peft_model(**encoded, adapter_names=["critic"], output_hidden_states=True)
    hs_actor = out_actor.last_hidden_state if hasattr(out_actor, "last_hidden_state") else out_actor.hidden_states[-1]
    hs_critic = out_critic.last_hidden_state if hasattr(out_critic, "last_hidden_state") else out_critic.hidden_states[-1]
    assert not torch.allclose(hs_actor, hs_critic), \
        "actor vs critic adapter_names= dispatch produced identical output -- adapter identity not respected"
    print("[PASS] adapter_names= in eval mode dispatches to distinct adapters")

    # (b) train() + set_adapter() + plain forward succeeds (no ValueError)
    peft_model.train()
    peft_model.set_adapter("actor")
    target_params = [p for n, p in peft_model.named_parameters() if ".actor." in n]
    for p in target_params:
        p.requires_grad = True
    try:
        out = peft_model(**encoded, output_hidden_states=True)
    except ValueError as e:
        raise AssertionError(f"forward under train()+set_adapter() raised unexpectedly: {e}")
    loss = out.last_hidden_state.float().sum() if hasattr(out, "last_hidden_state") else out.hidden_states[-1].float().sum()
    loss.backward()
    assert any(p.grad is not None for p in target_params), "no gradient reached the actor adapter's LoRA params"
    print("[PASS] train()+set_adapter() forward/backward succeeds under exclusive-lock-equivalent conditions")

    for p in target_params:
        p.requires_grad = False

    # (c) restore eval() -- mirrors run_llm_finetune_cycle's own bracketing
    peft_model.eval()
    assert not peft_model.training, "expected eval mode restored after the guarded block"
    with torch.no_grad():
        out_actor_2 = peft_model(**encoded, adapter_names=["actor"], output_hidden_states=True)
    hs_actor_2 = (
        out_actor_2.last_hidden_state if hasattr(out_actor_2, "last_hidden_state")
        else out_actor_2.hidden_states[-1]
    )
    with torch.no_grad():
        out_critic_2 = peft_model(**encoded, adapter_names=["critic"], output_hidden_states=True)
    hs_critic_2 = (
        out_critic_2.last_hidden_state if hasattr(out_critic_2, "last_hidden_state")
        else out_critic_2.hidden_states[-1]
    )
    assert not torch.allclose(hs_actor_2, hs_critic_2), \
        "adapter_names= dispatch broken after train()/eval() round-trip -- invariant not restored"
    print("[PASS] adapter_names= dispatch works correctly again after the train()/eval() round-trip")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

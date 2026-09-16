import copy

import torch
from torch_geometric.data import Batch, Data

from ppo.gnn_llm_actor_critic import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GNNLLMActor,
    GNNLLMCritic,
    build_shared_llm_backbone,
    default_qwen2_lora_config,
)

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _fake_graph_batch(n_nodes=6, n_edges=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, NODE_FEATURE_DIM, generator=g)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    edge_attr = torch.randn(n_edges, EDGE_FEATURE_DIM, generator=g)
    return Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"Building shared backbone from {TEST_MODEL_NAME} with 'actor'/'critic' adapters (bfloat16)...")
    tokenizer, peft_model = build_shared_llm_backbone(
        TEST_MODEL_NAME,
        {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
        torch_dtype=torch.bfloat16,
    )
    shared_backbone = (tokenizer, peft_model)

    base_param_dtypes = {p.dtype for n, p in peft_model.named_parameters() if "lora_" not in n}
    print(f"frozen base parameter dtypes: {base_param_dtypes}")
    assert base_param_dtypes == {torch.bfloat16}, (
        f"expected the frozen base to load in bfloat16 (torch_dtype= wasn't honored), got {base_param_dtypes}"
    )
    print("[PASS] frozen base loaded in bfloat16 (not the fp32 default -- this is the actual OOM fix for 7B)")

    actor = GNNLLMActor(
        num_edit_ops=52, hidden_dim=64, use_llm=True,
        shared_llm_backbone=shared_backbone, adapter_name="actor",
        edit_count_mode="fixed",
    ).to(device)
    critic = GNNLLMCritic(
        hidden_dim=64, use_llm=True,
        shared_llm_backbone=shared_backbone, adapter_name="critic",
    ).to(device)

    assert actor.llm is critic.llm, "actor.llm and critic.llm must be the SAME object"
    print("[PASS] actor.llm is critic.llm (base weights genuinely shared, not duplicated)")

    batch = _fake_graph_batch().to(device)
    descriptor = ["Molecule CCO against target alpha_synuclein; prioritizing ADMET (0.30)."]
    edit_logits, _ = actor(batch, descriptor)
    value_vector = critic(batch, descriptor)
    print(f"actor edit_logits shape={tuple(edit_logits.shape)}  critic value_vector shape={tuple(value_vector.shape)}")
    assert edit_logits.shape == (1, 52)
    assert value_vector.shape == (1, 4)
    print("[PASS] actor/critic forward passes run correctly through the shared backbone")

    critic_lora_params_before = {
        name: p.detach().clone()
        for name, p in peft_model.named_parameters()
        if ".critic." in name and p.requires_grad
    }
    assert critic_lora_params_before, "no 'critic' adapter LoRA params found -- adapter setup is broken"

    actor_optimizer = torch.optim.Adam(
        [p for p in actor.parameters() if p.requires_grad], lr=1e-3,
    )
    actor_optimizer.zero_grad()
    loss = edit_logits.sum()
    loss.backward()
    actor_optimizer.step()

    unchanged = 0
    for name, p in peft_model.named_parameters():
        if name in critic_lora_params_before:
            if torch.equal(p.detach(), critic_lora_params_before[name]):
                unchanged += 1
    assert unchanged == len(critic_lora_params_before), (
        f"expected all {len(critic_lora_params_before)} 'critic' adapter params to be unchanged "
        f"after an actor-only update, but only {unchanged} were"
    )
    print(f"[PASS] actor-only update left all {unchanged} 'critic' adapter parameters unchanged "
          f"(gradient isolation between adapters confirmed)")

    seen_ids = set()
    trainable_params = []
    for p in list(actor.parameters()) + list(critic.parameters()):
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable_params.append(p)
    raw_trainable_count = sum(1 for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad)
    print(f"raw trainable param count (with duplicates) = {raw_trainable_count}, "
          f"deduped = {len(trainable_params)}")
    assert len(trainable_params) < raw_trainable_count, (
        "expected deduping to actually remove duplicate shared-backbone params -- "
        "if this fails, the shared backbone isn't being detected as shared"
    )
    print("[PASS] parameter dedup removes the shared backbone's duplicate entries")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

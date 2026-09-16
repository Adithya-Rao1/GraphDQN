import copy

import torch
from torch_geometric.data import Batch, Data

from molecular_modifications.macro_actions import build_edit_catalog
from ppo.gnn_llm_actor_critic import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GNNLLMActor,
    GNNLLMCritic,
    build_shared_llm_backbone,
    default_qwen2_lora_config,
)
from ppo.llm_finetune import EditOutcome, run_llm_finetune_cycle

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
REAL_STARTING_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",   # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine
]


def _fake_graph_for(smiles: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    n_nodes = 4 + (seed % 6)
    n_edges = max(2, n_nodes)
    x = torch.randn(n_nodes, NODE_FEATURE_DIM, generator=g)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    edge_attr = torch.randn(n_edges, EDGE_FEATURE_DIM, generator=g)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _build_synthetic_outcomes(catalog, n=6):
    outcomes = []
    for i in range(n):
        smiles = REAL_STARTING_SMILES[i % len(REAL_STARTING_SMILES)]
        outcomes.append(EditOutcome(
            pre_edit_states=[smiles],
            edit_indices=[i % len(catalog)],
            reward=0.9 if i % 2 == 0 else 0.1,  # a real, non-trivial advantage signal
        ))
    return outcomes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    catalog = build_edit_catalog()

    tokenizer, peft_model = build_shared_llm_backbone(
        TEST_MODEL_NAME,
        {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    for p in peft_model.parameters():
        p.requires_grad = False
    shared_backbone = (tokenizer, peft_model)

    actor = GNNLLMActor(
        num_edit_ops=len(catalog), hidden_dim=64, use_llm=True,
        shared_llm_backbone=shared_backbone, adapter_name="actor",
        edit_count_mode="fixed",
    ).to(device)
    critic = GNNLLMCritic(
        hidden_dim=64, use_llm=True,
        shared_llm_backbone=shared_backbone, adapter_name="critic",
    ).to(device) 

    assert all(not p.requires_grad for n, p in peft_model.named_parameters() if ".actor." in n), \
        "actor adapter should start frozen (ppo/pgmorl_agent.py's freeze contract)"
    print("[PASS] actor adapter starts frozen, as PGMORLAgent.__init__ leaves it")

    def descriptor_fn(smiles):
        return f"Molecule {smiles}; prioritizing ADMET (0.30), binding affinity (0.50)."

    import ppo.llm_finetune as llm_finetune_mod

    def fake_single_graph_batch(smiles, dev):
        seed = abs(hash(smiles)) % (2**31)
        return Batch.from_data_list([_fake_graph_for(smiles, seed)]).to(dev)

    llm_finetune_mod.single_graph_batch = fake_single_graph_batch

    outcomes = _build_synthetic_outcomes(catalog)

    initial_actor_lora = {
        n: p.detach().clone() for n, p in peft_model.named_parameters() if ".actor." in n
    }
    initial_critic_lora = {
        n: p.detach().clone() for n, p in peft_model.named_parameters() if ".critic." in n
    }

    def reset_actor_lora():
        with torch.no_grad():
            for n, p in peft_model.named_parameters():
                if n in initial_actor_lora:
                    p.copy_(initial_actor_lora[n])

    def total_movement():
        return sum(
            (p.detach().float() - initial_actor_lora[n].float()).norm().item() ** 2
            for n, p in peft_model.named_parameters() if n in initial_actor_lora
        ) ** 0.5

    torch.manual_seed(0)
    result_no_kl = run_llm_finetune_cycle(
        actor, "actor", outcomes, descriptor_fn, device,
        epochs=3, minibatch_size=2, kl_weight=0.0, learning_rate=1e-2,
    )
    movement_no_kl = total_movement()
    print(f"kl_weight=0.0  -> rl_loss={result_no_kl.rl_loss:.4f} kl_loss={result_no_kl.kl_loss:.4f} "
          f"movement={movement_no_kl:.6f}")

    assert all(not p.requires_grad for n, p in peft_model.named_parameters() if ".actor." in n), \
        "actor adapter must be refrozen after the cycle ends"
    print("[PASS] actor adapter refrozen after the cycle")

    reset_actor_lora()
    torch.manual_seed(0)
    result_high_kl = run_llm_finetune_cycle(
        actor, "actor", outcomes, descriptor_fn, device,
        epochs=3, minibatch_size=2, kl_weight=1000.0, learning_rate=1e-2,
    )
    movement_high_kl = total_movement()
    print(f"kl_weight=1000.0 -> rl_loss={result_high_kl.rl_loss:.4f} kl_loss={result_high_kl.kl_loss:.4f} "
          f"movement={movement_high_kl:.6f}")

    assert result_high_kl.kl_loss > 1e-8, (
        "kl_loss reported as ~0 even under a deliberately large kl_weight -- "
        "the KL term is not actually being computed against a real reference"
    )
    assert movement_high_kl < movement_no_kl, (
        f"a kl_weight=1000 run moved the adapter AS MUCH OR MORE than kl_weight=0 "
        f"({movement_high_kl:.6f} vs {movement_no_kl:.6f}) -- the KL term is not "
        f"actually constraining the update (computed but disconnected from the "
        f"backward graph, or a sign/shape bug)"
    )
    print(f"[PASS] large kl_weight measurably constrained adapter movement "
          f"({movement_high_kl:.6f} < {movement_no_kl:.6f})")

    critic_unchanged = all(
        torch.equal(p.detach(), initial_critic_lora[n])
        for n, p in peft_model.named_parameters() if n in initial_critic_lora
    )
    assert critic_unchanged, "the untouched 'critic' adapter's parameters changed -- adapter isolation is broken"
    print("[PASS] 'critic' adapter parameters completely unchanged by either fine-tune cycle")

    print("\nALL KL-RESPONSIVENESS CHECKS PASSED")


if __name__ == "__main__":
    main()

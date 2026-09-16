import torch

from ppo.gnn_llm_actor_critic import build_shared_llm_backbone, default_qwen2_lora_config

TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _perturb_lora_b(peft_model, adapter_name: str, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if f".{adapter_name}." in name and "lora_B" in name:
                noise = torch.randn(p.shape, generator=gen).to(p.device, p.dtype) * 0.05
                p.add_(noise)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tokenizer, peft_model = build_shared_llm_backbone(
        TEST_MODEL_NAME,
        {"actor": default_qwen2_lora_config(), "critic": default_qwen2_lora_config()},
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    peft_model.to(device)

    peft_model.eval()
    assert not peft_model.training, "expected eval mode after construction"
    print("[PASS] backbone starts in eval mode")

    text = ["Molecule CCO against target sequence MKV; prioritizing binding affinity."]
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    _perturb_lora_b(peft_model, "actor", seed=1)
    _perturb_lora_b(peft_model, "critic", seed=2)

    with torch.no_grad():
        out_actor = peft_model(**encoded, adapter_names=["actor"], output_hidden_states=True)
        out_critic = peft_model(**encoded, adapter_names=["critic"], output_hidden_states=True)
    hs_actor = out_actor.last_hidden_state if hasattr(out_actor, "last_hidden_state") else out_actor.hidden_states[-1]
    hs_critic = out_critic.last_hidden_state if hasattr(out_critic, "last_hidden_state") else out_critic.hidden_states[-1]
    assert not torch.allclose(hs_actor, hs_critic), \
        "actor vs critic adapter_names= dispatch produced identical output -- adapter identity not respected"
    print("[PASS] adapter_names= in eval mode dispatches to distinct adapters")

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

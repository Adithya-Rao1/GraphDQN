import argparse
import json
import os
import time

import torch

import dqn.dqn_hyperparams as dqn_hyp
from dqn.dqn_env import MoleculeEnv
from experiments.data.targets import TARGETS, DEFAULT_TARGET
from experiments.data.starting_molecules import sample_pilot_molecules
from molecular_modifications.macro_actions import build_edit_catalog
from reward.multi_objective import RewardConfig
from ppo.pgmorl_agent import DEFAULT_LLM_MODEL_NAME, PGMORLAgent
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility


def run(target_name=DEFAULT_TARGET, seed=0, num_molecules=30, num_episodes=200,
        max_steps=dqn_hyp.max_steps, fixed_edit_count=1, ppo_epochs=4,
        hidden_dim=256, lr=1e-4, gamma=dqn_hyp.gamma, gae_lambda=0.95, clip_eps=0.2,
        entropy_coef=0.01, value_coef=0.5,
        admet_weight=dqn_hyp.admet_weight, binding_weight=dqn_hyp.binding_weight,
        synthetic_weight=dqn_hyp.synthetic_weight, selectivity_weight=0.0,
        use_llm=False, llm_model_name=DEFAULT_LLM_MODEL_NAME, llm_torch_dtype=torch.bfloat16,
        reward_batch_size=8, predictors_device=None,
        checkpoint_interval=50, run_id=None, use_wandb=False,
        checkpoint_root='./checkpoints/pgmorl_ppo', results_root='./experiments/results'):
    run_id = run_id or f"pgmorl_ppo_{target_name}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictors_device = torch.device(predictors_device) if predictors_device else device

    target_seq = TARGETS[target_name]
    start_mols = sample_pilot_molecules(n=num_molecules, seed=seed)

    catalog = build_edit_catalog()
    reward_config = RewardConfig(admet_weight=admet_weight, binding_weight=binding_weight, synthetic_weight=synthetic_weight, selectivity_weight=selectivity_weight,)

    admet_model = ADMETModel(predictors_device)
    binding_model = Plapt(device=str(predictors_device))
    sa_model = SyntheticAccessibility()

    agent = PGMORLAgent(
        catalog=catalog, reward_config=reward_config, target_seq=target_seq, device=device,
        admet_model=admet_model, binding_model=binding_model, sa_model=sa_model,
        hidden_dim=hidden_dim, use_llm=use_llm, llm_model_name=llm_model_name if use_llm else None,
        llm_torch_dtype=llm_torch_dtype, reward_batch_size=reward_batch_size,
        edit_count_mode="fixed",
        fixed_edit_count=fixed_edit_count, gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, entropy_coef=entropy_coef, value_coef=value_coef, lr=lr,
    )

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "graphdqn"),
            name=run_id,
            config={"algorithm": "pgmorl_ppo", "target": target_name, "seed": seed,
                    "num_molecules": num_molecules, "num_episodes": num_episodes,
                    "fixed_edit_count": fixed_edit_count},
            tags=["pgmorl_ppo", target_name, f"seed{seed}"],
        )

    checkpoint_dir = os.path.join(checkpoint_root, run_id)
    start_time = time.time()
    episode_rewards = []
    episode_zero_edit_fallback_counts = []

    for episode in range(num_episodes):
        start_mol = start_mols[episode % len(start_mols)]
        env = MoleculeEnv(init_mol=start_mol, max_steps=max_steps)
        env.initialize()

        zero_edit_fallbacks = 0
        for step in range(max_steps):
            entry = agent.act(env)
            if entry.k_used == 0:
                zero_edit_fallbacks += 1
            if entry.done:
                break

        update_stats = agent.update(ppo_epochs=ppo_epochs)
        final_reward = update_stats["episode_reward"]
        episode_rewards.append(final_reward)
        episode_zero_edit_fallback_counts.append(zero_edit_fallbacks)

        if wandb_run:
            wandb_run.log({"episode": episode, "reward": final_reward, **update_stats})

        if checkpoint_interval and episode % checkpoint_interval == 0 and episode > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.save_checkpoint(os.path.join(checkpoint_dir, f"episode_{episode}.pt"))

        if episode % 10 == 0:
            print(f"[{run_id}] episode {episode} reward={final_reward:.4f} "
                  f"policy_loss={update_stats['policy_loss']:.4f} value_loss={update_stats['value_loss']:.4f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))

    summary = {
        "run_id": run_id,
        "algorithm": "pgmorl_ppo",
        "target": target_name,
        "seed": seed,
        "num_molecules": num_molecules,
        "num_episodes": num_episodes,
        "fixed_edit_count": fixed_edit_count,
        "use_llm": use_llm,
        "llm_model_name": llm_model_name if use_llm else None,
        "final_reward": episode_rewards[-1] if episode_rewards else None,
        "mean_reward_last_10": (sum(episode_rewards[-10:]) / len(episode_rewards[-10:])) if episode_rewards else None,
        "mean_zero_edit_fallback_rate": (
            sum(episode_zero_edit_fallback_counts) / (len(episode_zero_edit_fallback_counts) * max_steps)
        ) if episode_zero_edit_fallback_counts else None,
        "wall_clock_seconds": time.time() - start_time,
        "checkpoint_dir": checkpoint_dir,
    }

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run:
        wandb_run.log(summary)
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-name", default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-molecules", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=dqn_hyp.max_steps)
    parser.add_argument("--fixed-edit-count", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--use-llm", action="store_true",)
    parser.add_argument("--llm-model-name", default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--llm-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],)
    parser.add_argument("--reward-batch-size", type=int, default=8,)
    parser.add_argument("--predictors-device", default=None, choices=[None, "cpu", "cuda"],)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result_summary = run(
        target_name=args.target_name,
        seed=args.seed,
        num_molecules=args.num_molecules,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        fixed_edit_count=args.fixed_edit_count,
        ppo_epochs=args.ppo_epochs,
        use_llm=args.use_llm,
        llm_model_name=args.llm_model_name,
        llm_torch_dtype=getattr(torch, args.llm_dtype),
        reward_batch_size=args.reward_batch_size or None,
        predictors_device=args.predictors_device,
        checkpoint_interval=args.checkpoint_interval,
        run_id=args.run_id,
        use_wandb=args.wandb,
    )
    print(json.dumps(result_summary, indent=2))

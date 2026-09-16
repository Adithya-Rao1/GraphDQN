import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from dqn.utils import create_graph, obs_to_loader
from molecular_modifications.macro_actions import EditOp, MacroActionResult, compose_macro_action
from ppo.gnn_llm_actor_critic import GNNLLMActor, GNNLLMCritic, edit_log_prob_and_entropy, sample_edit
from reward.multi_objective import RewardConfig, compute_reward

VALID_EDIT_COUNT_MODES = ("fixed", "random", "learned")


def _single_graph_batch(smiles: str, device):
    loader = obs_to_loader(create_graph([smiles]), batch_size=1)
    (batch,) = list(loader)
    return batch.to(device)


@dataclass
class MacroStepMemory:
    pre_edit_states: List[str]       
    edit_ids: List[str]              
    edit_indices: List[int]           
    old_edit_log_probs: List[float]  
    initial_state: str                
    k_used: int
    k_log_prob: Optional[float]       
    reward: float                    
    reward_vector: List[float]        
    value_scalar: float              
    value_vector: List[float]         
    done: bool


@dataclass
class RolloutStats:
    num_macro_steps: int = 0
    num_fully_applied: int = 0
    num_zero_edit_fallbacks: int = 0
    total_reward: float = 0.0


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"weight vector must sum to a positive value, got {weights}")
    return [w / total for w in weights]


class PGMORLAgent:
    def __init__(
        self,
        catalog: List[EditOp],
        reward_config: RewardConfig,
        target_seq: str,
        device,
        off_target_seq: Optional[str] = None,
        admet_model=None,
        binding_model=None,
        sa_model=None,
        hidden_dim: int = 256,
        use_llm: bool = False,
        llm_model_name: Optional[str] = None,
        lora_config=None,
        edit_count_mode: str = "fixed",
        fixed_edit_count: int = 1,
        edit_count_range: Optional[Sequence[int]] = None,
        k_max: Optional[int] = None,
        max_resample_attempts: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
    ):
        if edit_count_mode not in VALID_EDIT_COUNT_MODES:
            raise ValueError(f"edit_count_mode must be one of {VALID_EDIT_COUNT_MODES}, got {edit_count_mode!r}")
        if edit_count_mode == "random" and not edit_count_range:
            raise ValueError("edit_count_range=(min, max) is required when edit_count_mode == 'random'")
        if edit_count_mode == "learned" and not k_max:
            raise ValueError("k_max is required when edit_count_mode == 'learned'")

        self.catalog = catalog
        self.catalog_by_id: Dict[str, EditOp] = {op.id: op for op in catalog}
        self.edit_id_to_index: Dict[str, int] = {op.id: i for i, op in enumerate(catalog)}

        self.device = device

        self.reward_config = reward_config
        self.weight_vector = _normalize_weights([
            reward_config.admet_weight, reward_config.binding_weight,
            reward_config.synthetic_weight, reward_config.selectivity_weight,
        ])
        self.target_seq = target_seq
        self.off_target_seq = off_target_seq

        self.admet_model = admet_model
        self.binding_model = binding_model
        self.sa_model = sa_model
        if self.admet_model is None:
            from ADMET.model import ADMETModel
            self.admet_model = ADMETModel(device)
        if self.binding_model is None:
            from binding_module.binding_affinity.plapt import Plapt
            self.binding_model = Plapt(device=str(device))
        if self.sa_model is None:
            from synthetic_accessibility.sa_score import SyntheticAccessibility
            self.sa_model = SyntheticAccessibility()

        self.edit_count_mode = edit_count_mode
        self.fixed_edit_count = fixed_edit_count
        self.edit_count_range = tuple(edit_count_range) if edit_count_range else None
        self.k_max = k_max
        self.max_resample_attempts = max_resample_attempts

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = GNNLLMActor(
            num_edit_ops=len(catalog), hidden_dim=hidden_dim, use_llm=use_llm,
            llm_model_name=llm_model_name, lora_config=lora_config,
            edit_count_mode=edit_count_mode, k_max=k_max,
        ).to(device)
        self.critic = GNNLLMCritic(
            hidden_dim=hidden_dim, use_llm=use_llm,
            llm_model_name=llm_model_name, lora_config=lora_config,
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr,
        )

        self.memory: List[MacroStepMemory] = []

    def _compute_reward(self, smiles: str) -> dict:
        return compute_reward(
            smiles=smiles, target_seq=self.target_seq, device=self.device,
            off_target_seq=self.off_target_seq,
            admet_model=self.admet_model, binding_model=self.binding_model, sa_model=self.sa_model,
            **self.reward_config.to_compute_reward_kwargs(),
        )

    def _descriptor_text(self, smiles: str, target_name: Optional[str]) -> str:
        w = self.weight_vector
        target_clause = f" against target {target_name}" if target_name else ""
        return (
            f"Molecule {smiles}{target_clause}; prioritizing ADMET ({w[0]:.2f}), "
            f"binding affinity ({w[1]:.2f}), synthetic accessibility ({w[2]:.2f}), "
            f"selectivity ({w[3]:.2f})."
        )

    def _sample_k(self, state_smiles: str, target_name: Optional[str]):
        if self.edit_count_mode == "fixed":
            return self.fixed_edit_count, None
        if self.edit_count_mode == "random":
            lo, hi = self.edit_count_range
            return random.randint(lo, hi), None

        # learned
        batch = _single_graph_batch(state_smiles, self.device)
        descriptor = [self._descriptor_text(state_smiles, target_name)] if self.actor.use_llm else None
        with torch.no_grad():
            _, k_logits = self.actor(batch, descriptor)
        k_idx, k_log_prob, _ = sample_edit(k_logits.squeeze(0))
        return int(k_idx.item()) + 1, k_log_prob.item()  # k in {1..k_max}

    def _attempt_macro_action(self, initial_state: str, k: int, target_name: Optional[str]):
        pre_edit_states: List[str] = []
        edit_ids: List[str] = []
        edit_indices: List[int] = []
        old_log_probs: List[float] = []

        current_smiles = initial_state
        for _ in range(k):
            batch = _single_graph_batch(current_smiles, self.device)
            descriptor = [self._descriptor_text(current_smiles, target_name)] if self.actor.use_llm else None
            with torch.no_grad():
                edit_logits, _ = self.actor(batch, descriptor)
            idx, log_prob, _ = sample_edit(edit_logits.squeeze(0))
            edit_id = self.catalog[int(idx.item())].id

            result = compose_macro_action(current_smiles, [edit_id], self.catalog_by_id)
            if result is None or not result.applied_edit_ids:
                break 

            pre_edit_states.append(current_smiles)
            edit_ids.append(edit_id)
            edit_indices.append(int(idx.item()))
            old_log_probs.append(log_prob.item())
            current_smiles = result.final_smiles

        return pre_edit_states, edit_ids, edit_indices, old_log_probs, current_smiles

    def act(self, env, target_name: Optional[str] = None) -> MacroStepMemory:
        initial_state = env.state
        k, k_log_prob = self._sample_k(initial_state, target_name)

        pre_edit_states, edit_ids, edit_indices, old_log_probs = [], [], [], []
        final_smiles = initial_state
        for attempt in range(self.max_resample_attempts):
            pre_edit_states, edit_ids, edit_indices, old_log_probs, final_smiles = (
                self._attempt_macro_action(initial_state, k, target_name)
            )
            if edit_ids:
                break

        value_batch = _single_graph_batch(initial_state, self.device)
        descriptor = [self._descriptor_text(initial_state, target_name)] if self.critic.use_llm else None
        with torch.no_grad():
            value_vector = self.critic(value_batch, descriptor).squeeze(0)
        value_vector_list = value_vector.tolist()
        value_scalar = float(sum(w * v for w, v in zip(self.weight_vector, value_vector_list)))

        result = env.step_macro(final_smiles, edit_trace=edit_ids if edit_ids else None)
        reward_result = self._compute_reward(final_smiles)

        memory_entry = MacroStepMemory(
            pre_edit_states=pre_edit_states,
            edit_ids=edit_ids,
            edit_indices=edit_indices,
            old_edit_log_probs=old_log_probs,
            initial_state=initial_state,
            k_used=len(edit_ids),
            k_log_prob=k_log_prob,
            reward=float(reward_result["reward"]),
            reward_vector=list(reward_result["reward_vector"]),
            value_scalar=value_scalar,
            value_vector=value_vector_list,
            done=bool(result.terminated),
        )
        self.memory.append(memory_entry)
        return memory_entry

    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]):
        advantages = [0.0] * len(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else gae)
            advantages[t] = gae
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def _recompute_macro_log_prob_and_entropy(self, entry: MacroStepMemory, target_name: Optional[str]):
        if not entry.edit_ids:
            zero = torch.zeros((), device=self.device)
            return zero, zero

        total_log_prob = torch.zeros((), device=self.device)
        total_entropy = torch.zeros((), device=self.device)
        for state, action_idx in zip(entry.pre_edit_states, entry.edit_indices):
            batch = _single_graph_batch(state, self.device)
            descriptor = [self._descriptor_text(state, target_name)] if self.actor.use_llm else None
            edit_logits, _ = self.actor(batch, descriptor)
            action_tensor = torch.tensor(action_idx, device=self.device)
            log_prob, entropy = edit_log_prob_and_entropy(edit_logits.squeeze(0), action_tensor)
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy
        return total_log_prob, total_entropy

    def update(self, ppo_epochs: int = 4, target_name: Optional[str] = None) -> dict:
        if not self.memory:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        rewards = [m.reward for m in self.memory]
        values = [m.value_scalar for m in self.memory]
        dones = [m.done for m in self.memory]
        advantages, returns = self._compute_gae(rewards, values, dones)

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if adv_tensor.numel() > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        vector_returns = []
        for dim in range(4):
            dim_rewards = [m.reward_vector[dim] for m in self.memory]
            dim_values = [m.value_vector[dim] for m in self.memory]
            _, dim_returns = self._compute_gae(dim_rewards, dim_values, dones)
            vector_returns.append(dim_returns)
        vector_returns_tensor = torch.tensor(vector_returns, dtype=torch.float32, device=self.device).T  # [T, 4]

        old_log_probs = torch.tensor(
            [sum(m.old_edit_log_probs) + (m.k_log_prob or 0.0) for m in self.memory],
            dtype=torch.float32, device=self.device,
        )
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(ppo_epochs):
            new_log_probs = []
            entropies = []
            new_value_vectors = []
            for entry in self.memory:
                log_prob, entropy = self._recompute_macro_log_prob_and_entropy(entry, target_name)
                if entry.k_log_prob is not None and self.actor.k_head is not None:
                    k_batch = _single_graph_batch(entry.initial_state, self.device)
                    k_descriptor = [self._descriptor_text(entry.initial_state, target_name)] if self.actor.use_llm else None
                    _, k_logits = self.actor(k_batch, k_descriptor)
                    k_action = torch.tensor(entry.k_used - 1, device=self.device)
                    k_log_prob, k_entropy = edit_log_prob_and_entropy(k_logits.squeeze(0), k_action)
                    log_prob = log_prob + k_log_prob
                    entropy = entropy + k_entropy

                value_batch = _single_graph_batch(entry.initial_state, self.device)
                value_descriptor = [self._descriptor_text(entry.initial_state, target_name)] if self.critic.use_llm else None
                value_vector = self.critic(value_batch, value_descriptor).squeeze(0)

                new_log_probs.append(log_prob)
                entropies.append(entropy)
                new_value_vectors.append(value_vector)

            new_log_probs = torch.stack(new_log_probs)
            entropies = torch.stack(entropies)
            new_value_vectors = torch.stack(new_value_vectors) 

            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_1 = ratio * adv_tensor
            surrogate_2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_tensor
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            entropy_bonus = entropies.mean()

            value_loss = nn.functional.mse_loss(new_value_vectors, vector_returns_tensor)

            loss = policy_loss - self.entropy_coef * entropy_bonus + self.value_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = entropy_bonus.item()

        self.memory.clear()
        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "mean_return": float(returns_tensor.mean().item()) if len(returns) else 0.0,
        }

    def save_checkpoint(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "weight_vector": self.weight_vector,
        }, path)

    def load_checkpoint(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

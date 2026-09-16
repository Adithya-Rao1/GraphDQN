import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ppo.gnn_llm_actor_critic import GNNLLMActor, edit_log_prob_and_entropy, sample_edit
from ppo.pgmorl_agent import single_graph_batch


@dataclass
class EditOutcome:
    """Algorithm-layer mirror of one webapp EditOutcomeLog row (see
    webapp/backend/models.py) -- decoupled from SQLAlchemy."""
    pre_edit_states: List[str]
    edit_indices: List[int]
    reward: float


@dataclass
class FineTuneCycleResult:
    num_examples: int
    rl_loss: float
    kl_loss: float
    total_loss: float
    num_updates: int


def _forward_sequence(actor: GNNLLMActor, outcome: EditOutcome,
                       descriptor_fn: Callable[[str], str], device):
    total_log_prob = torch.zeros((), device=device)
    logits_list = []
    for state, action_idx in zip(outcome.pre_edit_states, outcome.edit_indices):
        batch = single_graph_batch(state, device)
        descriptor = [descriptor_fn(state)] if actor.use_llm else None
        edit_logits, _ = actor(batch, descriptor)
        edit_logits = edit_logits.squeeze(0)
        action_tensor = torch.tensor(action_idx, device=device)
        log_prob, _ = edit_log_prob_and_entropy(edit_logits, action_tensor)
        total_log_prob = total_log_prob + log_prob
        logits_list.append(edit_logits)
    return total_log_prob, logits_list


def run_llm_finetune_cycle(
    actor: GNNLLMActor,
    adapter_name: str,
    outcomes: Sequence[EditOutcome],
    descriptor_fn: Callable[[str], str],
    device,
    epochs: int = 4,
    minibatch_size: int = 4,
    kl_weight: float = 0.1,
    clip_eps: float = 0.2,
    learning_rate: float = 1e-5,
    rw_lock=None,
) -> FineTuneCycleResult:
    if not outcomes:
        return FineTuneCycleResult(num_examples=0, rl_loss=0.0, kl_loss=0.0, total_loss=0.0, num_updates=0)
    if not actor.use_llm:
        raise ValueError("run_llm_finetune_cycle requires an actor built with use_llm=True")

    write_ctx = rw_lock.gen_wlock() if rw_lock is not None else nullcontext()

    with write_ctx:
        peft_model = actor.llm
        peft_model.set_adapter(adapter_name)

        target_params = [p for n, p in peft_model.named_parameters() if f".{adapter_name}." in n]
        if not target_params:
            raise ValueError(f"no LoRA parameters found for adapter {adapter_name!r} -- adapter setup is broken")
        for p in target_params:
            p.requires_grad = True

        optimizer = torch.optim.Adam(target_params, lr=learning_rate)

        baseline_reward = sum(o.reward for o in outcomes) / len(outcomes)

        old_log_probs: List[torch.Tensor] = []
        old_edit_logits_ref: List[List[torch.Tensor]] = []
        with torch.no_grad():
            for outcome in outcomes:
                lp, logits_list = _forward_sequence(actor, outcome, descriptor_fn, device)
                old_log_probs.append(lp)
                old_edit_logits_ref.append(logits_list)

        total_rl_loss = total_kl_loss = total_loss_sum = 0.0
        num_updates = 0

        for _ in range(epochs):
            indices = list(range(len(outcomes)))
            random.shuffle(indices)

            for start in range(0, len(indices), minibatch_size):
                batch_indices = indices[start:start + minibatch_size]

                rl_terms = []
                kl_terms = []
                for i in batch_indices:
                    outcome = outcomes[i]
                    new_log_prob, new_logits_list = _forward_sequence(actor, outcome, descriptor_fn, device)

                    advantage = outcome.reward - baseline_reward
                    ratio = torch.exp(new_log_prob - old_log_probs[i])
                    surrogate_1 = ratio * advantage
                    surrogate_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
                    rl_terms.append(-torch.min(surrogate_1, surrogate_2))

                    step_kls = []
                    for new_logits, old_logits in zip(new_logits_list, old_edit_logits_ref[i]):
                        new_log_probs_full = F.log_softmax(new_logits, dim=-1)
                        old_probs_full = F.softmax(old_logits, dim=-1)
                        step_kls.append(F.kl_div(new_log_probs_full, old_probs_full, reduction="sum"))
                    kl_terms.append(
                        torch.stack(step_kls).mean() if step_kls else torch.zeros((), device=device)
                    )

                rl_loss = torch.stack(rl_terms).mean()
                kl_loss = torch.stack(kl_terms).mean()
                loss = rl_loss + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_rl_loss += rl_loss.item()
                total_kl_loss += kl_loss.item()
                total_loss_sum += loss.item()
                num_updates += 1

        for p in target_params:
            p.requires_grad = False

    return FineTuneCycleResult(
        num_examples=len(outcomes),
        rl_loss=total_rl_loss / max(num_updates, 1),
        kl_loss=total_kl_loss / max(num_updates, 1),
        total_loss=total_loss_sum / max(num_updates, 1),
        num_updates=num_updates,
    )

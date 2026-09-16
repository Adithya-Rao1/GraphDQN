import dataclasses
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


class SweepCancelled(Exception):
    pass

from ppo.pgmorl_agent import PGMORLAgent
from ppo.pgmorl_predictor import (
    DEFAULT_REFERENCE_POINT,
    OBJECTIVE_DIM,
    PerformancePredictor,
    PerformanceRecord,
    non_dominated_indices,
    score_candidate_assignment,
)
from reward.multi_objective import RewardConfig


def sample_population_weights(center: Sequence[float], population_size: int, concentration_alpha: float,
                               rng: np.random.Generator) -> List[List[float]]:
    center = np.asarray(center, dtype=float)
    total = center.sum()
    if total <= 0:
        raise ValueError(f"center weight vector must sum to a positive value, got {center}")
    center = center / total
    alpha = np.clip(concentration_alpha * center, 1e-3, None)
    samples = rng.dirichlet(alpha, size=population_size)
    return [s.tolist() for s in samples]


@dataclass
class PopulationMember:
    member_id: str
    agent: PGMORLAgent
    weight_vector: List[float]
    objective_vector: List[float] = field(default_factory=lambda: [0.0] * OBJECTIVE_DIM)
    total_training_steps: int = 0


@dataclass
class ParetoSweepResult:
    members: List[PopulationMember]
    records: List[PerformanceRecord]
    predictor: PerformancePredictor


def measure_objective_vector(agent: PGMORLAgent, env_factory: Callable[[], object],
                               num_eval_episodes: int, max_steps: int) -> List[float]:
    agent.memory.clear()
    for _ in range(num_eval_episodes):
        env = env_factory()
        for _ in range(max_steps):
            entry = agent.act(env)
            if entry.done:
                break
    agent._compute_rewards_for_memory()
    vectors = np.array([m.reward_vector for m in agent.memory], dtype=float)
    agent.memory.clear()
    return vectors.mean(axis=0).tolist() if len(vectors) else [0.0] * OBJECTIVE_DIM


def train_member_one_round(member: PopulationMember, env_factory: Callable[[], object],
                             episodes_per_round: int, max_steps: int, ppo_epochs: int,
                             cancel_event: Optional[threading.Event] = None,
                             on_step_scored: Optional[Callable[[str, object], None]] = None) -> None:
    step_cb = (lambda entry: on_step_scored(member.member_id, entry)) if on_step_scored is not None else None
    for _ in range(episodes_per_round):
        if cancel_event is not None and cancel_event.is_set():
            raise SweepCancelled(f"Sweep cancelled mid-round for {member.member_id}")
        env = env_factory()
        for _ in range(max_steps):
            entry = member.agent.act(env)
            if entry.done:
                break
        member.agent.update(ppo_epochs=ppo_epochs, on_step_scored=step_cb)
    member.total_training_steps += episodes_per_round


def run_pareto_sweep_sequential(
    base_reward_config: RewardConfig,
    build_agent_fn: Callable[[RewardConfig], PGMORLAgent],
    env_factory: Callable[[], object],
    population_size: int = 6,
    concentration_alpha: float = 20.0,
    num_rounds: int = 5,
    episodes_per_round: int = 10,
    eval_episodes_per_round: int = 3,
    max_steps: int = 40,
    ppo_epochs: int = 4,
    beta: float = 1.0,
    reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT,
    seed: int = 0,
    use_predictor: bool = True,
    cancel_event: Optional[threading.Event] = None,
    on_round_complete: Callable[[int, List[PopulationMember], List[PerformanceRecord]], None] = None,
    on_step_scored: Optional[Callable[[str, object], None]] = None,
) -> ParetoSweepResult:
    rng = np.random.default_rng(seed)
    center = [
        base_reward_config.admet_weight, base_reward_config.binding_weight,
        base_reward_config.synthetic_weight, base_reward_config.selectivity_weight,
    ]
    weight_vectors = sample_population_weights(center, population_size, concentration_alpha, rng)

    members: List[PopulationMember] = []
    for i, w in enumerate(weight_vectors):
        member_config = dataclasses.replace(
            base_reward_config, admet_weight=w[0], binding_weight=w[1],
            synthetic_weight=w[2], selectivity_weight=w[3],
        )
        agent = build_agent_fn(member_config)
        members.append(PopulationMember(member_id=f"member_{i}", agent=agent, weight_vector=w))

    predictor = PerformancePredictor()
    records: List[PerformanceRecord] = []

    for member in members:
        member.objective_vector = measure_objective_vector(
            member.agent, env_factory, eval_episodes_per_round, max_steps,
        )

    for round_idx in range(num_rounds):
        current_front = np.array([m.objective_vector for m in members])

        if not use_predictor:
            order = np.arange(len(members)) 
        elif predictor.is_fitted:
            scores = [
                score_candidate_assignment(
                    predictor, current_front, member.objective_vector, member.weight_vector,
                    episodes_per_round, beta, reference_point,
                )
                for member in members
            ]
            order = np.argsort(scores)[::-1]
        else:
            order = np.arange(len(members))
        for idx in order:
            member = members[int(idx)]
            objective_before = list(member.objective_vector)

            train_member_one_round(member, env_factory, episodes_per_round, max_steps, ppo_epochs,
                                    cancel_event=cancel_event, on_step_scored=on_step_scored)

            objective_after = measure_objective_vector(
                member.agent, env_factory, eval_episodes_per_round, max_steps,
            )
            member.objective_vector = objective_after

            records.append(PerformanceRecord(
                member_id=member.member_id,
                objective_vector_before=objective_before,
                weight_vector_used=member.weight_vector,
                training_steps_this_round=episodes_per_round,
                objective_vector_after=objective_after,
            ))

        if use_predictor:
            predictor.fit(records)

        if on_round_complete is not None:
            on_round_complete(round_idx, list(members), list(records))

    return ParetoSweepResult(members=members, records=records, predictor=predictor)

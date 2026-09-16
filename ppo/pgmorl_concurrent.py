import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from ppo.pgmorl_agent import PGMORLAgent
from ppo.pgmorl_population import (
    ParetoSweepResult,
    PopulationMember,
    measure_objective_vector,
    sample_population_weights,
    train_member_one_round,
)
from ppo.pgmorl_predictor import (
    DEFAULT_REFERENCE_POINT,
    PerformancePredictor,
    PerformanceRecord,
    score_candidate_assignment,
)
from reward.multi_objective import RewardConfig


def train_member_one_round_guarded(
    member: PopulationMember, env_factory: Callable[[], object],
    episodes_per_round: int, max_steps: int, ppo_epochs: int,
    rw_lock=None, use_cuda_stream: bool = True,
    cancel_event: Optional[threading.Event] = None,
    on_step_scored: Optional[Callable[[str, object], None]] = None,
) -> None:
    read_ctx = rw_lock.gen_rlock() if rw_lock is not None else nullcontext()
    stream = torch.cuda.Stream() if (use_cuda_stream and torch.cuda.is_available()) else None
    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()

    with read_ctx, stream_ctx:
        train_member_one_round(member, env_factory, episodes_per_round, max_steps, ppo_epochs,
                                cancel_event=cancel_event, on_step_scored=on_step_scored)
        if stream is not None:
            stream.synchronize()


def run_pareto_sweep_concurrent(
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
    max_concurrent_members: int = 3,
    rw_lock=None,
    use_cuda_streams: bool = True,
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
    records_lock = threading.Lock() 

    for member in members:
        member.objective_vector = measure_objective_vector(
            member.agent, env_factory, eval_episodes_per_round, max_steps, cancel_event=cancel_event,
        )

    def _run_one_member(member: PopulationMember) -> None:
        objective_before = list(member.objective_vector)

        train_member_one_round_guarded(
            member, env_factory, episodes_per_round, max_steps, ppo_epochs,
            rw_lock=rw_lock, use_cuda_stream=use_cuda_streams, cancel_event=cancel_event,
            on_step_scored=on_step_scored,
        )

        objective_after = measure_objective_vector(
            member.agent, env_factory, eval_episodes_per_round, max_steps, cancel_event=cancel_event,
        )
        member.objective_vector = objective_after

        record = PerformanceRecord(
            member_id=member.member_id,
            objective_vector_before=objective_before,
            weight_vector_used=member.weight_vector,
            training_steps_this_round=episodes_per_round,
            objective_vector_after=objective_after,
        )
        with records_lock:
            records.append(record)

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

        with ThreadPoolExecutor(max_workers=max_concurrent_members) as pool:
            futures = [pool.submit(_run_one_member, members[int(idx)]) for idx in order]
            for future in as_completed(futures):
                future.result()  

        if use_predictor:
            predictor.fit(records)

        if on_round_complete is not None:
            with records_lock:
                on_round_complete(round_idx, list(members), list(records))

    return ParetoSweepResult(members=members, records=records, predictor=predictor)

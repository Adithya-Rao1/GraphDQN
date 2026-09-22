from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

OBJECTIVE_DIM = 4  
DEFAULT_REFERENCE_POINT = (0.0, 0.0, 0.0, 0.0)  


@dataclass
class PerformanceRecord:
    member_id: str
    objective_vector_before: List[float]
    weight_vector_used: List[float]
    training_steps_this_round: int
    objective_vector_after: List[float]
    real_macro_steps_this_round: int = 0


class PerformancePredictor:
    def __init__(self, min_records_to_fit: int = 2):
        self.gps: List[Optional[GaussianProcessRegressor]] = [None] * OBJECTIVE_DIM
        self.min_records_to_fit = min_records_to_fit
        self._fitted = False

    @staticmethod
    def _build_features(records: Sequence[PerformanceRecord]) -> np.ndarray:
        return np.array([
            list(r.objective_vector_before) + list(r.weight_vector_used) + [r.training_steps_this_round]
            for r in records
        ], dtype=float)

    def fit(self, records: Sequence[PerformanceRecord]) -> None:
        if len(records) < self.min_records_to_fit:
            self._fitted = False
            return
        X = self._build_features(records)
        for dim in range(OBJECTIVE_DIM):
            y = np.array([r.objective_vector_after[dim] - r.objective_vector_before[dim] for r in records])
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
            gp.fit(X, y)
            self.gps[dim] = gp
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_delta(self, objective_vector_before: Sequence[float], weight_vector: Sequence[float],
                       training_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                f"PerformancePredictor.fit() must be called with >={self.min_records_to_fit} "
                f"records before predict_delta()"
            )
        x = np.array([list(objective_vector_before) + list(weight_vector) + [training_steps]], dtype=float)
        means = np.zeros(OBJECTIVE_DIM)
        stds = np.zeros(OBJECTIVE_DIM)
        for dim in range(OBJECTIVE_DIM):
            mean, std = self.gps[dim].predict(x, return_std=True)
            means[dim] = mean[0]
            stds[dim] = std[0]
        return means, stds


def hypervolume(front: np.ndarray, reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT) -> float:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return 0.0
    hv_indicator = HV(ref_point=-np.asarray(reference_point, dtype=float))
    return float(hv_indicator(-front))


def non_dominated_indices(objective_vectors: np.ndarray) -> np.ndarray:
    objective_vectors = np.asarray(objective_vectors, dtype=float)
    if objective_vectors.size == 0:
        return np.array([], dtype=int)
    front = NonDominatedSorting().do(-objective_vectors, only_non_dominated_front=True)
    return np.asarray(front)


def score_candidate_assignment(
    predictor: PerformancePredictor,
    current_front: np.ndarray,
    member_objective_vector: Sequence[float],
    candidate_weight_vector: Sequence[float],
    training_steps: int,
    beta: float,
    reference_point: Sequence[float] = DEFAULT_REFERENCE_POINT,
) -> float:
    mean_delta, std_delta = predictor.predict_delta(member_objective_vector, candidate_weight_vector, training_steps)
    optimistic_after = np.asarray(member_objective_vector, dtype=float) + mean_delta + beta * std_delta

    current_front = np.asarray(current_front, dtype=float)
    baseline_hv = hypervolume(current_front, reference_point)
    hypothetical_front = (
        np.vstack([current_front, optimistic_after[None, :]]) if current_front.size else optimistic_after[None, :]
    )
    new_hv = hypervolume(hypothetical_front, reference_point)

    return new_hv - baseline_hv

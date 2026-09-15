from reward.multi_objective import RewardConfig 

__all__ = ["RewardConfig", "normalize_rating", "blend_reward"]


def normalize_rating(rating: float) -> float:
    return max(0.0, min(1.0, rating / 100.0))


def blend_reward(computed_reward: float, normalized_rating: float, alpha: float) -> float:
    return (1 - alpha) * computed_reward + alpha * normalized_rating

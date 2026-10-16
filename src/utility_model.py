import numpy as np
from scipy.stats import beta
from dataclasses import dataclass


@dataclass
class BenchmarkProfile:
    raw_cobb_douglas: float
    calibrated_percentile: float
    effective_recall_prob: float
    dimension_contributions: dict[str, float]


def compute_cobb_douglas_benchmark(
    scores: dict[str, float],
    weights: dict[str, float],
    pop_alpha: float = 2.5,
    pop_beta: float = 2.5,
    base_recall_prob: float = 0.0,
    years_since_breakup: float = 1.0,
    decay_rate: float = 0.35,
) -> BenchmarkProfile:
    """
    Computes multiplicative Cobb-Douglas utility U = prod (s_k / 10)^(w_k)
    and maps it onto a population Beta(pop_alpha, pop_beta) CDF to obtain
    the calibrated benchmark percentile theta in (0, 1).
    """
    keys = list(scores.keys())
    s_vec = np.array([np.clip(scores[k], 0.5, 10.0) / 10.0 for k in keys], dtype=float)
    w_vec = np.array([max(weights.get(k, 1.0), 0.0) for k in keys], dtype=float)

    if np.sum(w_vec) == 0:
        w_vec = np.ones_like(w_vec) / len(w_vec)
    else:
        w_vec = w_vec / np.sum(w_vec)

    # Multiplicative Cobb-Douglas aggregation
    raw_u = float(np.prod(s_vec ** w_vec))

    # Map raw utility onto population percentile via Beta CDF
    theta = float(np.clip(beta.cdf(raw_u, pop_alpha, pop_beta), 0.01, 0.99))

    # Exponential recall decay: p(dt) = p_0 * exp(-gamma * dt)
    eff_recall = float(base_recall_prob * np.exp(-decay_rate * max(years_since_breakup, 0.0)))

    contributions = {k: float((scores[k] / 10.0) ** w_vec[idx]) for idx, k in enumerate(keys)}

    return BenchmarkProfile(
        raw_cobb_douglas=raw_u,
        calibrated_percentile=theta,
        effective_recall_prob=eff_recall,
        dimension_contributions=contributions,
    )

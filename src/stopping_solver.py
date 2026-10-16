import numpy as np
from dataclasses import dataclass


@dataclass
class StoppingSolution:
    prob_loml_in_past_ordinal: float
    prob_loml_in_past_calibrated: float
    prob_loml_in_future_calibrated: float
    ordinal_policy_win_prob: float
    bellman_expected_utility: float
    reservation_thresholds: np.ndarray
    effective_commit_thresholds: np.ndarray


def solve_optimal_stopping(
    n_past: int,
    n_future: int,
    theta_benchmark: float,
    effective_recall_prob: float = 0.0,
) -> StoppingSolution:
    """
    Computes exact probabilities of having met the LOML and solves the
    Bellman backward-induction recursion for future candidate thresholds.
    """
    n_total = n_past + n_future
    if n_total <= 0:
        raise ValueError("Total candidates (n_past + n_future) must be positive.")

    # 1. Blind Ordinal Prior: P(Best in past) = n_past / N
    prob_past_ordinal = n_past / n_total

    # 2. Informed Calibrated Prior: P(All n_future candidates < theta_benchmark) = theta^n_future
    prob_past_calibrated = float(theta_benchmark ** n_future)
    prob_future_calibrated = 1.0 - prob_past_calibrated

    # 3. Classical Secretary Win Probability from current state (ordinal policy)
    if n_future == 0:
        ordinal_win = effective_recall_prob * prob_past_ordinal
    elif n_past == 0:
        # Optimal cutoff r* for full horizon N
        r_star = max(1, int(np.round(n_total / np.e)))
        harmonic_sum = sum(1.0 / (j - 1) for j in range(r_star + 1, n_total + 1))
        ordinal_win = (r_star / n_total) * harmonic_sum
    else:
        harmonic_sum = sum(1.0 / (j - 1) for j in range(n_past + 1, n_total + 1))
        direct_future_win = (n_past / n_total) * harmonic_sum
        recall_rescue = effective_recall_prob * prob_past_ordinal
        ordinal_win = min(1.0, direct_future_win + recall_rescue)

    # 4. Bellman Backward Induction for Full-Information Reservation Thresholds
    # V[k] = maximal expected percentile when k future candidates remain
    # Terminal fallback V[0] = expected value of exercising the recall option
    V = np.zeros(n_future + 1)
    V[0] = effective_recall_prob * theta_benchmark

    for k in range(1, n_future + 1):
        v_prev = V[k - 1]
        # For X ~ Uniform(0, 1), E[max(X, v_prev)] = (1 + v_prev^2) / 2
        V[k] = 0.5 * (1.0 + v_prev ** 2)

    # Reservation threshold at future candidate step j (1 <= j <= n_future)
    # is the continuation value V[n_future - j]
    if n_future > 0:
        reservation_thresholds = V[:-1][::-1]
        # If the user strictly refuses anyone worse than theta_benchmark except at the end:
        effective_commit = np.maximum(reservation_thresholds, theta_benchmark * 0.85)
    else:
        reservation_thresholds = np.array([V[0]])
        effective_commit = np.array([theta_benchmark])

    return StoppingSolution(
        prob_loml_in_past_ordinal=float(prob_past_ordinal),
        prob_loml_in_past_calibrated=prob_past_calibrated,
        prob_loml_in_future_calibrated=prob_future_calibrated,
        ordinal_policy_win_prob=float(ordinal_win),
        bellman_expected_utility=float(V[n_future]),
        reservation_thresholds=reservation_thresholds,
        effective_commit_thresholds=effective_commit,
    )

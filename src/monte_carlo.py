import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationSummary:
    optimal_win_rate: float
    benchmark_win_rate: float
    impulsive_win_rate: float
    optimal_top2_rate: float
    chosen_percentiles_optimal: np.ndarray
    chosen_percentiles_benchmark: np.ndarray


def run_monte_carlo_simulations(
    n_past: int,
    n_future: int,
    theta_benchmark: float,
    reservation_thresholds: np.ndarray,
    effective_recall_prob: float = 0.0,
    n_sims: int = 10_000,
    seed: int = 42,
) -> SimulationSummary:
    """
    Simulates n_sims candidate sequences to evaluate empirical performance of stopping policies.
    """
    rng = np.random.default_rng(seed)

    if n_future <= 0:
        recalled = rng.random(n_sims) < effective_recall_prob
        scores = np.where(recalled, theta_benchmark, 0.0)
        return SimulationSummary(
            optimal_win_rate=float(np.mean(recalled)),
            benchmark_win_rate=float(np.mean(recalled)),
            impulsive_win_rate=0.0,
            optimal_top2_rate=float(np.mean(recalled)),
            chosen_percentiles_optimal=scores,
            chosen_percentiles_benchmark=scores,
        )

    # Draw future candidates ~ Uniform(0, 1), shape (n_sims, n_future)
    future_draws = rng.uniform(0.0, 1.0, size=(n_sims, n_future))
    recall_success = rng.random(n_sims) < effective_recall_prob

    # True global max across past benchmark and all future draws
    global_max = np.maximum(theta_benchmark, np.max(future_draws, axis=1))

    # Compute second-highest score for "Top 2" safety metric
    all_candidates = np.column_stack([np.full(n_sims, theta_benchmark), future_draws])
    sorted_all = np.sort(all_candidates, axis=1)
    second_best = sorted_all[:, -2] if all_candidates.shape[1] >= 2 else global_max

    # Strategy 1: Dynamic Bellman Optimal Policy
    chosen_opt = np.zeros(n_sims)
    decided_opt = np.zeros(n_sims, dtype=bool)
    for j in range(n_future):
        thresh = reservation_thresholds[j]
        accept = (~decided_opt) & (future_draws[:, j] >= thresh)
        chosen_opt[accept] = future_draws[accept, j]
        decided_opt |= accept

    # Unmatched at terminal step try recall; if recall fails, settle on last candidate
    unmatched = ~decided_opt
    chosen_opt[unmatched] = np.where(
        recall_success[unmatched], theta_benchmark, future_draws[unmatched, -1]
    )

    # Strategy 2: Strict Benchmark Policy (only accept if > theta_benchmark)
    chosen_bench = np.zeros(n_sims)
    decided_bench = np.zeros(n_sims, dtype=bool)
    for j in range(n_future):
        accept = (~decided_bench) & (future_draws[:, j] > theta_benchmark)
        chosen_bench[accept] = future_draws[accept, j]
        decided_bench |= accept

    unmatched_b = ~decided_bench
    chosen_bench[unmatched_b] = np.where(
        recall_success[unmatched_b], theta_benchmark, future_draws[unmatched_b, -1]
    )

    # Strategy 3: Impulsive Policy (always take very next candidate C_{n_past + 1})
    chosen_impulsive = future_draws[:, 0]

    opt_win = float(np.mean(np.isclose(chosen_opt, global_max)))
    bench_win = float(np.mean(np.isclose(chosen_bench, global_max)))
    imp_win = float(np.mean(np.isclose(chosen_impulsive, global_max)))
    opt_top2 = float(np.mean(chosen_opt >= second_best - 1e-9))

    return SimulationSummary(
        optimal_win_rate=opt_win,
        benchmark_win_rate=bench_win,
        impulsive_win_rate=imp_win,
        optimal_top2_rate=opt_top2,
        chosen_percentiles_optimal=chosen_opt,
        chosen_percentiles_benchmark=chosen_bench,
    )

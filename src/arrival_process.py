import numpy as np
from scipy.stats import beta
from dataclasses import dataclass


@dataclass
class ArrivalMetrics:
    ages: np.ndarray
    intensity: np.ndarray
    cumulative_candidates: np.ndarray
    effective_elapsed_fraction: float
    expected_past_candidates: float
    expected_future_candidates: float


class NonHomogeneousPoissonArrival:
    """
    Models romantic candidate arrivals via a Non-Homogeneous Poisson Process (NHPP)
    with a Beta-distributed intensity curve over [a_min, a_max].
    """

    def __init__(
        self,
        a_min: float = 16.0,
        a_max: float = 28.0,
        a_peak: float = 21.0,
        concentration: float = 4.0,
        total_expected_candidates: float = 6.0,
    ):
        if not (a_min < a_peak < a_max):
            raise ValueError("Require a_min < a_peak < a_max.")
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.a_peak = float(a_peak)
        self.concentration = float(concentration)
        self.total_expected = float(total_expected_candidates)

        # Map peak age to Beta(alpha, beta) mode = (alpha - 1) / (alpha + beta - 2)
        mode = np.clip((self.a_peak - self.a_min) / (self.a_max - self.a_min), 0.05, 0.95)
        self.alpha = 1.0 + mode * self.concentration
        self.beta_param = 1.0 + (1.0 - mode) * self.concentration

    def _normalize_age(self, age: np.ndarray | float) -> np.ndarray | float:
        span = self.a_max - self.a_min
        return np.clip((age - self.a_min) / span, 0.0, 1.0)

    def intensity_at(self, age: np.ndarray | float) -> np.ndarray | float:
        """Instantaneous arrival rate lambda(t) at a given age."""
        span = self.a_max - self.a_min
        u = self._normalize_age(age)
        return self.total_expected * beta.pdf(u, self.alpha, self.beta_param) / span

    def cumulative_fraction(self, age: np.ndarray | float) -> np.ndarray | float:
        """Fraction of total lifetime candidates met by a given age."""
        u = self._normalize_age(age)
        return beta.cdf(u, self.alpha, self.beta_param)

    def evaluate_timeline(self, current_age: float, num_points: int = 200) -> ArrivalMetrics:
        ages = np.linspace(self.a_min, self.a_max, num_points)
        intensity = self.intensity_at(ages)
        cum_frac = self.cumulative_fraction(ages)
        cum_candidates = self.total_expected * cum_frac

        eff_fraction = float(self.cumulative_fraction(current_age))
        past_exp = self.total_expected * eff_fraction
        future_exp = self.total_expected * (1.0 - eff_fraction)

        return ArrivalMetrics(
            ages=ages,
            intensity=intensity,
            cumulative_candidates=cum_candidates,
            effective_elapsed_fraction=eff_fraction,
            expected_past_candidates=past_exp,
            expected_future_candidates=future_exp,
        )

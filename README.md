# Optimal Stopping & LOML Probability Engine

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-ff4b4b.svg)
![SciPy](https://img.shields.io/badge/SciPy-Stochastic_Engine-8caa00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive stochastic modeling engine and decision-theory dashboard that estimates the probability of having already met your optimal long-term partner ("Love of My Life" / LOML) and computes dynamic commitment thresholds over a finite romantic horizon.

Moving beyond the textbook **$1/e \approx 36.8\%$ Secretary Problem**, this engine models romantic search as a **full-information Bayesian optimal stopping problem** with **Non-Homogeneous Poisson Process (NHPP)** candidate arrivals, **Cobb-Douglas multi-attribute utility**, and **Bellman dynamic programming** with time-decaying recall.

---

## Table of Contents

1. [Overview & Motivation](#1-overview--motivation)
2. [Mathematical Architecture](#2-mathematical-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation & Quickstart](#4-installation--quickstart)
5. [Interactive Simulation & Notebooks](#5-interactive-simulation--notebooks)
6. [Deployment](#6-deployment)
7. [References](#7-references)

---

## 1. Overview & Motivation

The classical Secretary Problem makes four restrictive assumptions that rarely hold in human relationships:
1. Candidates arrive at a constant, uniform rate over time.
2. You can only observe relative ordinal ranks ($1, 2, \dots, N$) rather than calibrated compatibility scores.
3. Passing on a candidate is permanent (**strictly zero recall**).
4. Utility is one-dimensional, ignoring the difference between short-term intensity and structural durability.

**`optimal-stopping-loml`** relaxes each of these constraints into a four-layer mathematical pipeline that users can parameterize with their own demographic timelines and relationship benchmarks.

---

## 2. Mathematical Architecture

### 2.1 Classical Ordinal Baseline & Asymptotic Limit
For a finite sequence of $N = n_{\text{past}} + n_{\text{future}}$ exchangeable candidates where the first $r = n_{\text{past}}$ candidates establish the benchmark $M_r$, the exact probability $\Phi(r, N)$ of selecting the global rank-1 candidate is:

$$\Phi(r, N) = \frac{r}{N} \sum_{j=r+1}^{N} \frac{1}{j-1}$$

In the continuous limit as $N \to \infty$ with exploration fraction $x = r/N$:

$$\lim_{N \to \infty} \Phi(r, N) = x \int_{x}^{1} \frac{1}{t}\,dt = -x \ln x \implies x^* = \frac{1}{e} \approx 36.8\%$$

---

### 2.2 Non-Homogeneous Poisson Arrival Process $\lambda(t)$
Social exposure is rarely uniform between ages $A_{\min}$ and $A_{\max}$; candidate encounters typically peak around university and early career transitions ($A_{\text{peak}}$) before tapering. We model candidate arrivals via a Beta-distributed intensity function $\lambda(t)$:

$$\lambda(t) = \frac{N_{\text{total}}}{A_{\max} - A_{\min}} f_{\text{Beta}}\left(\frac{t - A_{\min}}{A_{\max} - A_{\min}}; \alpha, \beta\right)$$

By the time-change theorem for Poisson processes, chronological age $t$ maps onto normalized operational time $\tau(t) \in [0, 1]$:

$$\tau(t) = \frac{\Lambda(t)}{\Lambda(A_{\max})} = \frac{\int_{A_{\min}}^{t} \lambda(u)\,du}{\int_{A_{\min}}^{A_{\max}} \lambda(u)\,du} = F_{\text{Beta}}\left(\frac{t - A_{\min}}{A_{\max} - A_{\min}}; \alpha, \beta\right)$$

This yields the exact chronological age $t^*$ at which the exploration phase ends:

$$t^* = A_{\min} + (A_{\max} - A_{\min}) F_{\text{Beta}}^{-1}\left(\frac{1}{e}; \alpha, \beta\right)$$

---

### 2.3 Cobb-Douglas Multi-Attribute Utility (Solving the "Peak Trap")
Evaluating a past partner using a linear additive score ($U = \sum w_k s_k$) exhibits infinite elasticity of substitution ($\sigma = \infty$), allowing an extreme spike in a single trait (e.g., *Devotion*) to mask a complete collapse in *Structural Durability*.

We evaluate benchmark candidates across $K = 4$ structural dimensions (*Devotion*, *Reciprocity*, *Structural Durability*, and *Shared Life Goals*) using a multiplicative **Cobb-Douglas utility function**:

$$U(\mathbf{s}) = \prod_{k=1}^{K} \left(\frac{s_k}{10}\right)^{w_k}, \quad \sum_{k=1}^{K} w_k = 1$$

* **Boundary Annihilation:** $\lim_{s_k \to 0^+} U(\mathbf{s}) = 0$. If relationship durability or mutual reciprocity collapses, overall utility approaches zero regardless of peak intensity.
* **Population Calibration:** Raw utility $U(\mathbf{s})$ is mapped onto a population prior $\text{Beta}(\alpha_0, \beta_0)$ to extract the calibrated benchmark percentile $\theta \in (0, 1)$.

Given $\theta$ and $n_{\text{future}}$ remaining draws, the informed Bayesian probability that at least one future candidate surpasses your benchmark is:

$$\mathbb{P}\left(\max_{1 \le j \le n_{\text{future}}} X_j > \theta\right) = 1 - \theta^{n_{\text{future}}}$$

---

### 2.4 Full-Information Bellman Recursion with Time-Decaying Recall
When a past benchmark $\theta$ remains accessible with base reconnection probability $p_0$ decaying exponentially over $\Delta t$ years since the breakup:

$$p(\Delta t) = p_0 \, e^{-\gamma \Delta t}$$

the terminal fallback value at horizon exhaustion ($k = 0$ future candidates remaining) is $V_0 = p(\Delta t) \cdot \theta$. By Bellman's Principle of Optimality, for $X \sim \text{Uniform}(0, 1)$, the expected value $V_k$ with $k$ candidates remaining satisfies the Moser–Cayley quadratic recursion:

$$V_k = \mathbb{E}\big[\max(X, V_{k-1})\big] = \int_{0}^{V_{k-1}} V_{k-1}\,dx + \int_{V_{k-1}}^{1} x\,dx = \frac{1 + V_{k-1}^2}{2}$$

At step $k$, the optimal policy commits to the current candidate if and only if their percentile score clears the continuation threshold $X \ge V_{k-1}$.

---

## 3. Repository Structure

```text
optimal-stopping-loml/
├── README.md                      # Project documentation and mathematical specification
├── requirements.txt               # Python dependencies
├── app.py                         # Interactive Streamlit dashboard and Plotly visualizers
├── src/
│   ├── __init__.py
│   ├── arrival_process.py         # NHPP Beta density & operational time transformations
│   ├── utility_model.py           # Cobb-Douglas multi-attribute utility & recall decay
│   ├── stopping_solver.py         # Analytical Bayesian probabilities & Bellman solver
│   └── monte_carlo.py             # 10,000-trajectory vectorized NumPy policy simulator
└── notebooks/
    └── math_derivations.ipynb     # Formal proofs, harmonic limits, and phase-space plots
```

---

## 4. Installation & Quickstart

### Prerequisites
* Python 3.11 or higher
* `pip` or `conda` environment manager

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/optimal-stopping-loml.git
   cd optimal-stopping-loml
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   The interactive web application will open automatically at `http://localhost:8501`.

---

## 5. Interactive Simulation & Notebooks

To explore the formal mathematical proofs and generate static publication-grade phase-space plots:

```bash
jupyter notebook notebooks/math_derivations.ipynb
```

The companion notebook includes:
* Finite-horizon harmonic curves $\Phi(r, N)$ vs. the continuous $-x \ln x$ limit.
* Instantaneous arrival curves $\lambda(t)$ and cumulative operational time $\tau(t)$ across social peak ages.
* 2D contour phase-space maps of $1 - \theta^{n_{\text{future}}}$ and Bellman continuation trajectories $V_k$.
* Indifference surface comparisons between Linear Additive and Cobb-Douglas Multiplicative utility models.

---

## 6. Deployment

This repository is ready for zero-configuration deployment on **Streamlit Community Cloud**:

1. Push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select `<your-username>/optimal-stopping-loml`, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

---

## 7. References

1. **Lindley, D. V. (1961).** *Dynamic Programming and Decision Theory.* Applied Statistics, 10(1), 39–51.
2. **Gilbert, J. P., & Mosteller, F. (1966).** *Recognizing the Maximum of a Sequence.* Journal of the American Statistical Association, 61(313), 35–73.
3. **Yang, M. C. K. (1974).** *Recognizing the Maximum of a Random Sequence Based on Relative Rank with Backward Solicitation.* Journal of Applied Probability, 11(3), 504–512.
4. **Ferguson, T. S. (1989).** *Who Solved the Secretary Problem?* Statistical Science, 4(3), 282–289.
5. **Fry, H. (2015).** *The Mathematics of Love: Patterns, Proofs, and the Search for the Ultimate Equation.* Simon & Schuster.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

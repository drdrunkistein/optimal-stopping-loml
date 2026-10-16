import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.arrival_process import NonHomogeneousPoissonArrival
from src.utility_model import compute_cobb_douglas_benchmark
from src.stopping_solver import solve_optimal_stopping
from src.monte_carlo import run_monte_carlo_simulations

st.set_page_config(
    page_title="Optimal Stopping & LOML Probability Engine",
    layout="wide",
)

st.title("Optimal Stopping & LOML Probability Engine")
st.caption(
    "A stochastic modeling tool combining Non-Homogeneous Poisson Arrivals, "
    "Cobb-Douglas Multi-Attribute Utility, and Bellman Dynamic Programming."
)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Timeline & Exposure Priors")
    a_min = st.slider("Age Started Dating (A_min)", 14, 30, 16)
    a_peak = st.slider("Peak Social Exposure Age", a_min + 1, 32, max(a_min+1,21))
    current_age = st.slider("Current Age", a_min , 35, max(a_min, 22))  
    a_max = st.slider("Target Settling Age (A_max)", a_peak+1, 40, max(a_peak+1,27))

    st.header("2. Candidate Counts")
    n_past = st.number_input("Serious Past Candidates (n_past)", min_value=1, max_value=15, value=1)
    n_future = st.number_input("Expected Future Candidates (n_future)", min_value=1, max_value=15, value=2)

    st.header("3. Best Past Partner Benchmark")
    st.markdown("Rate your top past connection across 4 structural dimensions:")
    s_devotion = st.slider("Devotion & Faith in You", 1.0, 10.0, 9.5, 0.5)
    s_reciprocity = st.slider("Your Reciprocal Feelings", 1.0, 10.0, 6.0, 0.5)
    s_durability = st.slider("Structural Durability & Trust", 1.0, 10.0, 3.5, 0.5)
    s_goals = st.slider("Shared Life Trajectory", 1.0, 10.0, 7.0, 0.5)

    st.header("4. Recall Parameters")
    allow_recall = st.checkbox("Recall possible for best past match?", value=False)
    if allow_recall:
        base_recall = st.slider("Base Reconnection Probability (p_0)", 0.05, 0.95, 0.50, 0.05)
        years_elapsed = st.slider("Years Since Breakup", 0.0, 10.0, 1.5, 0.5)
    else:
        base_recall, years_elapsed = 0.0, 1.0

# --- Run Mathematical Engine ---
scores = {
    "Devotion": s_devotion,
    "Reciprocity": s_reciprocity,
    "Durability": s_durability,
    "Shared Goals": s_goals,
}
weights = {"Devotion": 0.20, "Reciprocity": 0.30, "Durability": 0.35, "Shared Goals": 0.15}

benchmark = compute_cobb_douglas_benchmark(
    scores=scores,
    weights=weights,
    base_recall_prob=base_recall,
    years_since_breakup=years_elapsed,
)

nhpp = NonHomogeneousPoissonArrival(
    a_min=a_min,
    a_max=a_max,
    a_peak=a_peak,
    total_expected_candidates=n_past + n_future,
)
timeline = nhpp.evaluate_timeline(current_age=current_age)

solution = solve_optimal_stopping(
    n_past=int(n_past),
    n_future=int(n_future),
    theta_benchmark=benchmark.calibrated_percentile,
    effective_recall_prob=benchmark.effective_recall_prob,
)

sim = run_monte_carlo_simulations(
    n_past=int(n_past),
    n_future=int(n_future),
    theta_benchmark=benchmark.calibrated_percentile,
    reservation_thresholds=solution.reservation_thresholds,
    effective_recall_prob=benchmark.effective_recall_prob,
)

# --- Top KPI Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Calibrated Benchmark (θ)",
    f"{benchmark.calibrated_percentile * 100:.1f}th pct",
    f"Raw Cobb-Douglas: {benchmark.raw_cobb_douglas:.2f}",
)
col2.metric(
    "P(LOML Already Met | θ)",
    f"{solution.prob_loml_in_past_calibrated * 100:.1f}%",
    f"Blind Prior: {solution.prob_loml_in_past_ordinal * 100:.1f}%",
)
col3.metric(
    "P(Best Match Still Ahead)",
    f"{solution.prob_loml_in_future_calibrated * 100:.1f}%",
    f"Timeline Elapsed: {timeline.effective_elapsed_fraction * 100:.1f}%",
)
col4.metric(
    "Optimal Policy Win Rate",
    f"{sim.optimal_win_rate * 100:.1f}%",
    f"Top-2 Partner Odds: {sim.optimal_top2_rate * 100:.1f}%",
)

st.divider()

# --- Visualizations ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Non-Homogeneous Social Arrival Curve λ(t)")
    fig_arr = go.Figure()
    fig_arr.add_trace(
        go.Scatter(
            x=timeline.ages,
            y=timeline.intensity,
            mode="lines",
            fill="tozeroy",
            name="Arrival Rate λ(t)",
        )
    )
    fig_arr.add_vline(
        x=current_age,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current Age ({current_age})",
    )
    fig_arr.update_layout(
        xaxis_title="Age",
        yaxis_title="Candidate Arrival Density",
        height=380,
    )
    st.plotly_chart(fig_arr, use_container_width=True)

with right_col:
    st.subheader("Dynamic Bellman Reservation Thresholds")
    steps = [f"Future Candidate #{j+1}" for j in range(int(n_future))]
    fig_bell = go.Figure()
    fig_bell.add_trace(
        go.Bar(
            x=steps,
            y=solution.reservation_thresholds * 100,
            name="Bellman Minimum Percentile",
        )
    )
    fig_bell.add_hline(
        y=benchmark.calibrated_percentile * 100,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Past Benchmark θ ({benchmark.calibrated_percentile*100:.1f}%)",
    )
    fig_bell.update_layout(
        xaxis_title="Remaining Candidates Sequence",
        yaxis_title="Required Percentile to Commit (%)",
        yaxis_range=[0, 100],
        height=380,
    )
    st.plotly_chart(fig_bell, use_container_width=True) 

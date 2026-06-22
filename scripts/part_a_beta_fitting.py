"""
=================================================================
  PART A: CORRECTED INVERSE GAUSSIAN SHAPE PARAMETER
=================================================================
  Reproduces the β(p,δ) fit from Monte Carlo simulations.

  Model (discrete buffer system):
      I(t+1) = I(t) + Y(t) - D(t)
      with prob (1-p): Y=C, D~Poisson(λ)         (operational)
      with prob p:    Y=0, D~Poisson(λ)          (disruptive)
      collapse at first t with I(t) ≤ 0.

  Fit:  σ²_eff = λ + β(p,δ) · p(1-p)C²
        β(p,δ) = a · p^(-(b + c·δ))
        Expected: a≈0.2404, b≈0.4087, c≈0.01121, R²≈0.9523

  Outputs:
    data/part_a_simulations/dataset_grid_expanded.csv
    data/part_a_simulations/dataset_delta_fixed.csv
    data/part_a_simulations/combined_42pts_validity.csv
    results_part_a/beta_fit_summary.csv
    results_part_a/fig_beta_diagnostic.pdf
    results_part_a/fig_beta_residuals.pdf

  Run:  python part_a_beta_fitting.py
  Time: ~30-45 min on modern laptop (post-warmup Numba)
=================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, prange
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3

# =================================================================
# CONFIG
# =================================================================

RNG_SEED = 20260429
N_RUNS = 15000
MAX_STEPS = 300000
MIN_COLLAPSED = 500
DELTA_MIN = 0.5
DELTA_MAX = 20.0

# Validity domain for fitting
P_MAX_VALIDITY = 0.35
DELTA_MAX_VALIDITY = 15.0

# Base parameters
C_DEFAULT = 100
X0_DEFAULT = 50

# Output directories
DATA_DIR = "./data/part_a_simulations"
RESULTS_DIR = "./results_part_a"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =================================================================
# SECTION A -- SIMULATOR (JIT)
# =================================================================

@njit(cache=True)
def simulate_single_fpt(X0, C, lam, p_disrupt, max_steps):
    """
    Simulate one trajectory and return first-passage time to 0.
    
    Returns:
        t > 0  : collapse time
        -1     : did not collapse within max_steps
    """
    buffer = X0
    for t in range(1, max_steps + 1):
        D = np.random.poisson(lam)
        if np.random.random() > p_disrupt:
            buffer = buffer + C - D    # operational
        else:
            buffer = buffer - D         # disruptive
        if buffer <= 0:
            return t
    return -1


@njit(cache=True, parallel=True)
def simulate_n_fpt(X0, C, lam, p_disrupt, n_runs, max_steps, seed):
    """Run n_runs independent simulations. Parallelized over runs."""
    np.random.seed(seed)
    out = np.full(n_runs, -1, dtype=np.int64)
    for i in prange(n_runs):
        # Each thread needs its own seed sequence; we approximate
        # by jittering the base seed by i. Numba's prange does not
        # share np.random state cleanly across threads; this is
        # acceptable for MLE estimation purposes given large n_runs.
        np.random.seed(seed + i)
        out[i] = simulate_single_fpt(X0, C, lam, p_disrupt, max_steps)
    return out


# =================================================================
# SECTION B -- MLE FOR INVERSE GAUSSIAN
# =================================================================

def fit_ig_mle(tau_samples):
    """
    Closed-form MLE for Inverse Gaussian.
    
    Inputs:
        tau_samples : array of hitting times. Values <= 0 are treated
                      as right-censored (did not collapse) and removed.
    
    Returns:
        (mu_hat, eta_hat, n_used)
        mu_hat  = mean(tau)
        eta_hat = 1 / mean(1/tau - 1/mu_hat)
    """
    data = tau_samples[tau_samples > 0]
    n = len(data)
    if n < 10:
        return np.nan, np.nan, n
    mu_hat = float(np.mean(data))
    inv_mean = float(np.mean(1.0 / data))
    denom = inv_mean - 1.0 / mu_hat
    if denom <= 0:
        return mu_hat, np.nan, n
    eta_hat = 1.0 / denom
    return mu_hat, eta_hat, n


# =================================================================
# SECTION C -- GRID GENERATION
# =================================================================

def generate_grid_expanded(C=C_DEFAULT):
    """
    Expanded grid: cartesian product p × λ/C, filtered.
    
    Filters:
      - δ ∈ [DELTA_MIN, DELTA_MAX]
      - λ in (0, C)
    
    Returns list of (p, lam, C, label) tuples.
    Final count after Monte Carlo filtering (n_collapsed >= 500): ~47.
    """
    p_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35,
              0.40, 0.45, 0.50]
    util_list = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                 0.70, 0.75, 0.80, 0.85, 0.90]
    grid = []
    for p in p_list:
        for u in util_list:
            lam = u * C
            if lam <= 0 or lam >= C:
                continue
            delta = (1 - p) * C - lam
            if delta < DELTA_MIN or delta > DELTA_MAX:
                continue
            grid.append({
                "p": p, "lam": lam, "C": C, "delta": delta,
                "util": u, "source": "expanded",
                "label": f"exp_p{p:.2f}_u{u:.2f}",
            })
    return grid


def generate_grid_delta_fixed(C=C_DEFAULT):
    """
    δ-fixed grid: for each δ ∈ {2,5,10} and each p, compute λ.
    
    Filters:
      - 0 < λ < C
    
    Returns list of dicts.
    Expected count: 39 (13 per δ).
    """
    deltas = [2, 5, 10]
    p_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]
    grid = []
    for d in deltas:
        for p in p_list:
            lam = (1 - p) * C - d
            if lam <= 0 or lam >= C:
                continue
            grid.append({
                "p": p, "lam": lam, "C": C, "delta": float(d),
                "util": lam / C, "source": "delta_fixed",
                "label": f"d{d}_p{p:.2f}",
            })
    return grid


# =================================================================
# SECTION D -- RUN MONTE CARLO GRID
# =================================================================

def run_simulation_grid(grid, X0=X0_DEFAULT, n_runs=N_RUNS,
                         max_steps=MAX_STEPS, seed=RNG_SEED, verbose=True):
    """
    For each config in grid:
      1. Simulate n_runs trajectories.
      2. MLE fit IG (μ_hat, η_hat).
      3. Compute theoretical η and implied β.
    
    Returns DataFrame with one row per config.
    """
    rows = []
    n_total = len(grid)
    t_start = time.time()
    for idx, cfg in enumerate(grid):
        p = cfg["p"]
        lam = cfg["lam"]
        C = cfg["C"]
        delta = cfg["delta"]
        # Use a per-config seed offset for variety while staying reproducible
        seed_cfg = seed + idx * 1000
        t0 = time.time()
        tau = simulate_n_fpt(X0, C, lam, p, n_runs, max_steps, seed_cfg)
        n_collapsed = int(np.sum(tau > 0))
        n_censored = n_runs - n_collapsed
        mu_hat, eta_hat, n_used = fit_ig_mle(tau)
        # Theoretical IG (without correction)
        sigma_sq_marginal = p * (1 - p) * C ** 2 + lam
        eta_theory = X0 ** 2 / sigma_sq_marginal
        mu_theory = X0 / delta if delta > 0 else np.nan
        # Implied β: from σ²_eff = λ + β·p(1-p)C² = X0² / η_hat
        # => β = (X0²/η_hat - λ) / (p(1-p)C²)
        if eta_hat > 0 and not np.isnan(eta_hat):
            sigma_sq_eff_implied = X0 ** 2 / eta_hat
            denom_var = p * (1 - p) * C ** 2
            if denom_var > 0:
                beta_implied = (sigma_sq_eff_implied - lam) / denom_var
            else:
                beta_implied = np.nan
        else:
            sigma_sq_eff_implied = np.nan
            beta_implied = np.nan
        eta_ratio = eta_hat / eta_theory if eta_theory > 0 else np.nan
        dt = time.time() - t0
        if verbose:
            print(f"  [{idx+1}/{n_total}] {cfg['label']:25s} "
                  f"p={p:.2f} u={lam/C:.2f} d={delta:5.2f}: "
                  f"n_coll={n_collapsed:5d}/{n_runs} "
                  f"μ_hat={mu_hat:7.1f} η_hat={eta_hat:7.2f} "
                  f"β_impl={beta_implied:6.3f} ({dt:.1f}s)")
        rows.append({
            "label": cfg["label"], "source": cfg["source"],
            "p": p, "lam": lam, "C": C, "X0": X0, "delta": delta,
            "util": lam / C,
            "n_runs": n_runs, "n_collapsed": n_collapsed,
            "n_censored": n_censored,
            "mu_hat": mu_hat, "eta_hat": eta_hat,
            "mu_theory": mu_theory, "eta_theory": eta_theory,
            "eta_ratio": eta_ratio,
            "sigma_sq_marginal": sigma_sq_marginal,
            "sigma_sq_eff_implied": sigma_sq_eff_implied,
            "beta_implied": beta_implied,
        })
    elapsed = time.time() - t_start
    if verbose:
        print(f"  Total Monte Carlo time: {elapsed/60:.1f} min")
    return pd.DataFrame(rows)


# =================================================================
# SECTION E -- COMBINE AND FILTER DATASETS
# =================================================================

def filter_and_combine(df_expanded, df_delta_fixed,
                        min_collapsed=MIN_COLLAPSED,
                        p_max=P_MAX_VALIDITY, delta_max=DELTA_MAX_VALIDITY):
    """
    Filter by sufficient collapses, combine, dedup, restrict to validity.
    
    Returns:
        df_combined        : all valid points (no validity filter)
        df_validity        : only points with p<=0.35 and δ<=15
    """
    # Filter each dataset by data quality
    df_e = df_expanded[df_expanded["n_collapsed"] >= min_collapsed].copy()
    df_d = df_delta_fixed[df_delta_fixed["n_collapsed"] >= min_collapsed].copy()
    print(f"\n  After n_collapsed >= {min_collapsed}:")
    print(f"    expanded: {len(df_e)}/{len(df_expanded)}")
    print(f"    delta_fixed: {len(df_d)}/{len(df_delta_fixed)}")
    # Combine and dedup by (p, lam) rounded
    df_all = pd.concat([df_e, df_d], ignore_index=True)
    df_all["dedup_key"] = (
        df_all["p"].round(4).astype(str) + "_" +
        df_all["lam"].round(4).astype(str)
    )
    n_before = len(df_all)
    df_combined = df_all.drop_duplicates(subset="dedup_key", keep="first").copy()
    df_combined = df_combined.drop(columns=["dedup_key"])
    print(f"    combined (after dedup): {len(df_combined)}/{n_before}")
    # Apply validity filter
    df_validity = df_combined[
        (df_combined["p"] <= p_max) &
        (df_combined["delta"] <= delta_max)
    ].copy()
    print(f"    validity (p<={p_max}, δ<={delta_max}): {len(df_validity)}")
    return df_combined, df_validity


# =================================================================
# SECTION F -- FIT BETA FORMULA
# =================================================================

def beta_model(X, a, b, c):
    """β(p,δ) = a · p^(-(b + c·δ))"""
    p, delta = X
    return a * np.power(p, -(b + c * delta))


def fit_beta_formula(df_validity, p0=(0.24, 0.41, 0.011)):
    """
    Nonlinear least squares fit of β = a · p^(-(b + c·δ)).
    
    Uses only points with valid β_implied (positive, finite).
    
    Returns dict with coefficients, R², residuals, predictions.
    """
    df = df_validity.dropna(subset=["beta_implied"]).copy()
    df = df[df["beta_implied"] > 0]
    p_arr = df["p"].values
    d_arr = df["delta"].values
    y = df["beta_implied"].values
    X = (p_arr, d_arr)
    try:
        popt, pcov = curve_fit(
            beta_model, X, y, p0=p0,
            maxfev=20000,
            bounds=([1e-4, 1e-4, -1.0], [10.0, 5.0, 1.0]),
        )
    except Exception as e:
        print(f"  curve_fit failed: {e}")
        return None
    a, b, c = popt
    y_pred = beta_model(X, *popt)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Compute η-level error
    df["beta_pred"] = y_pred
    df["beta_residual"] = residuals
    sigma_sq_eff_pred = df["lam"] + y_pred * df["p"] * (1 - df["p"]) * df["C"] ** 2
    df["eta_pred"] = df["X0"] ** 2 / sigma_sq_eff_pred
    df["err_eta_pct"] = 100.0 * np.abs(df["eta_pred"] - df["eta_hat"]) / df["eta_hat"]
    summary = {
        "a": float(a), "b": float(b), "c": float(c),
        "R2": float(r2),
        "n_points": int(len(df)),
        "err_eta_mean": float(df["err_eta_pct"].mean()),
        "err_eta_max": float(df["err_eta_pct"].max()),
        "err_eta_median": float(df["err_eta_pct"].median()),
        "param_cov_diag": [float(x) for x in np.diag(pcov)],
    }
    return summary, df


def report_subgroup_errors(df_fit):
    """Print error breakdown by p and δ subgroups (paper Table 3)."""
    print("\n  Error breakdown by p:")
    for p_lo, p_hi in [(0, 0.10), (0.10, 0.20), (0.20, 0.35)]:
        sub = df_fit[(df_fit["p"] > p_lo) & (df_fit["p"] <= p_hi)]
        if len(sub) > 0:
            print(f"    {p_lo:.2f} < p <= {p_hi:.2f}: n={len(sub):2d}, "
                  f"mean={sub['err_eta_pct'].mean():5.2f}%, "
                  f"max={sub['err_eta_pct'].max():5.2f}%")
    print("\n  Error breakdown by δ:")
    for d_lo, d_hi in [(0, 3), (3, 7), (7, 10), (10, 15)]:
        sub = df_fit[(df_fit["delta"] > d_lo) & (df_fit["delta"] <= d_hi)]
        if len(sub) > 0:
            print(f"    {d_lo:2d} < δ <= {d_hi:2d}: n={len(sub):2d}, "
                  f"mean={sub['err_eta_pct'].mean():5.2f}%, "
                  f"max={sub['err_eta_pct'].max():5.2f}%")


# =================================================================
# SECTION G -- DIAGNOSTIC PLOTS
# =================================================================

def plot_beta_diagnostic(df_fit, summary, save=None):
    """β_implied vs β_predicted with ±15% band."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = df_fit["beta_pred"].values
    y = df_fit["beta_implied"].values
    sc = ax.scatter(x, y, c=df_fit["delta"], cmap="viridis",
                     s=60, alpha=0.85, edgecolor="k", linewidth=0.5)
    lo = min(x.min(), y.min()) * 0.9
    hi = max(x.max(), y.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.fill_between([lo, hi], [lo*0.85, hi*0.85], [lo*1.15, hi*1.15],
                     color="gray", alpha=0.15, label="±15%")
    ax.set_xlabel("β predicted (formula)")
    ax.set_ylabel("β implied (from MLE η)")
    ax.set_title(f"β fit: a={summary['a']:.4f}, b={summary['b']:.4f}, "
                 f"c={summary['c']:.5f}\nR²={summary['R2']:.4f}, "
                 f"n={summary['n_points']}")
    plt.colorbar(sc, ax=ax, label="δ")
    ax.legend(fontsize=9)
    ax = axes[1]
    sc = ax.scatter(df_fit["eta_hat"], df_fit["eta_pred"],
                     c=df_fit["p"], cmap="plasma",
                     s=60, alpha=0.85, edgecolor="k", linewidth=0.5)
    lo = min(df_fit["eta_hat"].min(), df_fit["eta_pred"].min()) * 0.9
    hi = max(df_fit["eta_hat"].max(), df_fit["eta_pred"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.fill_between([lo, hi], [lo*0.85, hi*0.85], [lo*1.15, hi*1.15],
                     color="gray", alpha=0.15, label="±15%")
    ax.set_xlabel("η fitted (MLE from simulations)")
    ax.set_ylabel("η predicted (β-corrected)")
    ax.set_title(f"η prediction (mean err={summary['err_eta_mean']:.2f}%, "
                 f"max={summary['err_eta_max']:.2f}%)")
    plt.colorbar(sc, ax=ax, label="p")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return fig


def plot_beta_residuals(df_fit, save=None):
    """Residual plots vs p and δ."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.scatter(df_fit["p"], df_fit["beta_residual"],
               c=df_fit["delta"], cmap="viridis", s=60, alpha=0.85,
               edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("p")
    ax.set_ylabel("β residual (implied - predicted)")
    ax.set_title("Residuals vs p")
    ax = axes[1]
    ax.scatter(df_fit["delta"], df_fit["beta_residual"],
               c=df_fit["p"], cmap="plasma", s=60, alpha=0.85,
               edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("δ")
    ax.set_ylabel("β residual")
    ax.set_title("Residuals vs δ")
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return fig


# =================================================================
# SECTION H -- VALIDATION AGAINST PUBLISHED COEFFICIENTS
# =================================================================

PUBLISHED = {"a": 0.2404, "b": 0.4087, "c": 0.01121, "R2": 0.9523}


def validate_against_published(summary):
    """Print comparison vs published coefficients from prior work."""
    print("\n" + "=" * 60)
    print("VALIDATION AGAINST PUBLISHED COEFFICIENTS")
    print("=" * 60)
    print(f"  Coefficient |  Published  |  This run   |  Δ rel")
    print(f"  ------------|-------------|-------------|---------")
    for key in ["a", "b", "c", "R2"]:
        pub = PUBLISHED[key]
        this = summary[key]
        rel = 100 * abs(this - pub) / abs(pub) if abs(pub) > 0 else np.nan
        print(f"  {key:11s} | {pub:11.4f} | {this:11.4f} | {rel:6.2f}%")
    a_close = abs(summary["a"] - PUBLISHED["a"]) / PUBLISHED["a"] < 0.10
    b_close = abs(summary["b"] - PUBLISHED["b"]) / PUBLISHED["b"] < 0.10
    c_close = abs(summary["c"] - PUBLISHED["c"]) / PUBLISHED["c"] < 0.30
    r2_close = abs(summary["R2"] - PUBLISHED["R2"]) < 0.05
    print()
    print(f"  a within 10%:  {a_close}")
    print(f"  b within 10%:  {b_close}")
    print(f"  c within 30%:  {c_close}  (more sensitive to noise)")
    print(f"  R² within 0.05: {r2_close}")
    if all([a_close, b_close, r2_close]):
        print("\n  >>> REPRODUCTION SUCCESSFUL.")
    else:
        print("\n  >>> Coefficients differ. Possible causes:")
        print("      - Different RNG seed (small effect, ~1-2%)")
        print("      - Different MAX_STEPS or N_RUNS")
        print("      - Different filtering (DELTA_MIN/MAX, MIN_COLLAPSED)")


# =================================================================
# SECTION I -- MAIN
# =================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("PART A: β(p,δ) FITTING")
    print("=" * 70)

    print("\n[1/6] Generating grids...")
    grid_e = generate_grid_expanded()
    grid_d = generate_grid_delta_fixed()
    print(f"  Expanded grid: {len(grid_e)} configs (after δ filter)")
    print(f"  δ-fixed grid:  {len(grid_d)} configs")

    print("\n[2/6] Warming up Numba JIT...")
    _ = simulate_n_fpt(X0_DEFAULT, C_DEFAULT, 70.0, 0.10,
                        100, 10000, RNG_SEED)
    print("  Warmup done.")

    print("\n[3/6] Running Monte Carlo on EXPANDED grid...")
    df_e = run_simulation_grid(grid_e, n_runs=N_RUNS,
                                max_steps=MAX_STEPS, seed=RNG_SEED)
    df_e.to_csv(f"{DATA_DIR}/dataset_grid_expanded.csv", index=False)

    print("\n[4/6] Running Monte Carlo on δ-FIXED grid...")
    df_d = run_simulation_grid(grid_d, n_runs=N_RUNS,
                                max_steps=MAX_STEPS,
                                seed=RNG_SEED + 500000)
    df_d.to_csv(f"{DATA_DIR}/dataset_delta_fixed.csv", index=False)

    print("\n[5/6] Filtering and combining datasets...")
    df_combined, df_validity = filter_and_combine(df_e, df_d)
    df_combined.to_csv(f"{DATA_DIR}/combined_all_valid.csv", index=False)
    df_validity.to_csv(f"{DATA_DIR}/combined_42pts_validity.csv", index=False)

    print("\n[6/6] Fitting β formula...")
    result = fit_beta_formula(df_validity)
    if result is None:
        print("  Fit failed. Aborting.")
        return
    summary, df_fit = result
    print(f"\n  RESULTS:")
    print(f"    a = {summary['a']:.6f}")
    print(f"    b = {summary['b']:.6f}")
    print(f"    c = {summary['c']:.6f}")
    print(f"    R² = {summary['R2']:.6f}")
    print(f"    n_points = {summary['n_points']}")
    print(f"    err_η mean = {summary['err_eta_mean']:.2f}%")
    print(f"    err_η max  = {summary['err_eta_max']:.2f}%")
    print(f"    err_η median = {summary['err_eta_median']:.2f}%")

    report_subgroup_errors(df_fit)

    pd.DataFrame([summary]).to_csv(
        f"{RESULTS_DIR}/beta_fit_summary.csv", index=False
    )
    df_fit.to_csv(f"{RESULTS_DIR}/beta_fit_pointwise.csv", index=False)

    print("\nGenerating diagnostic plots...")
    plot_beta_diagnostic(
        df_fit, summary,
        save=f"{RESULTS_DIR}/fig_beta_diagnostic.pdf"
    )
    plot_beta_residuals(
        df_fit,
        save=f"{RESULTS_DIR}/fig_beta_residuals.pdf"
    )

    validate_against_published(summary)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print(f"Outputs in: {DATA_DIR}/ and {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
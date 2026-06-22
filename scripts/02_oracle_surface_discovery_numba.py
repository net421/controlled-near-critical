import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

# ============================================================
# Oracle Surface Discovery
# Near-Critical Threshold Benchmark
# ============================================================

OUT = "oracle_surface_results"
os.makedirs(OUT, exist_ok=True)

SEED = 20260606

# Use moderate values first. Later increase N_RUNS to 20000 or 50000.
N_RUNS = 5000
T = 365

C = 100
X0 = 50
S_MAX = 180

K_FAIL = 100.0
K_PREVENT = 10.0
HOLDING = 0.02

# Scenario grid
P_VALUES = np.array([
    0.02, 0.05, 0.08, 0.10, 0.12,
    0.15, 0.18, 0.20, 0.25, 0.30
])

LAMBDA_VALUES = np.array([
    50, 55, 60, 65, 70, 75, 78, 80, 82, 84, 86, 88, 90
], dtype=np.float64)

THRESHOLDS = np.arange(0, 80, 5)


def stability_margin(p, lam):
    return (1.0 - p) * C - lam


def sigma2(p, lam):
    return lam + p * (1.0 - p) * C**2


@njit
def simulate_policy_numba(
    p,
    lam,
    threshold,
    n_runs,
    T,
    C,
    X0,
    S_MAX,
    K_FAIL,
    K_PREVENT,
    HOLDING,
    seed,
):
    np.random.seed(seed)

    total_cost_sum = 0.0
    service_numer = 0.0
    service_denom = 0.0
    stockout_run_count = 0.0
    stockouts_sum = 0.0
    preventive_sum = 0.0

    for run in range(n_runs):
        I = X0
        total_cost = 0.0
        stockouts = 0.0
        preventive = 0.0
        served_demand = 0.0
        total_demand = 0.0

        for t in range(T):
            if I <= threshold:
                total_cost += K_PREVENT
                preventive += 1.0
                I = X0

            if np.random.random() < p:
                replenishment = 0
            else:
                replenishment = C

            demand = np.random.poisson(lam)
            total_demand += demand

            before_demand = I + replenishment

            if before_demand >= demand:
                served = demand
            else:
                served = before_demand

            served_demand += served

            I = before_demand - demand

            if I > S_MAX:
                I = S_MAX

            if I <= 0:
                stockouts += 1.0
                total_cost += K_FAIL
                I = X0

            if I > 0:
                total_cost += HOLDING * I

        total_cost_sum += total_cost
        service_numer += served_demand
        service_denom += total_demand
        stockouts_sum += stockouts
        preventive_sum += preventive

        if stockouts > 0:
            stockout_run_count += 1.0

    avg_total_cost = total_cost_sum / n_runs
    service_level = service_numer / service_denom
    stockout_probability = stockout_run_count / n_runs
    avg_stockouts = stockouts_sum / n_runs
    avg_preventive = preventive_sum / n_runs

    return (
        avg_total_cost,
        service_level,
        stockout_probability,
        avg_stockouts,
        avg_preventive,
    )


def run_oracle_surface():
    rows = []
    t0 = time.time()

    scenario_id = 0

    for p in P_VALUES:
        for lam in LAMBDA_VALUES:
            d = stability_margin(p, lam)

            # Only stable systems
            if d <= 0:
                continue

            scenario_id += 1

            best_cost = np.inf
            best_threshold = None
            best_metrics = None

            for th in THRESHOLDS:
                seed = SEED + scenario_id * 1000 + int(th)

                (
                    avg_total_cost,
                    service_level,
                    stockout_probability,
                    avg_stockouts,
                    avg_preventive,
                ) = simulate_policy_numba(
                    p,
                    lam,
                    int(th),
                    N_RUNS,
                    T,
                    C,
                    X0,
                    S_MAX,
                    K_FAIL,
                    K_PREVENT,
                    HOLDING,
                    seed,
                )

                if avg_total_cost < best_cost:
                    best_cost = avg_total_cost
                    best_threshold = int(th)
                    best_metrics = (
                        service_level,
                        stockout_probability,
                        avg_stockouts,
                        avg_preventive,
                    )

            service_level, stockout_probability, avg_stockouts, avg_preventive = best_metrics

            row = {
                "scenario_id": scenario_id,
                "p": p,
                "lambda": lam,
                "delta": d,
                "rho": lam / C,
                "sigma2": sigma2(p, lam),
                "oracle_threshold": best_threshold,
                "oracle_cost": best_cost,
                "service_level": service_level,
                "stockout_probability": stockout_probability,
                "avg_stockouts": avg_stockouts,
                "avg_preventive_actions": avg_preventive,
            }

            rows.append(row)

            print(
                f"scenario={scenario_id:03d} "
                f"p={p:.2f} lam={lam:.1f} delta={d:.2f} "
                f"oracle_s={best_threshold:02d} cost={best_cost:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "oracle_surface.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(df)} scenarios in {elapsed:.1f} seconds.")
    return df


def make_plots(df):
    # Threshold vs delta
    plt.figure(figsize=(9, 6))
    plt.scatter(df["delta"], df["oracle_threshold"], alpha=0.75)
    plt.xlabel("Stability margin delta")
    plt.ylabel("Oracle threshold")
    plt.title("Oracle Threshold vs Stability Margin")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "threshold_vs_delta.png"), dpi=160)
    plt.close()

    # Threshold vs rho
    plt.figure(figsize=(9, 6))
    plt.scatter(df["rho"], df["oracle_threshold"], alpha=0.75)
    plt.xlabel("Utilization rho = lambda / C")
    plt.ylabel("Oracle threshold")
    plt.title("Oracle Threshold vs Utilization")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "threshold_vs_rho.png"), dpi=160)
    plt.close()

    # Threshold vs p
    plt.figure(figsize=(9, 6))
    plt.scatter(df["p"], df["oracle_threshold"], alpha=0.75)
    plt.xlabel("Disruption probability p")
    plt.ylabel("Oracle threshold")
    plt.title("Oracle Threshold vs Disruption Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "threshold_vs_p.png"), dpi=160)
    plt.close()

    # Heatmap-style pivot: p x lambda
    pivot = df.pivot_table(
        index="p",
        columns="lambda",
        values="oracle_threshold",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Oracle threshold")
    plt.xticks(
        np.arange(len(pivot.columns)),
        [str(int(x)) for x in pivot.columns],
        rotation=45,
    )
    plt.yticks(
        np.arange(len(pivot.index)),
        [f"{x:.2f}" for x in pivot.index],
    )
    plt.xlabel("lambda")
    plt.ylabel("p")
    plt.title("Oracle Threshold Surface")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "oracle_threshold_heatmap.png"), dpi=160)
    plt.close()

    # Near-critical subset
    near = df[df["delta"] <= 10].copy()
    if len(near) > 0:
        plt.figure(figsize=(9, 6))
        plt.scatter(near["delta"], near["oracle_threshold"], alpha=0.75)
        plt.xlabel("Stability margin delta")
        plt.ylabel("Oracle threshold")
        plt.title("Oracle Threshold in Near-Critical Regime: delta <= 10")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "threshold_vs_delta_near_critical.png"), dpi=160)
        plt.close()


def fit_simple_rule(df):
    """
    Fit an interpretable linear model:
        s* ≈ a + b log(1 + 1/delta) + c p + d rho

    No sklearn dependency; pure numpy least squares.
    """

    x1 = np.log1p(1.0 / df["delta"].values)
    x2 = df["p"].values
    x3 = df["rho"].values
    y = df["oracle_threshold"].values

    X = np.column_stack([
        np.ones(len(df)),
        x1,
        x2,
        x3,
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    df_rule = df.copy()
    df_rule["rule_threshold_raw"] = yhat
    df_rule["rule_threshold_rounded"] = np.clip(
        5 * np.round(yhat / 5),
        0,
        75,
    ).astype(int)

    df_rule.to_csv(os.path.join(OUT, "oracle_surface_with_rule_fit.csv"), index=False)

    with open(os.path.join(OUT, "learned_rule.txt"), "w", encoding="utf-8") as f:
        f.write("Learned interpretable rule:\n\n")
        f.write("s* ≈ a + b log(1 + 1/delta) + c p + d rho\n\n")
        f.write(f"a = {beta[0]:.6f}\n")
        f.write(f"b = {beta[1]:.6f}\n")
        f.write(f"c = {beta[2]:.6f}\n")
        f.write(f"d = {beta[3]:.6f}\n")
        f.write(f"R2 = {r2:.6f}\n")

    print("\n=== LEARNED RULE ===")
    print("s* ≈ a + b log(1 + 1/delta) + c p + d rho")
    print(f"a  = {beta[0]:.4f}")
    print(f"b  = {beta[1]:.4f}")
    print(f"c  = {beta[2]:.4f}")
    print(f"d  = {beta[3]:.4f}")
    print(f"R2 = {r2:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y, yhat, alpha=0.75)
    lo = min(y.min(), yhat.min())
    hi = max(y.max(), yhat.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Oracle threshold")
    plt.ylabel("Predicted threshold")
    plt.title(f"Oracle Threshold Rule Fit, R2={r2:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rule_fit_actual_vs_predicted.png"), dpi=160)
    plt.close()

    return beta, r2, df_rule


def write_report(df, beta, r2):
    near = df[df["delta"] <= 10]
    relaxed = df[df["delta"] > 20]

    report = []
    report.append("# Oracle Surface Discovery")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append(
        "This benchmark discovers the cost-minimizing buffer threshold over a grid "
        "of disruption probabilities and demand intensities."
    )
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"Scenarios evaluated: {len(df)}")
    report.append(f"Mean oracle threshold: {df['oracle_threshold'].mean():.2f}")
    report.append(f"Median oracle threshold: {df['oracle_threshold'].median():.2f}")
    report.append(f"Max oracle threshold: {df['oracle_threshold'].max():.2f}")
    report.append("")
    report.append("## Near-critical behavior")
    report.append("")
    if len(near) > 0:
        report.append(f"Near-critical scenarios, delta <= 10: {len(near)}")
        report.append(f"Mean threshold in near-critical scenarios: {near['oracle_threshold'].mean():.2f}")
    if len(relaxed) > 0:
        report.append(f"Relaxed scenarios, delta > 20: {len(relaxed)}")
        report.append(f"Mean threshold in relaxed scenarios: {relaxed['oracle_threshold'].mean():.2f}")
    report.append("")
    report.append("## Learned rule")
    report.append("")
    report.append("s* ≈ a + b log(1 + 1/delta) + c p + d rho")
    report.append("")
    report.append(f"a = {beta[0]:.6f}")
    report.append(f"b = {beta[1]:.6f}")
    report.append(f"c = {beta[2]:.6f}")
    report.append(f"d = {beta[3]:.6f}")
    report.append(f"R2 = {r2:.6f}")
    report.append("")
    report.append("## Files")
    report.append("")
    report.append("- oracle_surface.csv")
    report.append("- oracle_surface_with_rule_fit.csv")
    report.append("- learned_rule.txt")
    report.append("- threshold_vs_delta.png")
    report.append("- threshold_vs_rho.png")
    report.append("- threshold_vs_p.png")
    report.append("- oracle_threshold_heatmap.png")
    report.append("- threshold_vs_delta_near_critical.png")
    report.append("- rule_fit_actual_vs_predicted.png")

    with open(os.path.join(OUT, "executive_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    print("Warming up Numba...")
    _ = simulate_policy_numba(
        0.10,
        80.0,
        20,
        10,
        5,
        C,
        X0,
        S_MAX,
        K_FAIL,
        K_PREVENT,
        HOLDING,
        SEED,
    )

    print("Running oracle surface discovery...")
    df = run_oracle_surface()

    print("Making plots...")
    make_plots(df)

    print("Fitting simple interpretable rule...")
    beta, r2, df_rule = fit_simple_rule(df)

    print("Writing report...")
    write_report(df, beta, r2)

    print(f"\nFiles written to: {OUT}/")
    print("- oracle_surface.csv")
    print("- oracle_surface_with_rule_fit.csv")
    print("- learned_rule.txt")
    print("- executive_summary.md")
    print("- threshold_vs_delta.png")
    print("- threshold_vs_rho.png")
    print("- threshold_vs_p.png")
    print("- oracle_threshold_heatmap.png")
    print("- threshold_vs_delta_near_critical.png")
    print("- rule_fit_actual_vs_predicted.png")

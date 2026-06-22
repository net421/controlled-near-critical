import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

OUT = "critical_boundary_results"
os.makedirs(OUT, exist_ok=True)

SEED = 20260607

# Exploration mode. Later we can raise this to 10000 or 20000.
N_RUNS = 4000
T = 365

C = 100
X0 = 50
S_MAX = 180

K_FAIL = 100.0
K_PREVENT = 10.0
HOLDING = 0.02

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


def run_boundary_discovery():
    rows = []
    t0 = time.time()
    scenario_id = 0

    for p in P_VALUES:
        for lam in LAMBDA_VALUES:
            d = stability_margin(p, lam)

            if d <= 0:
                continue

            scenario_id += 1

            # Run-to-failure baseline
            seed_rtf = SEED + scenario_id * 10000
            (
                rtf_cost,
                rtf_service,
                rtf_stockout_prob,
                rtf_avg_stockouts,
                rtf_avg_preventive,
            ) = simulate_policy_numba(
                p, lam, 0,
                N_RUNS, T,
                C, X0, S_MAX,
                K_FAIL, K_PREVENT, HOLDING,
                seed_rtf,
            )

            # Oracle grid search
            best_cost = np.inf
            best_threshold = 0
            best_service = None
            best_stockout_prob = None
            best_avg_stockouts = None
            best_avg_preventive = None

            for th in THRESHOLDS:
                seed = SEED + scenario_id * 10000 + int(th) + 123

                (
                    cost,
                    service,
                    stockout_prob,
                    avg_stockouts,
                    avg_preventive,
                ) = simulate_policy_numba(
                    p, lam, int(th),
                    N_RUNS, T,
                    C, X0, S_MAX,
                    K_FAIL, K_PREVENT, HOLDING,
                    seed,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_threshold = int(th)
                    best_service = service
                    best_stockout_prob = stockout_prob
                    best_avg_stockouts = avg_stockouts
                    best_avg_preventive = avg_preventive

            absolute_benefit = rtf_cost - best_cost
            relative_benefit = absolute_benefit / rtf_cost if rtf_cost > 0 else 0.0
            service_gain = best_service - rtf_service
            stockout_reduction = rtf_stockout_prob - best_stockout_prob

            intervention_is_valuable = int(absolute_benefit > 0)
            material_intervention = int(relative_benefit >= 0.05)

            row = {
                "scenario_id": scenario_id,
                "p": p,
                "lambda": lam,
                "delta": d,
                "rho": lam / C,
                "sigma2": sigma2(p, lam),

                "rtf_cost": rtf_cost,
                "rtf_service_level": rtf_service,
                "rtf_stockout_probability": rtf_stockout_prob,
                "rtf_avg_stockouts": rtf_avg_stockouts,

                "oracle_threshold": best_threshold,
                "oracle_cost": best_cost,
                "oracle_service_level": best_service,
                "oracle_stockout_probability": best_stockout_prob,
                "oracle_avg_stockouts": best_avg_stockouts,
                "oracle_avg_preventive": best_avg_preventive,

                "absolute_benefit": absolute_benefit,
                "relative_benefit": relative_benefit,
                "service_gain": service_gain,
                "stockout_reduction": stockout_reduction,
                "intervention_is_valuable": intervention_is_valuable,
                "material_intervention_5pct": material_intervention,
            }

            rows.append(row)

            print(
                f"scenario={scenario_id:03d} "
                f"p={p:.2f} lam={lam:.1f} delta={d:.2f} "
                f"s*={best_threshold:02d} "
                f"benefit={relative_benefit*100:6.2f}% "
                f"mat={material_intervention}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "critical_boundary.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(df)} scenarios in {elapsed:.1f} seconds.")
    return df


def summarize_boundary(df):
    bins = [-1, 2, 5, 10, 15, 20, 30, 100]
    labels = [
        "delta<=2",
        "2<delta<=5",
        "5<delta<=10",
        "10<delta<=15",
        "15<delta<=20",
        "20<delta<=30",
        "delta>30",
    ]

    df = df.copy()
    df["delta_bin"] = pd.cut(df["delta"], bins=bins, labels=labels)

    summary = df.groupby("delta_bin", observed=True).agg(
        scenarios=("scenario_id", "count"),
        mean_delta=("delta", "mean"),
        mean_oracle_threshold=("oracle_threshold", "mean"),
        mean_relative_benefit=("relative_benefit", "mean"),
        median_relative_benefit=("relative_benefit", "median"),
        pct_material_intervention=("material_intervention_5pct", "mean"),
        mean_service_gain=("service_gain", "mean"),
        mean_stockout_reduction=("stockout_reduction", "mean"),
    ).reset_index()

    summary.to_csv(os.path.join(OUT, "critical_boundary_by_delta_bin.csv"), index=False)

    print("\n=== CRITICAL BOUNDARY SUMMARY BY DELTA BIN ===\n")
    print(summary.to_string(index=False))

    return summary


def make_plots(df, summary):
    # Relative benefit vs delta
    plt.figure(figsize=(9, 6))
    plt.scatter(df["delta"], 100 * df["relative_benefit"], alpha=0.75)
    plt.axhline(5, linestyle="--")
    plt.xlabel("Stability margin delta")
    plt.ylabel("Relative benefit of oracle vs run-to-failure (%)")
    plt.title("Critical Boundary: Benefit vs Stability Margin")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "benefit_vs_delta.png"), dpi=160)
    plt.close()

    # Material intervention probability by delta bin
    plt.figure(figsize=(10, 6))
    plt.bar(summary["delta_bin"].astype(str), 100 * summary["pct_material_intervention"])
    plt.ylabel("% scenarios where intervention benefit >= 5%")
    plt.title("Material Intervention Probability by Delta Regime")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "material_intervention_by_delta_bin.png"), dpi=160)
    plt.close()

    # Oracle threshold vs benefit
    plt.figure(figsize=(9, 6))
    plt.scatter(df["oracle_threshold"], 100 * df["relative_benefit"], alpha=0.75)
    plt.xlabel("Oracle threshold")
    plt.ylabel("Relative benefit vs run-to-failure (%)")
    plt.title("Benefit vs Oracle Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "benefit_vs_oracle_threshold.png"), dpi=160)
    plt.close()

    # Heatmap: relative benefit p x lambda
    pivot = df.pivot_table(
        index="p",
        columns="lambda",
        values="relative_benefit",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(100 * pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Relative benefit (%)")
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
    plt.title("Critical Boundary Benefit Surface")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "benefit_heatmap.png"), dpi=160)
    plt.close()

    # Service gain
    plt.figure(figsize=(9, 6))
    plt.scatter(df["delta"], 100 * df["service_gain"], alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Stability margin delta")
    plt.ylabel("Service level gain (percentage points)")
    plt.title("Service Gain vs Stability Margin")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "service_gain_vs_delta.png"), dpi=160)
    plt.close()


def estimate_boundary(summary):
    """
    Crude empirical boundary:
    first delta bin where pct_material_intervention drops below 50%.
    """

    rows = summary.copy()
    boundary_text = "No clear boundary detected."

    for _, row in rows.iterrows():
        if row["pct_material_intervention"] < 0.50:
            boundary_text = (
                "Empirical boundary appears around bin: "
                f"{row['delta_bin']} "
                "(material intervention probability falls below 50%)."
            )
            break

    return boundary_text


def write_report(df, summary, boundary_text):
    near = df[df["delta"] <= 10]
    relaxed = df[df["delta"] > 20]

    report = []
    report.append("# Benchmark 3 — Critical Boundary Discovery")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append(
        "This benchmark estimates when preventive threshold control becomes "
        "economically valuable relative to run-to-failure."
    )
    report.append("")
    report.append("## Core Question")
    report.append("")
    report.append("When does the system cross from relaxed operation into a regime where intervention is materially valuable?")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(f"Scenarios evaluated: {len(df)}")
    report.append(f"Mean relative benefit: {100*df['relative_benefit'].mean():.2f}%")
    report.append(f"Median relative benefit: {100*df['relative_benefit'].median():.2f}%")
    report.append(f"Scenarios with >=5% benefit: {100*df['material_intervention_5pct'].mean():.2f}%")
    report.append("")
    report.append("## Near-critical vs relaxed")
    report.append("")
    if len(near) > 0:
        report.append(f"Near-critical scenarios, delta <= 10: {len(near)}")
        report.append(f"Mean benefit, delta <= 10: {100*near['relative_benefit'].mean():.2f}%")
        report.append(f"Material intervention rate, delta <= 10: {100*near['material_intervention_5pct'].mean():.2f}%")
        report.append("")
    if len(relaxed) > 0:
        report.append(f"Relaxed scenarios, delta > 20: {len(relaxed)}")
        report.append(f"Mean benefit, delta > 20: {100*relaxed['relative_benefit'].mean():.2f}%")
        report.append(f"Material intervention rate, delta > 20: {100*relaxed['material_intervention_5pct'].mean():.2f}%")
        report.append("")
    report.append("## Boundary estimate")
    report.append("")
    report.append(boundary_text)
    report.append("")
    report.append("## Delta-bin summary")
    report.append("")
    report.append(summary.to_string(index=False))
    report.append("")
    report.append("## Files")
    report.append("")
    report.append("- critical_boundary.csv")
    report.append("- critical_boundary_by_delta_bin.csv")
    report.append("- benefit_vs_delta.png")
    report.append("- material_intervention_by_delta_bin.png")
    report.append("- benefit_vs_oracle_threshold.png")
    report.append("- benefit_heatmap.png")
    report.append("- service_gain_vs_delta.png")

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

    print("Running Critical Boundary Discovery...")
    df = run_boundary_discovery()

    summary = summarize_boundary(df)

    print("Making plots...")
    make_plots(df, summary)

    boundary_text = estimate_boundary(summary)

    print("\n=== EMPIRICAL BOUNDARY ESTIMATE ===\n")
    print(boundary_text)

    write_report(df, summary, boundary_text)

    print(f"\nFiles written to: {OUT}/")
    print("- critical_boundary.csv")
    print("- critical_boundary_by_delta_bin.csv")
    print("- benefit_vs_delta.png")
    print("- material_intervention_by_delta_bin.png")
    print("- benefit_vs_oracle_threshold.png")
    print("- benefit_heatmap.png")
    print("- service_gain_vs_delta.png")
    print("- executive_summary.md")

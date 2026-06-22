import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

OUT = "critical_boundary_cost_ratio_broad_results"
os.makedirs(OUT, exist_ok=True)

SEED = 20260607

N_RUNS = 5000
T = 365

C = 100
X0 = 50
S_MAX = 180

K_PREVENT = 10.0
HOLDING = 0.02

COST_RATIOS = np.array([1, 5, 10, 25, 50, 100], dtype=np.float64)

P_VALUES = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30])

LAMBDA_VALUES = np.array([50, 55, 60, 65, 70, 75, 78, 80, 82, 84, 86, 88, 90], dtype=np.float64)

THRESHOLDS = np.arange(0, 80, 5)


def stability_margin(p, lam):
    return (1.0 - p) * C - lam


def sigma2(p, lam):
    return lam + p * (1.0 - p) * C**2


@njit
def simulate_policy_numba(
    p, lam, threshold, n_runs, T, C, X0, S_MAX,
    K_FAIL, K_PREVENT, HOLDING, seed
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

    return (
        total_cost_sum / n_runs,
        service_numer / service_denom,
        stockout_run_count / n_runs,
        stockouts_sum / n_runs,
        preventive_sum / n_runs,
    )


def run_cost_ratio_benchmark():
    rows = []
    t0 = time.time()
    scenario_id = 0

    for ratio in COST_RATIOS:
        K_FAIL = K_PREVENT * ratio

        for p in P_VALUES:
            for lam in LAMBDA_VALUES:
                d = stability_margin(p, lam)

                if d <= 0:
                    continue

                scenario_id += 1
                seed_base = SEED + scenario_id * 10000 + int(ratio * 100)

                rtf_cost, rtf_service, rtf_stockout_prob, rtf_avg_stockouts, _ = simulate_policy_numba(
                    p, lam, 0,
                    N_RUNS, T,
                    C, X0, S_MAX,
                    K_FAIL, K_PREVENT, HOLDING,
                    seed_base,
                )

                best_cost = np.inf
                best_threshold = 0
                best_service = 0.0
                best_stockout_prob = 0.0
                best_avg_stockouts = 0.0
                best_avg_preventive = 0.0

                for th in THRESHOLDS:
                    seed = seed_base + int(th) + 123

                    cost, service, stockout_prob, avg_stockouts, avg_preventive = simulate_policy_numba(
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

                rows.append({
                    "scenario_id": scenario_id,
                    "cost_ratio": ratio,
                    "K_fail": K_FAIL,
                    "K_prevent": K_PREVENT,
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
                    "material_intervention_5pct": int(relative_benefit >= 0.05),
                    "material_intervention_10pct": int(relative_benefit >= 0.10),
                    "material_intervention_15pct": int(relative_benefit >= 0.15),
                })

                print(
                    f"ratio={ratio:6.1f} "
                    f"p={p:.2f} lam={lam:.1f} delta={d:6.2f} "
                    f"s*={best_threshold:02d} "
                    f"benefit={100*relative_benefit:7.2f}%"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "critical_boundary_cost_ratio_fine.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(df)} scenario-ratio combinations in {elapsed:.1f} seconds.")
    return df


def summarize(df):
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

    by_ratio = df.groupby("cost_ratio", observed=True).agg(
        scenarios=("scenario_id", "count"),
        mean_relative_benefit=("relative_benefit", "mean"),
        median_relative_benefit=("relative_benefit", "median"),
        pct_material_5pct=("material_intervention_5pct", "mean"),
        pct_material_10pct=("material_intervention_10pct", "mean"),
        pct_material_15pct=("material_intervention_15pct", "mean"),
        mean_oracle_threshold=("oracle_threshold", "mean"),
        mean_service_gain=("service_gain", "mean"),
        mean_stockout_reduction=("stockout_reduction", "mean"),
    ).reset_index()

    by_delta_ratio = df.groupby(["delta_bin", "cost_ratio"], observed=True).agg(
        scenarios=("scenario_id", "count"),
        mean_delta=("delta", "mean"),
        mean_relative_benefit=("relative_benefit", "mean"),
        median_relative_benefit=("relative_benefit", "median"),
        pct_material_5pct=("material_intervention_5pct", "mean"),
        pct_material_10pct=("material_intervention_10pct", "mean"),
        pct_material_15pct=("material_intervention_15pct", "mean"),
        mean_oracle_threshold=("oracle_threshold", "mean"),
        mean_service_gain=("service_gain", "mean"),
        mean_stockout_reduction=("stockout_reduction", "mean"),
    ).reset_index()

    by_ratio.to_csv(os.path.join(OUT, "benefit_by_cost_ratio_fine.csv"), index=False)
    by_delta_ratio.to_csv(os.path.join(OUT, "benefit_by_delta_regime_and_cost_ratio_fine.csv"), index=False)

    print("\n=== BENEFIT BY COST RATIO ===\n")
    print(by_ratio.to_string(index=False))

    print("\n=== BENEFIT BY DELTA REGIME AND COST RATIO ===\n")
    print(by_delta_ratio.to_string(index=False))

    return by_ratio, by_delta_ratio


def make_plots(df, by_ratio, by_delta_ratio):
    plt.figure(figsize=(9, 6))
    plt.plot(by_ratio["cost_ratio"], 100 * by_ratio["mean_relative_benefit"], marker="o")
    plt.xlabel("Cost ratio K_fail / K_prevent")
    plt.ylabel("Mean relative benefit (%)")
    plt.title("Fine Sweep: Mean Benefit vs Cost Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fine_mean_benefit_vs_cost_ratio.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(by_ratio["cost_ratio"], 100 * by_ratio["pct_material_5pct"], marker="o", label=">=5%")
    plt.plot(by_ratio["cost_ratio"], 100 * by_ratio["pct_material_10pct"], marker="o", label=">=10%")
    plt.plot(by_ratio["cost_ratio"], 100 * by_ratio["pct_material_15pct"], marker="o", label=">=15%")
    plt.xlabel("Cost ratio K_fail / K_prevent")
    plt.ylabel("% scenarios")
    plt.title("Fine Sweep: Material Intervention Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fine_material_probability_vs_cost_ratio.png"), dpi=160)
    plt.close()

    pivot = by_delta_ratio.pivot_table(
        index="delta_bin",
        columns="cost_ratio",
        values="mean_relative_benefit",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(100 * pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Mean relative benefit (%)")
    plt.xticks(np.arange(len(pivot.columns)), [str(int(x)) for x in pivot.columns], rotation=45)
    plt.yticks(np.arange(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("Cost ratio")
    plt.ylabel("Delta regime")
    plt.title("Fine Sweep: Economic Critical Boundary Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fine_economic_critical_boundary_heatmap.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(df["delta"], 100 * df["relative_benefit"], alpha=0.45, c=df["cost_ratio"])
    plt.colorbar(label="Cost ratio")
    plt.xlabel("Stability margin delta")
    plt.ylabel("Relative benefit (%)")
    plt.title("Fine Sweep: Benefit vs Delta Colored by Cost Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fine_benefit_vs_delta_colored_by_ratio.png"), dpi=160)
    plt.close()


def estimate_boundary(by_ratio):
    material = by_ratio[by_ratio["pct_material_5pct"] >= 0.50]
    if len(material) == 0:
        return "No global cost-ratio boundary found for >=50% scenarios with benefit >=5%."
    r = material.iloc[0]
    return (
        f"Global empirical boundary: cost ratio ≈ {r['cost_ratio']:.0f}, "
        f"where {100*r['pct_material_5pct']:.1f}% of scenarios exceed 5% benefit."
    )


def write_report(df, by_ratio, by_delta_ratio, boundary_text):
    report = []
    report.append("# Benchmark 4A — Broad Critical Boundary vs Cost Ratio")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append("Broad sweep over cost ratios 1–100 to detect the economic critical boundary.")
    report.append("")
    report.append("## Setup")
    report.append("")
    report.append(f"N_RUNS = {N_RUNS}")
    report.append(f"T = {T}")
    report.append(f"K_PREVENT = {K_PREVENT}")
    report.append(f"Cost ratios tested = {list(COST_RATIOS)}")
    report.append("")
    report.append("## Boundary estimate")
    report.append("")
    report.append(boundary_text)
    report.append("")
    report.append("## Benefit by Cost Ratio")
    report.append("")
    report.append(by_ratio.to_string(index=False))
    report.append("")
    report.append("## Benefit by Delta Regime and Cost Ratio")
    report.append("")
    report.append(by_delta_ratio.to_string(index=False))

    with open(os.path.join(OUT, "executive_summary_fine.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    print("Warming up Numba...")
    _ = simulate_policy_numba(
        0.10, 80.0, 20,
        10, 5,
        C, X0, S_MAX,
        100.0, K_PREVENT, HOLDING,
        SEED,
    )

    print("Running Benchmark 4A — Broad Critical Boundary vs Cost Ratio...")
    df = run_cost_ratio_benchmark()

    by_ratio, by_delta_ratio = summarize(df)

    print("Making plots...")
    make_plots(df, by_ratio, by_delta_ratio)

    boundary_text = estimate_boundary(by_ratio)

    print("\n=== EMPIRICAL ECONOMIC BOUNDARY ===\n")
    print(boundary_text)

    write_report(df, by_ratio, by_delta_ratio, boundary_text)

    print(f"\nFiles written to: {OUT}/")

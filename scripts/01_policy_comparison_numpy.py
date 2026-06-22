import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "near_critical_benchmark_results"
os.makedirs(OUT, exist_ok=True)

SEED = 20260606
N_RUNS = 5000
T = 365

C = 100
X0 = 50
S_MAX = 180

K_FAIL = 100.0
K_PREVENT = 10.0
HOLDING = 0.02

REGIMES = [
    {"name": "relaxed",       "p": 0.05, "lam": 70},
    {"name": "moderate",      "p": 0.10, "lam": 80},
    {"name": "near_critical", "p": 0.15, "lam": 83},
    {"name": "fragile",       "p": 0.20, "lam": 79},
]

def stability_margin(p, lam):
    return (1 - p) * C - lam

def sigma2(p, lam):
    return lam + p * (1 - p) * C**2

def classic_safety_threshold(p, lam, z=1.65):
    return int(math.ceil(z * math.sqrt(sigma2(p, lam))))

def near_critical_threshold(p, lam, gamma=0.18):
    d = max(stability_margin(p, lam), 1e-6)
    return int(math.ceil(gamma * sigma2(p, lam) / d))

def simulate_policy(p, lam, threshold, rng):
    I = np.full(N_RUNS, X0, dtype=float)
    total_cost = np.zeros(N_RUNS)
    stockouts = np.zeros(N_RUNS)
    preventive = np.zeros(N_RUNS)
    served_demand = np.zeros(N_RUNS)
    total_demand = np.zeros(N_RUNS)

    for _ in range(T):
        intervene = I <= threshold
        total_cost += intervene * K_PREVENT
        preventive += intervene
        I[intervene] = X0

        disruptive = rng.random(N_RUNS) < p
        replenishment = np.where(disruptive, 0, C)
        demand = rng.poisson(lam, N_RUNS)
        total_demand += demand

        before_demand = I + replenishment
        served = np.minimum(before_demand, demand)
        served_demand += served
        I = np.minimum(before_demand - demand, S_MAX)

        failed = I <= 0
        stockouts += failed
        total_cost += failed * K_FAIL
        I[failed] = X0
        total_cost += HOLDING * np.maximum(I, 0)

    return {
        "threshold": threshold,
        "service_level": served_demand.sum() / total_demand.sum(),
        "stockout_probability": (stockouts > 0).mean(),
        "avg_stockouts": stockouts.mean(),
        "avg_preventive_actions": preventive.mean(),
        "avg_total_cost": total_cost.mean(),
    }

def evaluate_regime(regime):
    p = regime["p"]
    lam = regime["lam"]
    d = stability_margin(p, lam)
    policies = [
        ("run_to_failure", 0),
        ("classic_safety_z165", classic_safety_threshold(p, lam, z=1.65)),
        ("classic_safety_z233", classic_safety_threshold(p, lam, z=2.33)),
        ("near_critical_gamma018", near_critical_threshold(p, lam, gamma=0.18)),
        ("near_critical_gamma030", near_critical_threshold(p, lam, gamma=0.30)),
    ]
    rows = []
    for policy_name, threshold in policies:
        rng = np.random.default_rng(SEED)
        result = simulate_policy(p, lam, threshold, rng)
        result.update({"regime": regime["name"], "policy": policy_name, "p": p, "lambda": lam, "delta": d, "rho": lam / C, "sigma2": sigma2(p, lam)})
        rows.append(result)

    oracle_rows = []
    for threshold in range(0, 80, 5):
        rng = np.random.default_rng(SEED)
        result = simulate_policy(p, lam, threshold, rng)
        result.update({"regime": regime["name"], "policy": "oracle_grid_threshold", "p": p, "lambda": lam, "delta": d, "rho": lam / C, "sigma2": sigma2(p, lam)})
        oracle_rows.append(result)
    rows.append(min(oracle_rows, key=lambda x: x["avg_total_cost"]))
    return rows, oracle_rows

all_rows, oracle_all = [], []
for regime in REGIMES:
    rows, oracle_rows = evaluate_regime(regime)
    all_rows.extend(rows)
    oracle_all.extend(oracle_rows)

df = pd.DataFrame(all_rows)
oracle_df = pd.DataFrame(oracle_all)
df.to_csv(os.path.join(OUT, "policy_benchmark_summary.csv"), index=False)
oracle_df.to_csv(os.path.join(OUT, "oracle_threshold_grid.csv"), index=False)

print("\n=== POLICY BENCHMARK SUMMARY ===\n")
print(df[["regime","policy","p","lambda","delta","threshold","service_level","stockout_probability","avg_stockouts","avg_preventive_actions","avg_total_cost"]].to_string(index=False))

best_by_regime = df.loc[df.groupby("regime")["avg_total_cost"].idxmin()].copy()
print("\n=== BEST POLICY BY REGIME ===\n")
print(best_by_regime[["regime","policy","delta","threshold","service_level","stockout_probability","avg_total_cost"]].to_string(index=False))

plt.figure(figsize=(12, 6))
plot_df = df.copy(); plot_df["label"] = plot_df["regime"] + " | " + plot_df["policy"]
plt.barh(plot_df["label"], plot_df["avg_total_cost"])
plt.xlabel("Average total cost"); plt.title("Policy Benchmark: Average Total Cost")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "policy_cost_comparison.png"), dpi=160); plt.close()

plt.figure(figsize=(12, 6))
plt.barh(plot_df["label"], plot_df["stockout_probability"])
plt.xlabel("Stockout probability"); plt.title("Policy Benchmark: Stockout Probability")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "policy_stockout_probability.png"), dpi=160); plt.close()

for regime in REGIMES:
    od = oracle_df[oracle_df["regime"] == regime["name"]].sort_values("threshold")
    plt.figure(figsize=(8, 5)); plt.plot(od["threshold"], od["avg_total_cost"], marker="o")
    plt.xlabel("Threshold"); plt.ylabel("Average total cost"); plt.title(f"Oracle Threshold Cost Curve — {regime['name']}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"oracle_curve_{regime['name']}.png"), dpi=160); plt.close()

report_lines = [
    "# Near-Critical Policy Benchmark", "", "## Purpose", "",
    "Controlled one-node benchmark before scaling to a full supply chain digital twin.", "",
    "The goal is to test whether near-critical buffer policies show measurable advantage against classical safety-stock-style policies.", "",
    "## Best Policy by Regime", "",
    best_by_regime[["regime","policy","delta","threshold","service_level","stockout_probability","avg_total_cost"]].to_string(index=False), "",
    "## Interpretation Guide", "",
    "- If near-critical policies beat classic policies in near-critical or fragile regimes, the approach has signal.",
    "- If the oracle grid beats everything, the next step is to learn an adaptive approximation to the oracle threshold.",
    "- If classic policies dominate, the near-critical rule needs reformulation before going into the digital twin.",
]
with open(os.path.join(OUT, "executive_summary.md"), "w", encoding="utf-8") as f: f.write("\n".join(report_lines))
print(f"\nFiles written to: {OUT}/")

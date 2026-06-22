import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

IN = "oracle_surface_results/oracle_surface.csv"
OUT = "oracle_surface_results/rule_fits"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN).copy()

df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["p", "lambda", "delta", "rho", "sigma2", "oracle_threshold"]
)

y = df["oracle_threshold"].values.astype(float)

# ============================================================
# Model definitions
# ============================================================

def linear_rule(X, a, b, c, d):
    delta, p, rho, sigma2 = X
    return a + b * np.log1p(1.0 / delta) + c * p + d * rho

def inv_delta_saturated_rule(X, a, b, c, d, delta0, s_cap):
    delta, p, rho, sigma2 = X
    raw = a + b / (delta + delta0) + c * p + d * rho
    return np.minimum(s_cap, np.maximum(0.0, raw))

def sigma_delta_saturated_rule(X, a, b, c, delta0, s_cap):
    delta, p, rho, sigma2 = X
    raw = a + b * sigma2 / (delta + delta0) + c * p
    return np.minimum(s_cap, np.maximum(0.0, raw))

def log_saturated_rule(X, a, b, c, d, s_cap):
    delta, p, rho, sigma2 = X
    raw = a + b * np.log1p(1.0 / delta) + c * p + d * rho
    return np.minimum(s_cap, np.maximum(0.0, raw))

X = np.vstack([
    df["delta"].values.astype(float),
    df["p"].values.astype(float),
    df["rho"].values.astype(float),
    df["sigma2"].values.astype(float),
])

def evaluate(name, y_true, y_pred):
    y_round = np.clip(5 * np.round(y_pred / 5), 0, 75)

    return {
        "model": name,
        "r2_raw": r2_score(y_true, y_pred),
        "mae_raw": mean_absolute_error(y_true, y_pred),
        "r2_rounded": r2_score(y_true, y_round),
        "mae_rounded": mean_absolute_error(y_true, y_round),
    }, y_round

results = []
predictions = df.copy()

# ============================================================
# Fit 1: linear rule
# ============================================================

popt_linear, _ = curve_fit(
    linear_rule,
    X,
    y,
    p0=[0, 30, 100, 10],
    maxfev=100000,
)

pred_linear = linear_rule(X, *popt_linear)
res, rounded = evaluate("linear_log_rule", y, pred_linear)
results.append(res)
predictions["pred_linear"] = pred_linear
predictions["pred_linear_rounded"] = rounded

# ============================================================
# Fit 2: saturated inverse-delta rule
# s* = min(s_cap, max(0, a + b/(delta+delta0) + c*p + d*rho))
# ============================================================

popt_inv, _ = curve_fit(
    inv_delta_saturated_rule,
    X,
    y,
    p0=[0, 40, 80, 10, 2, 45],
    bounds=(
        [-100, 0, -300, -300, 0.1, 10],
        [100, 500, 500, 500, 50, 80],
    ),
    maxfev=200000,
)

pred_inv = inv_delta_saturated_rule(X, *popt_inv)
res, rounded = evaluate("saturated_inverse_delta_rule", y, pred_inv)
results.append(res)
predictions["pred_saturated_inverse_delta"] = pred_inv
predictions["pred_saturated_inverse_delta_rounded"] = rounded

# ============================================================
# Fit 3: saturated sigma/delta rule
# s* = min(s_cap, max(0, a + b*sigma2/(delta+delta0) + c*p))
# ============================================================

popt_sig, _ = curve_fit(
    sigma_delta_saturated_rule,
    X,
    y,
    p0=[0, 0.1, 80, 5, 45],
    bounds=(
        [-100, 0, -300, 0.1, 10],
        [100, 10, 500, 50, 80],
    ),
    maxfev=200000,
)

pred_sig = sigma_delta_saturated_rule(X, *popt_sig)
res, rounded = evaluate("saturated_sigma_delta_rule", y, pred_sig)
results.append(res)
predictions["pred_saturated_sigma_delta"] = pred_sig
predictions["pred_saturated_sigma_delta_rounded"] = rounded

# ============================================================
# Fit 4: saturated log rule
# ============================================================

popt_logsat, _ = curve_fit(
    log_saturated_rule,
    X,
    y,
    p0=[0, 30, 100, 10, 45],
    bounds=(
        [-100, 0, -300, -300, 10],
        [100, 500, 500, 500, 80],
    ),
    maxfev=200000,
)

pred_logsat = log_saturated_rule(X, *popt_logsat)
res, rounded = evaluate("saturated_log_rule", y, pred_logsat)
results.append(res)
predictions["pred_saturated_log"] = pred_logsat
predictions["pred_saturated_log_rounded"] = rounded

# ============================================================
# Random Forest ceiling model
# ============================================================

features = ["p", "lambda", "delta", "rho", "sigma2"]
X_ml = df[features].values
y_ml = y

X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
)

rf.fit(X_train, y_train)

rf_pred_all = rf.predict(X_ml)
rf_pred_test = rf.predict(X_test)

res_all, rf_round = evaluate("random_forest_all_data", y_ml, rf_pred_all)
res_test, _ = evaluate("random_forest_test_split", y_test, rf_pred_test)

results.append(res_all)
results.append(res_test)

predictions["pred_random_forest"] = rf_pred_all
predictions["pred_random_forest_rounded"] = rf_round

# ============================================================
# Save outputs
# ============================================================

results_df = pd.DataFrame(results).sort_values("r2_rounded", ascending=False)
results_df.to_csv(os.path.join(OUT, "rule_fit_comparison.csv"), index=False)
predictions.to_csv(os.path.join(OUT, "oracle_surface_predictions.csv"), index=False)

with open(os.path.join(OUT, "fitted_parameters.txt"), "w", encoding="utf-8") as f:
    f.write("LINEAR LOG RULE\n")
    f.write("s* = a + b log(1 + 1/delta) + c p + d rho\n")
    f.write(f"a,b,c,d = {popt_linear}\n\n")

    f.write("SATURATED INVERSE DELTA RULE\n")
    f.write("s* = min(s_cap, max(0, a + b/(delta+delta0) + c p + d rho))\n")
    f.write(f"a,b,c,d,delta0,s_cap = {popt_inv}\n\n")

    f.write("SATURATED SIGMA/DELTA RULE\n")
    f.write("s* = min(s_cap, max(0, a + b*sigma2/(delta+delta0) + c p))\n")
    f.write(f"a,b,c,delta0,s_cap = {popt_sig}\n\n")

    f.write("SATURATED LOG RULE\n")
    f.write("s* = min(s_cap, max(0, a + b log(1+1/delta) + c p + d rho))\n")
    f.write(f"a,b,c,d,s_cap = {popt_logsat}\n\n")

# ============================================================
# Plots
# ============================================================

def actual_vs_pred(col, title, filename):
    plt.figure(figsize=(7, 7))
    plt.scatter(predictions["oracle_threshold"], predictions[col], alpha=0.75)
    lo = min(predictions["oracle_threshold"].min(), predictions[col].min())
    hi = max(predictions["oracle_threshold"].max(), predictions[col].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Oracle threshold")
    plt.ylabel("Predicted threshold")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=160)
    plt.close()

actual_vs_pred(
    "pred_saturated_inverse_delta",
    "Saturated Inverse-Delta Rule: Actual vs Predicted",
    "actual_vs_pred_saturated_inverse_delta.png",
)

actual_vs_pred(
    "pred_saturated_sigma_delta",
    "Saturated Sigma/Delta Rule: Actual vs Predicted",
    "actual_vs_pred_saturated_sigma_delta.png",
)

actual_vs_pred(
    "pred_saturated_log",
    "Saturated Log Rule: Actual vs Predicted",
    "actual_vs_pred_saturated_log.png",
)

actual_vs_pred(
    "pred_random_forest",
    "Random Forest Ceiling Model: Actual vs Predicted",
    "actual_vs_pred_random_forest.png",
)

# Residual plot for best interpretable model
interpretable = results_df[~results_df["model"].str.contains("random_forest")].iloc[0]["model"]

model_to_col = {
    "linear_log_rule": "pred_linear",
    "saturated_inverse_delta_rule": "pred_saturated_inverse_delta",
    "saturated_sigma_delta_rule": "pred_saturated_sigma_delta",
    "saturated_log_rule": "pred_saturated_log",
}

best_col = model_to_col[interpretable]
predictions["best_interpretable_residual"] = (
    predictions["oracle_threshold"] - predictions[best_col]
)

plt.figure(figsize=(9, 6))
plt.scatter(predictions["delta"], predictions["best_interpretable_residual"], alpha=0.75)
plt.axhline(0, linestyle="--")
plt.xlabel("delta")
plt.ylabel("oracle - predicted")
plt.title(f"Residuals vs Delta — {interpretable}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "best_interpretable_residuals_vs_delta.png"), dpi=160)
plt.close()

# ============================================================
# Report
# ============================================================

with open(os.path.join(OUT, "executive_summary.md"), "w", encoding="utf-8") as f:
    f.write("# Benchmark 2B — Saturated Rule Fit\n\n")
    f.write("## Model Comparison\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"Best interpretable model: {interpretable}\n\n")
    f.write("Interpretation:\n")
    f.write("- If the saturated models improve clearly over the linear model, the oracle surface is nonlinear but structured.\n")
    f.write("- If the Random Forest is much better than all analytic rules, the surface has interactions that require a richer policy approximation.\n")
    f.write("- If rounded MAE is below 5 threshold units, the rule is operationally close because thresholds are evaluated in increments of 5.\n")

print("\n=== RULE FIT COMPARISON ===\n")
print(results_df.to_string(index=False))

print("\nFiles written to:")
print(OUT)
print("- rule_fit_comparison.csv")
print("- oracle_surface_predictions.csv")
print("- fitted_parameters.txt")
print("- executive_summary.md")

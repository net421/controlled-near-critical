# Reproducibility Notes

## Environment

Use Python 3.10+ or 3.12 with the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The simulations use NumPy, Pandas, Matplotlib, SciPy, and Numba. Some scripts warm up Numba JIT compilation and then run Monte Carlo grids.

## Suggested run order

```bash
python scripts/01_policy_comparison_numpy.py
python scripts/02_oracle_surface_discovery_numba.py
python scripts/02b_fit_saturated_oracle_rule.py
python scripts/03_critical_boundary_discovery_numba.py
python scripts/04a_cost_ratio_boundary_broad_numba.py
python scripts/04b_cost_ratio_boundary_fine_numba.py
```

Optional Part A reproduction from the Paper B code document:

```bash
python scripts/part_a_beta_fitting.py
```

## Expected outputs

The scripts write outputs into the following directories:

```text
results/benchmark_1_policy_comparison/
results/benchmark_2_oracle_surface/
results/benchmark_4_cost_ratio_boundary/
data/part_a_simulations/
results_part_a/
```

## Computational note

The full Part A beta-fitting script is computationally heavier than the smoke checks. It uses Monte Carlo grids and may take 30-45 minutes on a modern laptop after Numba warm-up.

## Repository policy

Text, source code, CSV summaries, and markdown reports are tracked in Git.

Large or binary manuscripts and figures should preferably be distributed through:

- GitHub Releases
- Zenodo archive
- supplementary-material archive

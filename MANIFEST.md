# Repository Manifest

## Source packages used

- `controlled_benchmark_clean(3).zip`: controlled benchmark scripts, CSV outputs, markdown summaries, and figure files.
- `CODIGO PAPER B(2).docx`: Paper B code document containing the Part A beta-fitting reproduction script.
- `supply network near critical(2).zip`: related Digital Twin / industrial application package used for research lineage and cross-repository connection.

## Files tracked in this repository

### Scripts

- `scripts/part_a_beta_fitting.py`
- `scripts/01_policy_comparison_numpy.py`
- `scripts/02_oracle_surface_discovery_numba.py`
- `scripts/02b_fit_saturated_oracle_rule.py`
- `scripts/03_critical_boundary_discovery_numba.py`
- `scripts/04a_cost_ratio_boundary_broad_numba.py`
- `scripts/04b_cost_ratio_boundary_fine_numba.py`

### Results

CSV and text/markdown outputs from:

- benchmark 1: policy comparison
- benchmark 2: oracle surface
- benchmark 4: cost-ratio boundary

### Manuscript material

- `manuscript/paper_extension_section.md`

### Documentation

- `docs/research_lineage.md`
- `docs/reproducibility.md`

## Binary files not directly committed in this first pass

Some `.png` and `.docx` files were present in the source packages. They are better attached through a GitHub Release or Zenodo deposit to keep the Git repository lightweight and clean.

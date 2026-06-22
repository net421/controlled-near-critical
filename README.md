# Controlled Near-Critical Benchmark (Paper B)

This repository contains the reproducible benchmark, scripts, results, figures, and manuscript materials associated with **Paper B**.

The work builds on the foundational **Near-Critical Systems** framework and documents the controlled benchmark used to support the industrial **Supply Chain Digital Twin** application.

**Paper A archived release:** https://doi.org/10.5281/zenodo.20792811

## Research position

```text
Near-Critical Systems / Paper A
        ↓
Controlled Near-Critical Benchmark / Paper B
        ↓
Supply Chain Digital Twin application
```

This repository should be read as the **experimental bridge** between the original near-critical theory and the later Digital Twin implementation.

## What this repository contains

- controlled benchmark scripts for near-critical preventive threshold policies
- Monte Carlo policy-comparison experiments
- oracle-threshold surface discovery
- physical/economic critical-boundary analysis
- cost-ratio boundary experiments
- Paper B manuscript/source materials
- reproducibility notes and research-lineage documentation

## Main contributions

1. **Controlled benchmark design**  
   A clean experimental environment for studying near-critical collapse risk and preventive intervention.

2. **Threshold-policy comparison**  
   Comparison between run-to-failure, static safety rules, and near-critical/oracle threshold policies.

3. **Physical/economic boundary analysis**  
   Results showing when preventive intervention becomes materially valuable as a function of stability margin and failure/prevention cost ratio.

4. **Bridge to Digital Twin deployment**  
   The benchmark supports the later supply-chain Digital Twin application, where near-critical logic is transferred to a networked industrial setting.

## Repository structure

```text
controlled-near-critical/
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── MANIFEST.md
├── .gitignore
├── .github/workflows/research-smoke.yml
├── scripts/
│   ├── part_a_beta_fitting.py
│   ├── 01_policy_comparison_numpy.py
│   ├── 02_oracle_surface_discovery_numba.py
│   ├── 02b_fit_saturated_oracle_rule.py
│   ├── 03_critical_boundary_discovery_numba.py
│   ├── 04a_cost_ratio_boundary_broad_numba.py
│   └── 04b_cost_ratio_boundary_fine_numba.py
├── results/
│   ├── benchmark_1_policy_comparison/
│   ├── benchmark_2_oracle_surface/
│   └── benchmark_4_cost_ratio_boundary/
├── docs/
│   ├── research_lineage.md
│   └── reproducibility.md
└── manuscript/
    └── paper_extension_section.md
```

## How to reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the controlled benchmark stages:

```bash
python scripts/01_policy_comparison_numpy.py
python scripts/02_oracle_surface_discovery_numba.py
python scripts/02b_fit_saturated_oracle_rule.py
python scripts/03_critical_boundary_discovery_numba.py
python scripts/04a_cost_ratio_boundary_broad_numba.py
python scripts/04b_cost_ratio_boundary_fine_numba.py
```

Run the Part A beta-fitting reproduction from the Paper B code document:

```bash
python scripts/part_a_beta_fitting.py
```

Some simulations are computationally heavy and may take tens of minutes depending on CPU and Numba warm-up.

## Related repositories

- [Near-Critical Systems / Paper A](https://github.com/net421/near-critical-systems): foundational theory and simulation framework. Archived release DOI: https://doi.org/10.5281/zenodo.20792811
- [Supply Chain Digital Twin](https://github.com/net421/Supply-Chain-Digital-Twin): industrial Digital Twin application using the near-critical framework.

## Status

Active research repository. The text files, scripts, CSV outputs, and manuscript section are tracked directly in GitHub. Larger binary manuscripts/figures can be attached later as a GitHub Release or archived through Zenodo.

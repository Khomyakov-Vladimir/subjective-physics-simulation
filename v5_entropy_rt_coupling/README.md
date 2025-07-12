# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15867963.svg)](https://doi.org/10.5281/zenodo.15867963)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This directory contains code and data for **Version 5** of the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/v5_entropy_rt_coupling
├── main.py # Run the full simulation (standard + adaptive)
├── observer.py # Observer with multiple adaptation modes
├── trade_off_functional_lambda_comparison.py
├── adaptive_perceptual_dynamics.py
├── landauer_extension.py
├── compare_entropy_fixed_adaptive.py
├── von_neumann_entropy.py
├── simulate_entropy_rt_full.py
├── compare_dirichlet_params.py
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── figures/
│ ├── entropy_vs_epsilon.pdf
│ ├── norm_vs_time.pdf
│ ├── trace_distance_vs_epsilon.pdf
│ ├── adaptive_entropy_entropy.pdf
│ ├── adaptive_entropy_norm.pdf
│ ├── adaptive_norm_entropy.pdf
│ ├── adaptive_norm_norm.pdf
│ ├── adaptive_perceptual_dynamics.pdf
│ ├── adaptive_threshold_entropy.pdf
│ ├── adaptive_threshold_norm.pdf
│ ├── adaptive_trace_distance_entropy.pdf
│ ├── adaptive_trace_distance_norm.pdf
│ ├── L_epsilon_lambda_comparison_plot.pdf
│ ├── deltaS_vs_epsilon.pdf
│ ├── energy_vs_epsilon.pdf
│ ├── entropy_comparison.pdf
│ ├── trace_distance_over_time.pdf
│ ├── entropy_over_time.pdf
│ ├── epsilon_over_time.pdf
│ ├── histogram_rt.pdf
│ ├── scatter_entropy_rt.pdf
│ ├── cdf_rt.pdf
│ ├── entropy_hist_comparison.pdf
│ ├── rt_cdf_comparison.pdf
│ ├── entropy_boxplot.pdf
│ └── rt_boxplot.pdf
└── data/
  ├── tradeoff_data_lambda_comparison.csv
  ├── tradeoff_data_lambda1e20.csv
  ├── adaptive_metrics_entropy.npz
  ├── adaptive_metrics_norm.npz
  ├── simulation_data.npz
  ├── simulated_data.csv
  ├── simulated_data_dirichlet_1_1.csv
  ├── simulated_data_dirichlet_2_2.csv
  ├── simulated_data_dirichlet_5_5.csv
  ├── simulated_data_dirichlet_10_10.csv
  └── simulated_data_all_configs.csv
```


## Description

Version 5 investigates the coupling between subjective entropy and reaction time in simulated perceptual decision-making.

It includes:

* Large-scale simulation of observer entropy sampled from Dirichlet distributions.

* Entropy-based modeling of reaction time under additive Gaussian noise.

* Statistical analysis: Spearman and Pearson correlations, bootstrapped confidence intervals.

* Cross-comparison across multiple Dirichlet configurations.

* Publication-ready visualizations: histograms, CDFs, scatter plots, boxplots.

## Files

- **simulate_entropy_rt_full.py**  
  Runs the main simulation, saves raw data to CSV, and generates:
  - `figures/histogram_rt.pdf`
  - `figures/scatter_entropy_rt.pdf`
  - `figures/cdf_rt.pdf`

- **compare_dirichlet_params.py**  
  Compares different Dirichlet prior configurations:
  - `figures/entropy_hist_comparison.pdf`
  - `figures/rt_cdf_comparison.pdf`
  - `figures/entropy_boxplot.pdf`
  - `figures/rt_boxplot.pdf`

- **figures/**  
  Contains all publication-quality plots.

- **data/**  
  Includes simulation output files and CSV tables.

## Running the simulation

1. Install dependencies:

```bash
pip install numpy matplotlib scipy pandas tqdm
```

2. Run the main entropy–reaction time simulation:

```bash
python simulate_entropy_rt_full.py
```

3. Run the Dirichlet parameter comparison:

```bash
python compare_dirichlet_params.py
```

## Citation

If you use this code, please cite the article:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (5.0)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.15867963]

---

## 📄 License

MIT License. See `LICENSE` file for details.

## 📜 Version History and DOI Links

For DOI versioning and archival history, see:

- [All versions](https://doi.org/10.5281/zenodo.15719389)
- [Version 1 only](https://doi.org/10.5281/zenodo.15719390)
- [Version 2 only](https://doi.org/10.5281/zenodo.15751229)
- [Version 3 only](https://doi.org/10.5281/zenodo.15780239)
- [Version 4 only](https://doi.org/10.5281/zenodo.15813188)
- [Version 5 only (current)](https://doi.org/10.5281/zenodo.15867963)

👉 **View this version on Zenodo:**  
[https://doi.org/10.5281/zenodo.15867963](https://doi.org/10.5281/zenodo.15867963)

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author = {Vladimir Khomyakov},
  title = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
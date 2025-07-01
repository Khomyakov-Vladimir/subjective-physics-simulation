# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This directory contains code and data for Version 3 of the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/
├── main.py # Run the full simulation (standard + adaptive)
├── observer.py # Observer with multiple adaptation modes
├── trade_off_functional_lambda_comparison.py
├── landauer_extension.py
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
│ └── energy_vs_epsilon.pdf
└── data/
  ├── tradeoff_data_lambda_comparison.csv
  ├── tradeoff_data_lambda1e20.csv
  ├── adaptive_metrics_entropy.npz
  └── adaptive_metrics_norm.npz
```

## Description

Version 3 introduces:
- Numerical simulation of the trade-off functional
  \[
    \mathcal{L}(\epsilon) = S(\epsilon) - \lambda \, E_{\text{disc}}(\epsilon)
  \]
  for different values of λ.
- Generation of CSV data tables and publication-quality figures.
- Extended analysis of the optimal perceptual threshold ε\*.
- Estimation of the minimal thermodynamic cost of entropy reduction using Landauer's principle.

## Files

- **trade_off_functional_lambda_comparison.py**  
  Computes and plots the trade-off functional for multiple λ.

- **landauer_extension.py**  
  Computes cognitive entropy reduction ΔS and corresponding minimal energetic cost per Landauer bound.  
  Produces:
  - `figures/deltaS_vs_epsilon.pdf`
  - `figures/energy_vs_epsilon.pdf`

- **adaptive_perceptual_dynamics.py**  
  Script generating a combined figure comparing perceptual threshold and cognitive entropy over adaptation steps for entropy-based and norm-based feedback modes.  
  Requires `adaptive_metrics_entropy.npz` and `adaptive_metrics_norm.npz` in the `data/` directory.
- **figures/**  
  Contains:
  - `L_epsilon_lambda_comparison_plot.pdf` – Main trade-off functional figure.
  - `adaptive_perceptual_dynamics.pdf` – Combined dynamics figure for publication.
- **data/**  
  Contains:
  - `tradeoff_data_lambda_comparison.csv` – Full data for all λ.
  - `tradeoff_data_lambda1e20.csv` – Data for λ=1e20.
  - `adaptive_metrics_entropy.npz` – Simulation results (entropy feedback).
  - `adaptive_metrics_norm.npz` – Simulation results (norm feedback).

## Running the simulation

1. Install dependencies (Python >=3.7):

```bash
pip install numpy matplotlib
```

2. Compute the trade-off functional:

```bash
python trade_off_functional_lambda_comparison.py
```

This will generate `L_epsilon_lambda_comparison_plot.pdf` and CSV data in `figures/` and `data/`.

3. Estimate Landauer energetic cost:

```bash
python landauer_extension.py
```

This script will:
- Compute ΔS and E_min for a range of ε.
- Save `deltaS_vs_epsilon.pdf` and `energy_vs_epsilon.pdf` in `figures/`.

4. Generate the combined adaptive dynamics figure:

```bash
python adaptive_perceptual_dynamics.py
```

The script will:
- Load simulation data from `data/`.
- Produce `adaptive_perceptual_dynamics.pdf` in `figures/`.
- Show the figure preview.

This figure can be included in the publication to illustrate the contrasting dynamics of perceptual threshold and entropy under different adaptation strategies.

## Citation

If you use this code, please cite the article:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.15780239]

---

## 📄 License

MIT License. See `LICENSE` file for details.

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author = {Vladimir Khomyakov},
  title = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
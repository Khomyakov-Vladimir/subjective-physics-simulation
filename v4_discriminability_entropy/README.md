# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15813188.svg)](https://doi.org/10.5281/zenodo.15813188)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This directory contains code and data for **Version 4** of the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/
├── main.py # Run the full simulation (standard + adaptive)
├── observer.py # Observer with multiple adaptation modes
├── trade_off_functional_lambda_comparison.py
├── adaptive_perceptual_dynamics.py
├── landauer_extension.py
├── compare_entropy_fixed_adaptive.py
├── von_neumann_entropy.py
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
│ └── epsilon_over_time.pdf
└── data/
  ├── tradeoff_data_lambda_comparison.csv
  ├── tradeoff_data_lambda1e20.csv
  ├── adaptive_metrics_entropy.npz
  └── adaptive_metrics_norm.npz
```


## Description

Version 4 introduces:

- Adaptive entropy suppression with an exponential threshold decay model:
  
  \[
  \epsilon(t) = \epsilon_0 \, \exp(-\lambda \, H(t))
  \]

- Analysis of phase transition behavior for large λ.
- Comparative simulations of adaptive vs. fixed thresholds.
- Additional figures visualizing entropy dynamics, perceptual threshold evolution, and trace distance over time.

## Files

- **compare_entropy_fixed_adaptive.py**  
  Generates comparative plots of entropy dynamics for adaptive and fixed threshold strategies:
  - `figures/entropy_comparison.pdf`
  - `figures/entropy_over_time.pdf`
  - `figures/epsilon_over_time.pdf`
  - `figures/trace_distance_over_time.pdf`

- **trade_off_functional_lambda_comparison.py**  
  Computes and plots the trade-off functional for multiple λ.

- **landauer_extension.py**  
  Computes cognitive entropy reduction ΔS and corresponding minimal energetic cost:
  - `figures/deltaS_vs_epsilon.pdf`
  - `figures/energy_vs_epsilon.pdf`

- **adaptive_perceptual_dynamics.py**  
  Generates combined dynamics figures for entropy-based and norm-based adaptation modes.

- **figures/**  
  Contains all publication-quality plots.

- **data/**  
  Includes simulation output files and CSV tables.

## Running the simulation

1. Install dependencies:

```bash
pip install numpy matplotlib
```

2. Run the comparative entropy simulation:

```bash
python compare_entropy_fixed_adaptive.py
```

This will produce:

* entropy_comparison.pdf

* entropy_over_time.pdf

* epsilon_over_time.pdf

* trace_distance_over_time.pdf

3. Compute the trade-off functional:

```bash
python trade_off_functional_lambda_comparison.py
```

4. Estimate Landauer energetic cost:

```bash
python landauer_extension.py
```

5. Generate adaptive dynamics figures:

```bash
python adaptive_perceptual_dynamics.py
```

## Citation

If you use this code, please cite the article:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.15813188]

---

## 📄 License

MIT License. See `LICENSE` file for details.

## 📜 Version History and DOI Links

For DOI versioning and archival history, see:

- [All versions](https://doi.org/10.5281/zenodo.15719389)
- [Version 1 only](https://doi.org/10.5281/zenodo.15719390)
- [Version 2 only](https://doi.org/10.5281/zenodo.15751229)
- [Version 3 only](https://doi.org/10.5281/zenodo.15780239)
- [Version 4 only (current)](https://doi.org/10.5281/zenodo.15813188)

👉 **View this version on Zenodo:**  
[https://doi.org/10.5281/zenodo.15813188](https://doi.org/10.5281/zenodo.15813188)

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author = {Vladimir Khomyakov},
  title = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
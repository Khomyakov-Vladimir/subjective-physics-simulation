# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains simulation code and data supporting the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**  
by Vladimir Khomyakov  
(Zenodo DOI: [10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389))

---

## Versions

- **v1_baseline/** — Initial minimal observer entropy simulation.
- **v2_adaptive_thresholds/** — Adaptive perceptual threshold ε(t) and extended visualizations.
- **v3_tradeoff_functional/** — Trade-off functional simulation with λ-parameter analysis, Landauer energetic cost analysis.
- **v4_discriminability_entropy/** — Adds adaptive entropy suppression, dynamic perceptual thresholds, phase transition tracking, and multi-condition comparisons.
- **v5_entropy_rt_coupling/** — Models the coupling between subjective entropy and reaction time under Dirichlet uncertainty; includes large-scale simulation, entropy–RT correlation, and confidence interval estimation.
- **v6_cognitive_geodesics/** — Introduces geodesic simulation in cognitive metric space, action-based dynamics, and curvature-driven discriminability analysis; implements cognitive trajectory integration and entropy functional regularization.

---

## Main Features

- Cognitive entropy model with geodesic integration  
- Landauer-bound energy dissipation under cognitive constraints  
- Subjective metric tensor \(\mathcal{G}_{ij}(\delta)\) and curvature effects  
- Trade-off functional and cognitive action computation  
- Thermodynamic cost estimation from observer-centric perspective  
- Publication-ready figures and data tables  

---

## 🔧 Installation

To install all required dependencies for **all published versions (v1–v6)** of the article:

```
pip install -r requirements.txt
```

The `requirements.txt` file specifies the minimal set of Python packages needed to reproduce all simulations, figures, and numerical results described in the following publication:

> Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics*. Zenodo. [https://doi.org/10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389)

---

## How to Run

Each version directory (e.g., `v3_tradeoff_functional/`) contains its own `README.md` describing how to:

- reproduce the key results;  
- rerun simulations;  
- regenerate all figures and data exports.  

---

## DOI Versioning and Archive

- [All versions](https://doi.org/10.5281/zenodo.15719389)  
- [Version 1 only](https://doi.org/10.5281/zenodo.15719390)  
- [Version 2 only](https://doi.org/10.5281/zenodo.15751229)  
- [Version 3 only](https://doi.org/10.5281/zenodo.15780239)  
- [Version 4 only](https://doi.org/10.5281/zenodo.15813188)  
- [Version 5 only](https://doi.org/10.5281/zenodo.15867963)  
- [Version 6 only](https://doi.org/10.5281/zenodo.XXXXXXXX)  

---

## 📜 License

MIT License (see individual LICENSE files per version).

---

## 📖 Citation

Use the corresponding BibTeX entry from each version’s `README.md` or `CITATION.cff`.

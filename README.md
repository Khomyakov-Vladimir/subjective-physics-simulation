# Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains simulation code and data supporting the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**  
by Vladimir Khomyakov  
(Zenodo DOI: [10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389))

---

## Versions

- **v1_entropy_hierarchy/** — Initial minimal observer entropy simulation.
- **v2_adaptive_thresholds/** — Adaptive perceptual threshold ε(t) and extended visualizations.
- **v3_tradeoff_functional/** — Trade-off functional simulation with λ-parameter analysis, Landauer energetic cost analysis.
- **v4_discriminability_entropy/** — Adds adaptive entropy suppression, dynamic perceptual thresholds, phase transition tracking, and multi-condition comparisons.
- **v5_entropy_rt_coupling/** — Models the coupling between subjective entropy and reaction time under Dirichlet uncertainty; includes large-scale simulation, entropy–RT correlation, and confidence interval estimation.
- **v6_cognitive_geodesics/** — Introduces geodesic simulation in cognitive metric space, action-based dynamics, and curvature-driven discriminability analysis; implements cognitive trajectory integration and entropy functional regularization.
- **v7_cognitive_reconstruction/** — Introduces cognitive retrodiction as a boundary value problem minimizing retrodictive entropy. Implements:
  - Damped geodesic simulation of cognitive trajectories using quadratic potential **V(y; B) = (y − B)²**
  - Entropy reduction analysis **ΔH = H(A) − H(A|B)** under belief intervention
  - Visualization of reconstructed cognitive states, entropy flow, and potential landscapes
  - Simulation scripts: `cognitive_entropy_reduction_simulation.py`, `cognitive_retrodiction_simulation.py`
- **v7.4_noise_augmented/** — Adds noise-augmented cognitive retrodiction under uncertainty in final observations:
  - `noise_dynamics_simulation.py` — explores perturbed final conditions **B′ = B + δ**  
  - `retrodiction_noise_variation.py` — simulates reconstructions from noisy boundaries  
  - Generates figures: `noise_dynamics.pdf`, `cog_reconstruction_noise.pdf`
- **v8_cognitive_dynamics/** — Introduces a fully dynamical framework for subjective physics based on cognitive entropy filtering, Σ-projection, and feedback-driven evolution:
  - `cognitive_decoherence_with_sigma.py` — simulates dynamic evolution of projected cognitive states under entropy-weighted filtering and boundary conditions; includes Σ-projection and parameter dependency analysis (region size, field types, and boundary conditions)
  - `dynamic_weight_feedback_enhanced.py` — implements cognitive feedback loops with bifurcation mechanisms, retrospection window for future prediction, and adaptive reconfiguration under entropy/flux constraints
  - Generates article figures: `sigma_projection_result.pdf`, `dynamic_evolution.gif`, `parameter_study.pdf`, `dynamic_weight_feedback_results.pdf`, and `geometry_effects.pdf`
- **v9_dynamic_phase_portrait/** — Introduces entropy-driven cognitive phase space simulation with stochastic jumps, EEG feedback, and fluctuation-theorem compliance:
  - `phase_portrait.py` — simulates 3D/4D trajectories with Tsallis entropy, entropy gradient dynamics, stochastic cognitive jumps with ΔE and **P₊/P₋** annotations, and EEG synchronization
  - `plot_entropy_flux_and_jumps.py` — plots time-resolved entropy flux, jump detection, and energy dissipation across perceptual transitions
  - Visualizes Lyapunov stability, energy thresholds, and observer weight evolution in subjective phase space
  - Generates figures: `subjective_phase_portrait.pdf`, `4d_phase_portrait.pdf`, `entropy_flux_and_jumps_real.pdf`
- **v10_multi_agent_shared_reality/** — Extends the framework to multi-agent systems with **M ≥ 2** and introduces a refined Shared Reality Index (SRI) that scales with variance across heterogeneous discrete state spaces. Features include:
  - Generalised multi-agent cognitive dynamics
  - Σ-projection with expectation-level alignment
  - Dual diagnostics: distributional overlap **A(t)** and expectation alignment (SRI)
  - Overlap matrix visualization and PCA-projected cognitive trajectories
  - Provides operational markers for intersubjective convergence and shared reality constitution
- **v11_core_observer_entropy/** — Consolidates the framework into a minimal, self-contained formalism unifying entropy scaling, Σ-projection, and multi-agent dynamics. Provides:
  - Core definitions: projection operator **Fε**, observer entropy **S(ε)**, and trade-off functional **L(ε)**
  - Numerical experiments: entropy scaling, adaptive thresholds, RT distributions, and convergence in multi-agent settings.

- **Documentation Split**  
  - [Main article (v11.2, PDF)](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v11.2/subjective_physics_simulation_v11.2_main_article.pdf) — consolidated article  
  - [Extended notes (v11.2, PDF)](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v11.2/subjective_physics_simulation_v11.2_extended_notes.pdf) — supplementary material (retrodiction, weak values, EEG analogies, cultural variation)  
  - Serves as the **stable reference version** for future theoretical and experimental work.

Each version folder (e.g., `v1_entropy_hierarchy/`) contains a complete and self-contained implementation of that version's simulations.  
For example, to reproduce all three main plots from **version 1**, run `main.py` inside `v1_entropy_hierarchy/`:

```bash
cd v1_entropy_hierarchy
python main.py
```

This will generate:
- `entropy_vs_epsilon.pdf`
- `norm_vs_time.pdf`
- `trace_distance_vs_epsilon.pdf`

All dependencies are resolved via the shared Conda environment defined in `environment.yml`.

---

## Main Features

- Cognitive entropy model with geodesic integration  
- Landauer-bound energy dissipation under cognitive constraints  
- Subjective metric tensor **𝒢ᵢⱼ(δ)** and curvature effects  
- Trade-off functional and cognitive action computation  
- Thermodynamic cost estimation from observer-centric perspective  
- Noise-augmented cognitive reconstruction under boundary uncertainty  
- Publication-ready figures and data tables

---

## 🔧 Installation

To install all required dependencies for **all published versions (v1–v11.2.4 (technical core edition))** of the article:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file specifies the minimal set of Python packages needed to reproduce all simulations, figures, and numerical results described in the following publication:

> Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics*. Zenodo. [https://doi.org/10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389)

## Python Environment

All scripts in versions **v1–v11.2.4 (technical core edition)** are fully reproducible using the following Conda environment:

```yaml
name: cogfun
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11.7
  - numpy=2.2.5
  - scikit-learn=1.6.1
  - matplotlib=3.10.3
  - pandas=2.2.3
  - pytorch=2.3.0
  - networkx=3.3
  - pygame=2.6.1
  - pip=24.0
  - pip:
    - galois==0.4.6
    - ogb==1.3.6
    - umap-learn==0.5.7
    - tqdm==4.67.1
    - torch-geometric==2.5.0
    - pytest==7.4.4
```

You can activate this environment with:

```bash
conda env create -f environment.yml
conda activate cogfun
```

The file `environment.yml` is included in the root of this repository.

---

## How to Run

Each version directory (e.g., `v3_tradeoff_functional/`) contains its own `README.md` describing how to:

- Reproduce the key results  
- Rerun simulations  
- Regenerate all figures and data exports  

---

## DOI Versioning and Archive

- [All versions](https://doi.org/10.5281/zenodo.15719389)
- [Version 1 only](https://doi.org/10.5281/zenodo.15719390)
- [Version 2 only](https://doi.org/10.5281/zenodo.15751229)
- [Version 3 only](https://doi.org/10.5281/zenodo.15780239)
- [Version 4 only](https://doi.org/10.5281/zenodo.15813188)
- [Version 5 only](https://doi.org/10.5281/zenodo.15867963)
- [Version 6 only](https://doi.org/10.5281/zenodo.16028303)
- [Version 7 only](https://doi.org/10.5281/zenodo.16368499)
- [Version 7.4 only](https://doi.org/10.5281/zenodo.16478500)
- [Version 8.0 only](https://doi.org/10.5281/zenodo.16728290)
- [Version 9.0 only](https://doi.org/10.5281/zenodo.16741400)
- [Version 10.0 only](https://doi.org/10.5281/zenodo.16888675)
- [Version 11.0 only](https://doi.org/10.5281/zenodo.17234483)
- [Version 11.1 only](https://doi.org/10.5281/zenodo.17252212)
- [Version 11.2.4 (technical core edition) only](https://doi.org/10.5281/zenodo.17312704)

---

## 📜 License

MIT License (see individual LICENSE files per version).

---

## 📖 Citation

Use the corresponding BibTeX entry from each version’s `README.md` or `CITATION.cff`.

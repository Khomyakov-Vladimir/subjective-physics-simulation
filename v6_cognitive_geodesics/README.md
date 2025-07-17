# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16028303.svg)](https://doi.org/10.5281/zenodo.16028303)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains code and data for:

**Version 6** of the article  
**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

This version extends the previous simulations by introducing a geodesic formulation of cognitive dynamics, derived from a subjective metric on the space of distinctions.

---

## Directory Structure

```
subjective-physics-simulation/v6_cognitive_geodesics
├── main.py # Run the full simulation (standard + adaptive)
├── observer.py # Observer with multiple adaptation modes
├── trade_off_functional_lambda_comparison.py
├── adaptive_perceptual_dynamics.py
├── landauer_extension.py
├── compare_entropy_fixed_adaptive.py
├── von_neumann_entropy.py
├── simulate_entropy_rt_full.py
├── compare_dirichlet_params.py
├── cognitive_geodesic_simulation.py # Simulates geodesics numerically
├── cognitive_geodesic_trajectories.py # Visualizes cognitive trajectories
├── geodesic_dynamics_cognitive_action.py # Computes action along geodesics
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
│ ├── rt_boxplot.pdf
│ ├── cognitive_geodesic.pdf
│ ├── cognitive_trajectories.pdf
│ └── cognitive_action_vs_time.pdf
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
  ├── simulated_data_all_configs.csv
  ├── path1_dynamics.csv
  ├── all_paths_dynamics.json
  └── geodesic_paths_dynamics.json
```

## Description and Theoretical Context

## 🧠 New Features in Version 6

This release introduces a geodesic model of subjective physics based on the following principles:

- Weak values are interpreted as **thresholds of cognitive distinction**.
- A subjective metric is defined over distinction space \( \mathcal{M}_R \), including internal symmetry space.
- A geodesic action is computed based on entropic curvature and observer-dependent distinctions.

### 🧩 New Scripts Introduced in v6:

#### `geodesic_dynamics_cognitive_action.py`
Calculates the geodesic action functional along simulated trajectories in the distinction space. Includes computation of entropic terms and symbolic expressions used in the article.

#### `cognitive_geodesic_simulation.py`
Performs numerical integration of geodesics using the cognitive metric \( g_{ij}(\delta) \). Includes initial conditions and boundary-value formulation.

#### `cognitive_geodesic_trajectories.py`
Plots the geodesic paths and overlays cognitive phase-space structure for visual comparison. Results support the interpretation of cognitive distinctions as physically relevant dynamics.

---

## Files

**cognitive_geodesic_simulation.py**  
Runs the main simulation, saves raw data to CSV, and generates:  
- `figures/cognitive_geodesic.pdf`

**cognitive_geodesic_trajectories.py**  
Simulates multiple geodesic trajectories in the distinction space with anisotropic metric and potential.  
Saves detailed path data in:  
- `data/path1_dynamics.csv`  
- `data/all_paths_dynamics.json`  
Generates visualization:  
- `figures/cognitive_trajectories.pdf`

**geodesic_dynamics_cognitive_action.py**  
Computes and plots cognitive action \( S(t) \) along each geodesic path.  
Saves cumulative geodesic dynamics to:  
- `data/geodesic_paths_dynamics.json`  
Generates final figure:  
- `figures/cognitive_action_vs_time.pdf`

**figures/**  
Contains all plots used for article figures and geodesic visualization.

**data/**  
Includes simulation outputs and exported dynamics (CSV, JSON).

## Running the simulation

1. Install dependencies:

```bash
pip install numpy matplotlib scipy pandas tqdm
```

2. Run the basic geodesic simulation in cognitive distinction space:

```bash
python cognitive_geodesic_simulation.py
```

3. Run the full simulation of multiple geodesic trajectories and export dynamics:

```bash
python cognitive_geodesic_trajectories.py
```

4. Run the geodesic action analysis and generate \( S(t) \) plots:

```bash
python geodesic_dynamics_cognitive_action.py
```

---

## Citation

If you use this code, please cite the article:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (6.0)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.16028303]

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
- [Version 5 only](https://doi.org/10.5281/zenodo.15867963)
- [Version 6 only (current)](https://doi.org/10.5281/zenodo.16028303)

👉 **View this version on Zenodo:**  
[https://doi.org/10.5281/zenodo.16028303](https://doi.org/10.5281/zenodo.16028303)

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author = {Vladimir Khomyakov},
  title = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
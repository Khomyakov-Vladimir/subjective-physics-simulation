# Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (v8.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16728290.svg)](https://doi.org/10.5281/zenodo.16728290)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains code and data for:

**Version 8** of the article  
**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

This version extends the previous simulations by introducing a retrodictive reconstruction model, formulated as a boundary value problem minimizing cognitive entropy.

---

## Directory Structure

```
subjective-physics-simulation/v8_cognitive_dynamics
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
├── cognitive_entropy_reduction_simulation.py
├── cognitive_retrodiction_simulation.py
├── cognitive_decoherence_with_sigma.py
├── dynamic_weight_feedback_enhanced.py
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
│ ├── cognitive_action_vs_time.pdf
│ ├── cognitive_entropy_reduction_simulation.pdf
│ ├── phase_portrait.pdf
│ ├── potential_landscape.pdf
│ ├── retrodicted_states.pdf
│ ├── state_trajectories.pdf
│ ├── cognitive_filter_results.pdf
│ ├── sigma_projection_result.pdf
│ ├── dynamic_evolution.gif
│ ├── parameter_study.pdf
│ ├── geometry_effects.pdf
│ └── dynamic_weight_feedback_results.pdf
├── results/
│ └── run_20250801_153830/
│   ├── main_article_figure.pdf
│   ├── simulation_data.npz
│   └── state_evolution.pdf
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
  ├── geodesic_paths_dynamics.json
  └── cognitive_filter_data.npz
```

## 📂 File Descriptions

Below is a complete description of all scripts included in this version. These files form a **self-contained and reproducible package** supporting the simulations, figures, and data analyses presented in the article.

### 🆕 New in Version 8

- **`cognitive_decoherence_with_sigma.py`**  
  Simulates entropy-based filtering of candidate cognitive field configurations, performs Σ-projection to select an observer-consistent state, and evolves the selected state under boundary conditions and stochastic fluctuations.  
  Includes parameter studies for:
  - observation region size  
  - boundary condition types  
  - input field structure  
  📊 Outputs:
  - `figures/cognitive_filter_results.pdf`  
  - `figures/sigma_projection_result.pdf`  
  - `figures/dynamic_evolution.gif`  
  - `figures/parameter_study.pdf`

- **`dynamic_weight_feedback_enhanced.py`**  
  Implements adaptive re-weighting of cognitive configurations under feedback from future-predicted entropy and flux values.  
  Includes:
  - retrospection window (for predictive feedback)  
  - cognitive jump bifurcations when configuration weights fall below a threshold  
  - structural geometry comparison for flux sensitivity  
  📈 Outputs:
  - `figures/dynamic_weight_feedback_results.pdf`  
  - `figures/geometry_effects.pdf`  
  - `results/run_*/state_evolution.pdf`

### 🆕 New in Version 7

- **`cognitive_entropy_reduction_simulation.py`**  
  Simulates entropy reduction \( \Delta H = H(A) - H(A|B) \) as a function of belief prior \( p_1 \).  
  📈 Output: `figures/cognitive_entropy_reduction_simulation.pdf`

- **`cognitive_retrodiction_simulation.py`**  
  Solves the damped boundary value problem modeling cognitive retrodiction as a geodesic entropy-reducing process.  
  📊 Outputs:  
  - `figures/state_trajectories.pdf`  
  - `figures/retrodicted_states.pdf`  
  - `figures/phase_portrait.pdf`  
  - `figures/potential_landscape.pdf`

## 🧠 Key Insight: Quantum Interference Reinterpreted

> "Interference is not a real 'jumping' of the photon, but a consequence of how our perception system reconstructs the past under uncertainty."

This reinterpretation is grounded in the framework of **Subjective Physics**, where quantum phenomena are not treated as fundamentally indeterministic, but as the result of **cognitive reconstruction** by a bounded observer. The presented model demonstrates how quantum-like effects — such as interference and apparent retrocausality — can emerge from **entropy-minimizing retrodictive inference**. In this approach, the observer's perceptual system reconstructs the most probable sequence of events (a cognitive geodesic) consistent with the final observation, thereby creating the illusion of "strange" behavior in quantum experiments.

### 🧠 From Version 6 (Retained)

- **`main.py`**  
  Central launcher script. Runs full simulation pipeline with standard and adaptive modes.

- **`observer.py`**  
  Core definition of the cognitive observer model supporting adaptive, fixed, and stochastic modes of inference.

- **`trade_off_functional_lambda_comparison.py`**  
  Compares different values of the trade-off functional \( L_\lambda \) for entropy-norm optimization.  
  📈 Output: `figures/L_epsilon_lambda_comparison_plot.pdf`

- **`adaptive_perceptual_dynamics.py`**  
  Models adaptive evolution of perceptual thresholds \( \varepsilon(t) \).  
  📊 Outputs:  
  - `figures/adaptive_perceptual_dynamics.pdf`  
  - `figures/adaptive_threshold_entropy.pdf`  
  - `figures/adaptive_threshold_norm.pdf`

- **`landauer_extension.py`**  
  Computes energy-entropy trade-offs based on Landauer’s principle generalized to subjective cognitive transitions.  
  📈 Outputs:  
  - `figures/deltaS_vs_epsilon.pdf`  
  - `figures/energy_vs_epsilon.pdf`

- **`compare_entropy_fixed_adaptive.py`**  
  Compares entropy dynamics between fixed and adaptive regimes.  
  📊 Outputs:  
  - `figures/entropy_comparison.pdf`  
  - `figures/entropy_over_time.pdf`  
  - `figures/epsilon_over_time.pdf`

- **`von_neumann_entropy.py`**  
  Calculates von Neumann entropy and trace distance in simulated observer states.  
  📈 Outputs:  
  - `figures/entropy_vs_epsilon.pdf`  
  - `figures/trace_distance_vs_epsilon.pdf`

- **`simulate_entropy_rt_full.py`**  
  Generates histograms and cumulative distributions of reaction times (RT) and entropy over trials.  
  📊 Outputs:  
  - `figures/histogram_rt.pdf`  
  - `figures/scatter_entropy_rt.pdf`  
  - `figures/cdf_rt.pdf`  
  - `figures/entropy_hist_comparison.pdf`  
  - `figures/rt_cdf_comparison.pdf`  
  - `figures/entropy_boxplot.pdf`  
  - `figures/rt_boxplot.pdf`

- **`compare_dirichlet_params.py`**  
  Compares simulation outcomes across Dirichlet prior configurations.  
  📁 Outputs:  
  - `data/simulated_data_dirichlet_*.csv`  
  - `data/simulated_data_all_configs.csv`

- **`cognitive_geodesic_simulation.py`**  
  Numerically solves geodesic paths in perceptual metric space.  
  📈 Output: `figures/cognitive_geodesic.pdf`

- **`cognitive_geodesic_trajectories.py`**  
  Visualizes cognitive trajectories in observer belief space.  
  📊 Output: `figures/cognitive_trajectories.pdf`

- **`geodesic_dynamics_cognitive_action.py`**  
  Computes cognitive action and its temporal evolution along geodesic paths.  
  📈 Output: `figures/cognitive_action_vs_time.pdf`

---

## How to Run

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

Run simulations:

```bash
python cognitive_entropy_reduction_simulation.py
python cognitive_retrodiction_simulation.py
```

---

## Citation

If you use this code, please cite the article:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (8.0)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.16728290]

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
- [Version 6 only](https://doi.org/10.5281/zenodo.16028303)
- [Version 7 only](https://doi.org/10.5281/zenodo.16368499)
- [Version 7.4 only](https://doi.org/10.5281/zenodo.16478500)
- [Version 8 only (current)](https://doi.org/10.5281/zenodo.16728290)

👉 **View this version on Zenodo:**  
[https://doi.org/10.5281/zenodo.16728290](https://doi.org/10.5281/zenodo.16728290)

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author = {Vladimir Khomyakov},
  title = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
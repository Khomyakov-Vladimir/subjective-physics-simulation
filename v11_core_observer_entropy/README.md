# Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (v11.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17234483.svg)](https://doi.org/10.5281/zenodo.17234483)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains code and data for:

**Version 11** of the article  
**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

This version consolidates earlier developments (v5–v10) into a reproducible, information-theoretic **minimal model** of Subjective Physics.  
It integrates:
- Entropy scaling and reaction-time coupling (v5)  
- Cognitive decoherence and Σ-projection (v9)  
- Multi-agent Shared Reality Index (v10)  

The v11 release establishes the **core observer entropy formalism** and serves as the stable reference point for future work.

---

## Directory Structure

```
subjective-physics-simulation/v11_core_observer_entropy/
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
├── phase_portrait.py
├── plot_entropy_flux_and_jumps.py
├── cognitive_decoherence_with_sigma_multi_agent_system.py
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
│ ├── dynamic_weight_feedback_results.pdf
│ ├── subjective_phase_portrait.pdf
│ ├── 4d_phase_portrait.pdf
│ ├── entropy_flux_and_jumps_real.pdf
│ ├── parameter_study_v10.pdf
│ ├── multi_agent_projection_v10.pdf
│ ├── shared_overlap_matrix.pdf
│ ├── sri_time_series.pdf
│ └── agent_trajectories_pca.pdf
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

## File Descriptions

Below is a complete description of all scripts included in this version. These files form a **self-contained and reproducible package** supporting the simulations, figures, and data analyses presented in the article.

## New in Version 11

- **Minimal Model Formalism**  
  Establishes a compact, self-contained framework unifying entropy scaling, Σ-projection, and multi-agent dynamics.  
  Core components:  
  - Projection operator \( F_\epsilon \)  
  - Observer entropy \( S(\epsilon) \)  
  - Thermodynamic trade-off functional \( L(\epsilon) \)  

- **Numerical Experiments**  
  - Entropy scaling and adaptive thresholds  
  - Reaction-time distributions and entropy coupling  
  - Multi-agent convergence dynamics under Σ-projection  

- **Documentation Split**  
  - [Main article (v11.0, PDF)](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v11.0/subjective_physics_simulation_v11.0_main_article.pdf) — consolidated article  
  - [Extended notes (v11.0, PDF)](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v11.0/subjective_physics_simulation_v11.0_extended_notes.pdf) — supplementary material (retrodiction, weak values, EEG analogies, cultural variation)  
 
- **Stable Reference Version**  
  Designated as the **stable baseline** for theoretical and experimental extensions of Subjective Physics.

---

### Quick Start: Core Workflow (v11.0)

To reproduce the **main article results** (`v11.0_main_article.pdf`),  
run the following scripts **in order**:

| Version | Core Script | Figures Generated |
|---------|-------------|-------------------|
| v5.0 | `simulate_entropy_rt_full.py` | histogram_rt.pdf, scatter_entropy_rt.pdf, cdf_rt.pdf |
| v8.0 | `cognitive_decoherence_with_sigma.py` | sigma_projection_result.pdf, cognitive_filter_results.pdf |
| v10.0 | `cognitive_decoherence_with_sigma_multi_agent_system.py` | sri_time_series.pdf, shared_overlap_matrix.pdf, agent_trajectories_pca.pdf |

Running these three steps is sufficient to regenerate all **main figures** of version 11.0.

---

## Structure of Version 11

Version 11 consolidates **all previous components** for reproducibility.
Conceptually, it should be understood as:

- **Core (Baseline Minimal Model):**
  Observer entropy formalism, entropy–RT scaling, Σ-projection as the universal filtering principle.

- **Extensions (Retained for Reproducibility):**
  Multi-agent dynamics (v10), retrodiction and weak values (v7), EEG-inspired phase portraits (v9), and other exploratory modules.  

This separation highlights that v11 primarily refactors and stabilizes the **core formalism**, 
while keeping prior exploratory features available for replication.

### Note  
“Baseline Minimal Model” here refers to the main article (`v11.0_main_article`),  
which supersedes the earlier baseline established in version 5.0 (`v5_entropy_rt_coupling`).  
This ensures continuity of the minimal model concept across versions.

---
## Core vs Extensions (Version 11.0)

To support both **scientific reproducibility** and **practical verification**,  
Version 11.0 defines a clear separation between the **Core workflow** and **Extensions**.

### Core Reproducibility Workflow
The **core** consists of the minimal set of scripts required to reproduce  
the main results of the article (`v11.0_main_article.pdf`).  
See the [Quick Start](#quick-start-core-workflow-v110) section above for the exact order of execution and generated figures.

### Extensions (v11.0_extended_notes) (for completeness and archival)
 All other scripts and figures remain available in the repository.  
 They cover exploratory modules from earlier versions (retrodiction v7, geodesics v6, EEG-inspired phase portraits v9, parameter studies v10, etc.).  

These **extensions (v11.0_extended_notes)** are not required to reproduce the article’s results,  
but are preserved to ensure full scientific transparency and repeatability.

| Version | Core Script | Figures Generated |
|---------|-------------|-------------------|
| v1.0–v2.0 | `main.py` | entropy_vs_epsilon.pdf, norm_vs_time.pdf, trace_distance_vs_epsilon.pdf (+ adaptive_* in v2) |
| v3.0 | `trade_off_functional_lambda_comparison.py`<br>`landauer_extension.py`<br>`adaptive_perceptual_dynamics.py` | L_epsilon_lambda_comparison_plot.pdf, deltaS_vs_epsilon.pdf, energy_vs_epsilon.pdf, adaptive_perceptual_dynamics.pdf, adaptive_threshold_entropy.pdf, adaptive_threshold_norm.pdf |
| v4.0 | `compare_entropy_fixed_adaptive.py` | entropy_comparison.pdf, entropy_over_time.pdf, epsilon_over_time.pdf, trace_distance_over_time.pdf |
| v5.0 | `simulate_entropy_rt_full.py` | histogram_rt.pdf, scatter_entropy_rt.pdf, cdf_rt.pdf |
| v8.0 | `dynamic_weight_feedback_enhanced.py`<br>`cognitive_decoherence_with_sigma.py` | parameter_study.pdf, geometry_effects.pdf, dynamic_weight_feedback_results.pdf<br>sigma_projection_result.pdf, cognitive_filter_results.pdf |
| v10.0 | `cognitive_decoherence_with_sigma_multi_agent_system.py` | sri_time_series.pdf, shared_overlap_matrix.pdf, agent_trajectories_pca.pdf |

Running these scripts in order (v5.0 → v8.0 → v10.0) is sufficient to reproduce the  
**main results** of version 11.0.

### Extensions (v11.0_extended_notes)
The **extensions** include all other scripts and figures, preserved for completeness and archival.  
They are not required to reproduce the **main article**, but ensure full transparency.

| Version | Extension Script | Figures Generated |
|---------|------------------|-------------------|
| v5.0 | `compare_dirichlet_params.py` | entropy_hist_comparison.pdf, rt_cdf_comparison.pdf, entropy_boxplot.pdf, rt_boxplot.pdf |
| v6.0 | Geodesic / trajectory modules | cognitive_geodesic.pdf, cognitive_trajectories.pdf |
| v7.0 | Retrodiction / weak values | retrodicted_states.pdf, cog_reconstruction_noise.pdf |
| v7.4 | `noise_dynamics_simulation.py` | noise_dynamics.pdf |
| v9.0 | EEG-inspired modules | entropy_flux_and_jumps_real.pdf, subjective_phase_portrait.pdf, 4d_phase_portrait.pdf |
| v10.0 | Parameter studies | parameter_study_v10.pdf, multi_agent_projection_v10.pdf |
| v11.0 | archived exploratory material | `v11.0_extended_notes.pdf` |

## Evolution of Core vs Extensions (v1.0 → v11.0)

The following table tracks the development of scripts, their classification into  
**Core** (minimal reproducible set for main articles) and **Extensions** (exploratory / supplementary).  
This ensures transparent scientific reproducibility across all versions.

| Version | Core Scripts / Figures | Extension Scripts / Figures |
|---------|------------------------|-----------------------------|
| **v1.0** | `main.py`, `observer.py` → `entropy_vs_epsilon.pdf`, `norm_vs_time.pdf`, `trace_distance_vs_epsilon.pdf` | – |
| **v2.0** | (same as v1.0) | Adaptive mode integrated in `main.py` → `adaptive_entropy_*`, `adaptive_norm_*`, `adaptive_threshold_*`, `adaptive_trace_distance_*` |
| **v3.0** | (same as v2.0) | `trade_off_functional_lambda_comparison.py` → `L_epsilon_lambda_comparison_plot.pdf`; `landauer_extension.py` → `deltaS_vs_epsilon.pdf`, `energy_vs_epsilon.pdf`; `adaptive_perceptual_dynamics.py` → `adaptive_perceptual_dynamics.pdf`; `adaptive_threshold_entropy.pdf`; `adaptive_threshold_norm.pdf` |
| **v4.0** | (same as v3.0) | `compare_entropy_fixed_adaptive.py` → `entropy_comparison.pdf`, `entropy_over_time.pdf`, `epsilon_over_time.pdf`, `trace_distance_over_time.pdf`; `von_neumann_entropy.py` |
| **v5.0** | `simulate_entropy_rt_full.py` → `histogram_rt.pdf`, `scatter_entropy_rt.pdf`, `cdf_rt.pdf` | `compare_dirichlet_params.py` → `entropy_hist_comparison.pdf`, `rt_cdf_comparison.pdf`, `entropy_boxplot.pdf`, `rt_boxplot.pdf` + all v1–v4 extensions |
| **v6.0** | (inherits v5.0 core) | Geodesic & state-trajectory modules → `cognitive_geodesic.pdf`, `state_trajectories.pdf`, etc. |
| **v7.0** | (inherits v5.0 core) | Retrodiction / weak values → `retrodicted_states.pdf`, related scripts |
| **v8.0** | `cognitive_decoherence_with_sigma.py` → `sigma_projection_result.pdf`, `cognitive_filter_results.pdf`  | Cultural / contextual extensions |
| **v9.0** | EEG-inspired modules → `subjective_phase_portrait.pdf`, `cognitive_trajectories.pdf`, `cog_reconstruction_noise.pdf` |
| **v10.0** | `cognitive_decoherence_with_sigma_multi_agent_system.py` → `sri_time_series.pdf`, `shared_overlap_matrix.pdf`, `agent_trajectories_pca.pdf` | Parameter studies → `parameter_study.pdf`, `parameter_study_v10.pdf`; other refinements |
| **v11.0** | **Core Workflow = { v5.0 + v9.0 + v10.0 }**: <br>• `simulate_entropy_rt_full.py` (RT scaling)<br>• `cognitive_decoherence_with_sigma.py` (Σ-projection)<br>• `cognitive_decoherence_with_sigma_multi_agent_system.py` (multi-agent SRI)<br>→ Figures: `histogram_rt.pdf`, `scatter_entropy_rt.pdf`, `cdf_rt.pdf`, `sigma_projection_result.pdf`, `cognitive_filter_results.pdf`, `sri_time_series.pdf`, `shared_overlap_matrix.pdf`, `agent_trajectories_pca.pdf` | All other legacy/archived scripts and figures moved to **`v11.0_extended_notes.pdf`** |

---

## Version 11 vs Version 10

| Feature | Version 10 | Version 11 |
|---------|-------------|-------------|
| Multi-agent Σ-projection | ✅ | ✅ (consolidated into core model) |
| Shared Reality Index (SRI) | ✅ (refined) | ✅ (integrated) |
| Retrodiction & weak values | Supplementary | Supplementary (archived in Extended Notes) |
| EEG-inspired phase portraits | Experimental | Supplementary (archived) |
| Stable baseline formalism | ❌ | ✅ |

---

## Key Enhancements in Version 11

- Unified entropic and projection-based formalism.  
- Explicit connection between entropy scaling, RT distributions, and observer thresholds.  
- Consolidated simulation scripts for reproducibility.  
- Extended notes moved to archival supplement.  

---

## New in Version 10

- **`cognitive_decoherence_with_sigma_multi_agent_system.py`**  
  Simulates cognitive filtering, Σ-projection, and dynamic evolution in a **multi-agent system** with inter-subjective agreement metrics. Implements:  
  - Generalised multi-agent dynamics ($M \geq 2$)  
  - Entropic dissipation ($\alpha$) and symmetric coupling ($\gamma$)  
  - Refined *Shared Reality Index (SRI)* with variance-scaled normalisation  
  - Agreement metric $A(t)$ for distributional similarity  
  - Overlap matrix $O_{ab}$ at final time  
  - PCA-based embedding of agent trajectories  
  Outputs:
  - `figures/sri_time_series.pdf`  
  - `figures/shared_overlap_matrix.pdf`  
  - `figures/agent_trajectories_pca.pdf`  
  - `figures/multi_agent_projection_v10.pdf`  
  - `figures/parameter_study_v10.pdf`

---

### Key Enhancements in Version 10

Version 10 extends the framework with **multi-agent shared reality dynamics**:

- **Generalised Inter-Agent Evolution**: $M \geq 2$ agents jointly update posteriors $w^{(a)}(t)$ under coupling and entropic dissipation.  
- **Refined Consensus Metrics**:  
  - $A(t)$ — distributional alignment (pointwise similarity).  
  - SRI — centroid-level alignment with variance scaling, robust across heterogeneous state spaces.  
  - $O_{ab}$ — final-time overlap quantifying pairwise coincidences.  
- **Visual Diagnostics**:  
  - Dual-axis plot of $A(t)$ vs SRI.  
  - Heatmap of $O_{ab}$ at consensus time.  
  - PCA-projected trajectories of agent expectations.  
- **Reproducibility**: fixed seeds, unique filenames for archival figures.  
- **Parameter Study**: entropy sensitivity to observation region, boundary conditions, and state generation types.

**Simulation parameters (default):**  
\[
N = 7,\quad M = 5,\quad \alpha = 0.4,\quad \gamma = 0.75,\quad T = 8.0,\quad \beta = 0.1
\]

---

### Output Figures (v10 additions):

- `figures/sri_time_series.pdf` — SRI vs $A(t)$ trajectories.  
- `figures/shared_overlap_matrix.pdf` — final-time overlap heatmap.  
- `figures/agent_trajectories_pca.pdf` — agent convergence in PCA space.  
- `figures/multi_agent_projection_v10.pdf` — Σ-projection of representative final states.  
- `figures/parameter_study_v10.pdf` — entropy, leakage, and variance across conditions.

---

### Limitations

- Current agent coupling is **symmetric**; future work may extend to heterogeneous networks (graph Laplacian coupling).  
- PCA is a **linear embedding**; nonlinear manifold learning could refine trajectory visualisation.  
- SRI is variance-based; further extensions could include higher-order moments of expectation distributions.  

---

### Version 10 vs Version 9

| Feature | Version 9 | Version 10 |
|---------|-----------|-------------|
| Σ-projection filtering | ✅ | ✅ |
| Stochastic jumps & entropy flux | ✅ | ❌ (not primary focus) |
| EEG feedback integration | ✅ | ❌ |
| 4D phase portraits | ✅ | ❌ |
| Multi-agent dynamics ($M\geq 2$) | ❌ | ✅ |
| Refined SRI with variance scaling | ❌ | ✅ |
| Overlap matrix diagnostic | ❌ | ✅ |
| PCA trajectory embedding | ❌ | ✅ |
| Parameter study module | ❌ | ✅ |

## New in Version 9

- **`phase_portrait.py`**  
  Simulates and visualizes cognitive phase space under entropy gradients and stochastic jumps. Implements:
  - Tsallis entropy dynamics  
  - EEG coupling (real or synthetic)  
  - Cognitive jumps with ΔE and fluctuation ratio annotations  
  - Lyapunov stability visualization  
  - Quantum-weighted trajectory evolution  
  - 3D and 4D dynamic phase portraits  
  Outputs:
  - `figures/subjective_phase_portrait.pdf`  
  - `figures/4d_phase_portrait.pdf`

- **`plot_entropy_flux_and_jumps.py`**  
  Plots temporal evolution of cognitive entropy and energy dissipation during perceptual transitions. Features:
  - Tsallis entropy flux over time  
  - Energy-based jump detection (ΔE)  
  - Fluctuation theorem verification (P₊/P₋)  
  - Dual-axis annotated visualization  
  Output:
  - `figures/entropy_flux_and_jumps_real.pdf`

---

### Key Enhancements in Version 9

Version 9 introduces a multidimensional simulation architecture capturing **dynamic cognitive state evolution**:

- **Quantum Cognition**: Interference, phase noise, and spontaneous collapse simulated via stochastic jump events.
- **Entropy Models**: Supports Tsallis (\(q = 1.2\)), Shannon, and Rényi entropy for subjective dynamics.
- **Neurocognitive Integration**: EEG attention data (real or synthetic) is used to modulate and synchronize observer weights \(w(t)\), allowing correlation tracking and validation.
- **Jump Transitions**: Phase-space jumps are annotated with energy changes ΔE and fluctuation ratios \(P_+/P_-\), supporting nonequilibrium thermodynamics.
- **4D Visualization**: Subjective phase portraits rendered with cognitive weight \(w(t)\), entropy \(S_q(\varepsilon)\), energy \(E_{\text{disc}}(\varepsilon)\), and EEG-based attention as color code.
- **Stability Analysis**: Local Lyapunov exponents computed dynamically, allowing exploration of system sensitivity.

**Equations modeled in simulation**:
- Weight dynamics:
  \[
  \frac{dw}{dt} = -\alpha \cdot \Phi_\Sigma(t) \cdot w + \beta \cdot (1 - w)
  \]
- Entropy flux:
  \[
  \Phi_\Sigma(t) = -\frac{dS_q}{dt}, \quad S_q(\epsilon) = \frac{\epsilon^{1-q} - 1}{1 - q}
  \]
- Perceptual thresholds:
  \[
  \epsilon(t) = \text{mode-dependent adaptation (entropy, norm, or Σ-projection)}
  \]
- Energy cost:
  \[
  E_{\text{disc}}(\epsilon) = k_B T \log(2) \cdot \log_2\left(\frac{1}{\epsilon}\right)
  \]
- Jump fluctuations:
  \[
  \frac{P_+}{P_-} = \exp\left(\frac{\Delta E}{k_B T}\right)
  \]

---

### EEG Synchronization

The simulation integrates with `data/subjective_eeg.csv` when available. If missing, EEG is synthetically generated as:
\[
a(t) = 0.5 + 0.3 \sin(2\pi t) + 0.1 \cdot \mathcal{N}(0,1)
\]
The correlation coefficient \( \rho = \text{corr}(\hat{w}(t), \hat{a}(t)) \) is reported to validate cognitive-neuro coupling.

---

### Output Figures (v9 additions):

- `figures/subjective_phase_portrait.pdf`: Annotated 3D trajectory with ΔE and \(P_+/P_-\) markers  
- `figures/4d_phase_portrait.pdf`: Cognitive trajectory + EEG attention in 4D  
- `figures/entropy_flux_and_jumps_real.pdf`: Time evolution of entropy flux with jump annotations

---

### Limitations

- EEG coupling is linear; future work may introduce nonlinear (e.g., mutual information or transfer entropy) methods.
- Current visualization is static; real-time or interactive rendering is a future direction.
- Perceptual thresholds are parameter-driven; self-organizing thresholds could enhance realism.

---

### Version 9 vs Version 8

| Feature | Version 8 | Version 9 |
|--------|------------|-------------|
| Σ-projection filtering | ✅ | ✅ |
| Retrodictive entropy reconstruction | ✅ | ✅ |
| EEG feedback integration | ❌ | ✅ |
| Stochastic cognitive jumps | ❌ | ✅ |
| 4D visualization | ❌ | ✅ |
| Fluctuation theorem compliance | ❌ | ✅ |

### New in Version 8

- **`cognitive_decoherence_with_sigma.py`**  
  Simulates entropy-based filtering of candidate cognitive field configurations, performs Σ-projection to select an observer-consistent state, and evolves the selected state under boundary conditions and stochastic fluctuations.  
  Includes parameter studies for:
  - observation region size  
  - boundary condition types  
  - input field structure  
  Outputs:
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
  Outputs:
  - `figures/dynamic_weight_feedback_results.pdf`  
  - `figures/geometry_effects.pdf`  
  - `results/run_*/state_evolution.pdf`

### New in Version 7

- **`cognitive_entropy_reduction_simulation.py`**  
  Simulates entropy reduction \( \Delta H = H(A) - H(A|B) \) as a function of belief prior \( p_1 \).  
  Output: `figures/cognitive_entropy_reduction_simulation.pdf`

- **`cognitive_retrodiction_simulation.py`**  
  Solves the damped boundary value problem modeling cognitive retrodiction as a geodesic entropy-reducing process.  
  Outputs:  
  - `figures/state_trajectories.pdf`  
  - `figures/retrodicted_states.pdf`  
  - `figures/phase_portrait.pdf`  
  - `figures/potential_landscape.pdf`

## Key Insight: Quantum Interference Reinterpreted

> "Interference is not a real 'jumping' of the photon, but a consequence of how our perception system reconstructs the past under uncertainty."

This reinterpretation is grounded in the framework of **Subjective Physics**, where quantum phenomena are not treated as fundamentally indeterministic, but as the result of **cognitive reconstruction** by a bounded observer. The presented model demonstrates how quantum-like effects — such as interference and apparent retrocausality — can emerge from **entropy-minimizing retrodictive inference**. In this approach, the observer's perceptual system reconstructs the most probable sequence of events (a cognitive geodesic) consistent with the final observation, thereby creating the illusion of "strange" behavior in quantum experiments.

### From Version 6 (Retained)

- **`main.py`**  
  Central launcher script. Runs full simulation pipeline with standard and adaptive modes.

- **`observer.py`**  
  Core definition of the cognitive observer model supporting adaptive, fixed, and stochastic modes of inference.

- **`trade_off_functional_lambda_comparison.py`**  
  Compares different values of the trade-off functional \( L_\lambda \) for entropy-norm optimization.  
  Output: `figures/L_epsilon_lambda_comparison_plot.pdf`

- **`adaptive_perceptual_dynamics.py`**  
  Models adaptive evolution of perceptual thresholds \( \varepsilon(t) \).  
  Outputs:  
  - `figures/adaptive_perceptual_dynamics.pdf`  
  - `figures/adaptive_threshold_entropy.pdf`  
  - `figures/adaptive_threshold_norm.pdf`

- **`landauer_extension.py`**  
  Computes energy-entropy trade-offs based on Landauer’s principle generalized to subjective cognitive transitions.  
  Outputs:  
  - `figures/deltaS_vs_epsilon.pdf`  
  - `figures/energy_vs_epsilon.pdf`

- **`compare_entropy_fixed_adaptive.py`**  
  Compares entropy dynamics between fixed and adaptive regimes.  
  Outputs:  
  - `figures/entropy_comparison.pdf`  
  - `figures/entropy_over_time.pdf`  
  - `figures/epsilon_over_time.pdf`

- **`von_neumann_entropy.py`**  
  Calculates von Neumann entropy and trace distance in simulated observer states.  
  Outputs:  
  - `figures/entropy_vs_epsilon.pdf`  
  - `figures/trace_distance_vs_epsilon.pdf`

- **`simulate_entropy_rt_full.py`**  
  Generates histograms and cumulative distributions of reaction times (RT) and entropy over trials.  
  Outputs:  
  - `figures/histogram_rt.pdf`  
  - `figures/scatter_entropy_rt.pdf`  
  - `figures/cdf_rt.pdf`  
  - `figures/entropy_hist_comparison.pdf`  
  - `figures/rt_cdf_comparison.pdf`  
  - `figures/entropy_boxplot.pdf`  
  - `figures/rt_boxplot.pdf`

- **`compare_dirichlet_params.py`**  
  Compares simulation outcomes across Dirichlet prior configurations.  
  Outputs:  
  - `data/simulated_data_dirichlet_*.csv`  
  - `data/simulated_data_all_configs.csv`

- **`cognitive_geodesic_simulation.py`**  
  Numerically solves geodesic paths in perceptual metric space.  
  Output: `figures/cognitive_geodesic.pdf`

- **`cognitive_geodesic_trajectories.py`**  
  Visualizes cognitive trajectories in observer belief space.  
  Output: `figures/cognitive_trajectories.pdf`

- **`geodesic_dynamics_cognitive_action.py`**  
  Computes cognitive action and its temporal evolution along geodesic paths.  
  Output: `figures/cognitive_action_vs_time.pdf`

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

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (11.0)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.17234483]

---

## License

MIT License. See `LICENSE` file for details.

## Version History and DOI Links

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
- [Version 8 only](https://doi.org/10.5281/zenodo.16728290)
- [Version 9 only](https://doi.org/10.5281/zenodo.16741400)
- [Version 10 only](https://doi.org/10.5281/zenodo.16888675)
- [Version 11 only (current)](https://doi.org/10.5281/zenodo.17234483)

**View this version on Zenodo:**  
[https://doi.org/10.5281/zenodo.17234483](https://doi.org/10.5281/zenodo.17234483)

## Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{khomyakov_vladimir_2025_subjective_physics_simulation,
  author    = {Vladimir Khomyakov},
  title     = {Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics},
  version   = {11.0},
  year      = {2025},
  doi       = {10.5281/zenodo.17234483},
  url       = {https://github.com/Khomyakov-Vladimir/subjective-physics-simulation}
}
```
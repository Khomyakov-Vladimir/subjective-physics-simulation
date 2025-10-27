# Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (v12.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17407408.svg)](https://doi.org/10.5281/zenodo.17407408)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This page presents the core article of _Cognitive Projection and Observer Entropy_ (Vladimir Khomyakov, 2025), a formal minimal framework for Subjective Physics defining observer entropy Sobs, projection operator F, and perceptual threshold ε.

**Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics** (Khomyakov, 2025) presents a _formal minimal computational framework_ for incorporating the observer into physical reality.  
The model defines _observer entropy Sobs_ as the information loss produced by a finite perceptual threshold (ε), implemented via a _projection operator F_ acting on a stochastic process.  
The framework analyses how entropy dynamics S(ε) and distinguishability D(ε) scale with the observer’s resolution, offering an information-theoretic foundation for _Subjective Physics_ and for reproducible simulations of cognitive projection phenomena.

Research Overview
-----------------

**Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics** is a 2025 paper that develops a formal model of subjective physics using principles of information theory.  
The work does not propose a new external physical theory; rather, it formalizes how an observer constructs reality under perceptual and informational constraints.

The model introduces a cognitive projection operator, which acts on stochastic processes by grouping states within a perceptual threshold.  
This operation reduces the number of distinguishable classes and thereby defines the cognitive entropy, quantifying how much structure is preserved or lost as perception becomes coarser.

### Key Research Concepts

**Subjective Physics:** A framework that places the observer at the center, treating perception and cognitive limitations as constitutive of physical experience.

**Cognitive Projection:** The coarse-graining of a stochastic process by an observer, determined by an indistinguishability threshold.

**Cognitive Entropy:** A measure of the number of perceptual classes distinguishable at resolution; increasing reduces resolution and thus collapses detail.

**Information-Theoretic Foundation:** The model adapts Landauer's principle of information cost to the cognitive domain, interpreting indistinguishability as informational reduction rather than thermodynamic generation of entropy.

**Minimal Model:** A simplified simulation framework illustrating how subjective constraints structure perceived reality.

### Methodology and Extensions

**Projection Operator:** Formally defined to act on multidimensional stochastic fields, modeling how perceptual resolution limits the observer's access to fine-grained states.

**Entropy and Trace Distance:** Numerical experiments explore how cognitive entropy and state distinguishability scale with perceptual thresholds.

**Adaptive Thresholds:** Feedback mechanisms are introduced whereby the perceptual threshold dynamically adapts as a function of either the stabilization of cognitive entropy or the norms of cognitive projections.

**Multi-Agent Σ-Projection:** Version v11.2 integrates the multi-agent framework introduced earlier and includes metrics such as the Shared Reality Index (SRI),  
a measure of intersubjective alignment based on the overlap of projected state spaces, to quantify convergence toward consensus.

### Scientific Perspective

The work emphasizes a conceptual shift: the observer is not simply embedded in the world; rather, the world is cognitively constructed by the observer.  
The framework serves as a computational and information-theoretic formalization of this principle, providing a minimal but extensible model for exploring subjective physics.

**Version 12.0 publication:** October 21, 2025  
**Author:** Vladimir Khomyakov  
**DOI (v12.0):** [10.5281/zenodo.17407408](https://doi.org/10.5281/zenodo.17407408)

Abstract
--------

Version 12.0 unifies the Subjective Physics framework into a reproducible, mathematically rigorous model that integrates theoretical, computational, and experimental developments from all prior stages (v1–v11).  
Core components include:

*   Definition of **observer entropy** Sobs(ε) as information loss under finite perceptual resolution, implemented by a projection operator Fε acting on stochastic processes,
*   Formal derivation of the **Cognitive Uncertainty Principle** (β–KL trade-off) linking perceptual precision and statistical simplicity via the Cramér–Rao bound derived from Fisher information,
*   **Dynamic observer-entropy law** dSobs/dt = f(I, M, C) connecting entropy, information inflow, and cognitive modulation,
*   **Adaptive spectral thresholds** enabling entropy-regulated perception and stabilization of cognitive dynamics,
*   **Σ-projection and cognitive decoherence** as mechanisms of shared reality and information-consistent state selection,
*   **Thermodynamic trade-off** and Landauer-based energy bounds linking perceptual discrimination to physical dissipation,
*   Complete **Python-based reproducibility** of all figures and simulations establishing a computational foundation for Subjective Physics.

The model consolidates observer entropy dynamics, information geometry, and cognitive thermodynamics into a minimal and verifiable framework of observer–world interaction.

Extended Notes (retrodiction, weak values, EEG-inspired portraits, cultural perspectives) are archived separately.

Simulation Package (v11.2, archival)
------------------------------------

**Core reproducible scripts:**

*   `simulate_entropy_dynamics.py` — time evolution of observer entropy under overload, equilibrium, and collapse regimes.
*   `generate_entropy_phase_map.py` — phase diagram in (α, β) parameter space classifying perceptual regimes.
*   `simulate_multiagent_synchronization.py` — synchronization vs. divergence in coupled observer systems.

Figures and Outputs
-------------------

*   `entropy_dynamics.pdf` — Time trajectories of observer entropy in three dynamical regimes.
*   `entropy_phase_map.pdf` — Phase diagram showing collapse, equilibrium, and overload regions.
*   `multiagent_synchronization.pdf` — Convergent (solid) and divergent (dashed) multi-agent entropy trajectories.

Execution
---------

\# Reproduce all core figures of v11.2:  
python simulate\_entropy\_dynamics.py  
python generate\_entropy\_phase\_map.py  
python simulate\_multiagent\_synchronization.py

Citation
--------

Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (12.0). Zenodo. https://doi.org/10.5281/zenodo.17407408

Supplementary Materials
-----------------------

The core model is accompanied by two official supplementary documents, archived under the same DOI:

*   [**A Technical Summary of Subjective Physics (v12.0)**](https://doi.org/10.5281/zenodo.17407408) — a concise overview of the framework, key equations, and simulation protocols ([PDF](https://zenodo.org/records/17407408/files/a-technical-summary-of-subjective-physics-v12.0-2025.pdf) Zenodo) ([PDF](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v12.0/a-technical-summary-of-subjective-physics-v12.0-2025.pdf) GitHub) ([PDF](https://digitalphysics.ru/articles/khomyakov_v/a-technical-summary-of-subjective-physics-v12.0-2025.pdf) digitalphysics.ru).
*   [**Extended Notes (v11.2)**](https://doi.org/10.5281/zenodo.17407408) — supplementary discussions on retrodiction, weak values, EEG-inspired portraits, and cultural perspectives ([PDF](https://zenodo.org/records/17407408/files/subjective_physics_simulation_v11.2_extended_notes.pdf) Zenodo) ([PDF](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v12.0/subjective_physics_simulation_v11.2_extended_notes.pdf) GitHub) ([PDF](https://digitalphysics.ru/articles/khomyakov_v/subjective_physics_simulation_v11.2_extended_notes.pdf) digitalphysics.ru).

Canonical vs Extended Versions
------------------------------

For readers interested in a minimal reproducible model, we recommend **version 5.0** as the _baseline_ (DOI: [10.5281/zenodo.15867963](https://doi.org/10.5281/zenodo.15867963)).  
For readers seeking the full consolidated release, please see **version 12.0** as the _advanced_ development (DOI: [10.5281/zenodo.17407408](https://doi.org/10.5281/zenodo.17407408)).

Version History
---------------

*   [v1-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v1.0/subjective_physics_simulation_v1.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (1.0). Zenodo. https://doi.org/10.5281/zenodo.15719390"): Initial definition of projection operator — DOI [10.5281/zenodo.15719390](https://doi.org/10.5281/zenodo.15719390 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (1.0). Zenodo. https://doi.org/10.5281/zenodo.15719390")
*   [v2-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v2.0/subjective_physics_simulation_v2.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (2.0). Zenodo. https://doi.org/10.5281/zenodo.15751229"): Entropy scaling analysis — DOI [10.5281/zenodo.15751229](https://doi.org/10.5281/zenodo.15751229 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (2.0). Zenodo. https://doi.org/10.5281/zenodo.15751229")
*   [v3-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v3.0/subjective_physics_simulation_v3.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (3.0). Zenodo. https://doi.org/10.5281/zenodo.15780239"): Energetic cost (Landauer model) — DOI [10.5281/zenodo.15780239](https://doi.org/10.5281/zenodo.15780239 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (3.0). Zenodo. https://doi.org/10.5281/zenodo.15780239")
*   [v4-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v4.0/subjective_physics_simulation_v4.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (4.0). Zenodo. https://doi.org/10.5281/zenodo.15813188"): Adaptive suppression and phase transitions — DOI [10.5281/zenodo.15813188](https://doi.org/10.5281/zenodo.15813188 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (4.0). Zenodo. https://doi.org/10.5281/zenodo.15813188")
*   [v5-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v5.0/subjective_physics_simulation_v5.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (5.0). Zenodo. https://doi.org/10.5281/zenodo.15867963"): Reaction times and Dirichlet priors — DOI [10.5281/zenodo.15867963](https://doi.org/10.5281/zenodo.15867963 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (5.0). Zenodo. https://doi.org/10.5281/zenodo.15867963")
*   [v6-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v6.0/subjective_physics_simulation_v6.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (6.0). Zenodo. https://doi.org/10.5281/zenodo.16028303"): Geodesic observer dynamics — DOI [10.5281/zenodo.16028303](https://doi.org/10.5281/zenodo.16028303 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (6.0). Zenodo. https://doi.org/10.5281/zenodo.16028303")
*   [v7-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v7.0/subjective_physics_simulation_v7.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (7.0). Zenodo. https://doi.org/10.5281/zenodo.16368499"): Cognitive retrodiction — DOI [10.5281/zenodo.16368499](https://doi.org/10.5281/zenodo.16368499 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (7.0). Zenodo. https://doi.org/10.5281/zenodo.16368499")
*   [v7.4-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v7.4/subjective_physics_simulation_v7.4.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (7.4). Zenodo. https://doi.org/10.5281/zenodo.16478500"): Noise-driven retrodiction — DOI [10.5281/zenodo.16478500](https://doi.org/10.5281/zenodo.16478500 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (7.4). Zenodo. https://doi.org/10.5281/zenodo.16478500")
*   [v8.0-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v8.0/subjective_physics_simulation_v8.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (8.0). Zenodo. https://doi.org/10.5281/zenodo.16728290"): Σ-projection, entropy feedback, bifurcations — DOI [10.5281/zenodo.16728290](https://doi.org/10.5281/zenodo.16728290 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (8.0). Zenodo. https://doi.org/10.5281/zenodo.16728290")
*   [v9.0-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v9.0/subjective_physics_simulation_v9.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (9.0). Zenodo. https://doi.org/10.5281/zenodo.16741400"): Stochastic phase portrait, EEG integration — DOI [10.5281/zenodo.16741400](https://doi.org/10.5281/zenodo.16741400 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (9.0). Zenodo. https://doi.org/10.5281/zenodo.16741400")
*   [v10.0-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v10.0/subjective_physics_simulation_v10.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (10.0). Zenodo. https://doi.org/10.5281/zenodo.16888675"): Generalized multi-agent Σ-projection, Shared Reality Index — DOI [10.5281/zenodo.16888675](https://doi.org/10.5281/zenodo.16888675 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (10.0). Zenodo. https://doi.org/10.5281/zenodo.16888675")
*   [v11.2-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v11.2.4/subjective_physics_simulation_v11.2_main_article.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (11.2.4 (technical core edition)). Zenodo. https://doi.org/10.5281/zenodo.17312704"): Consolidated minimal formalism, evolutionary law for observer entropy, and Extended Notes (archived). — DOI [10.5281/zenodo.17312704](https://doi.org/10.5281/zenodo.17312704 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (11.2.4 (technical core edition)). Zenodo. https://doi.org/10.5281/zenodo.17312704")
*   [v12.0-PDF-GitHub](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation/releases/download/v12.0/subjective_physics_simulation_v12.0_main_article.pdf "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (12.0). Zenodo. https://doi.org/10.5281/zenodo.17407408"): Unified theoretical release — formal Cognitive Uncertainty Principle, observer-entropy dynamics. — DOI [10.5281/zenodo.17407408](https://doi.org/10.5281/zenodo.17407408 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (12.0). Zenodo. https://doi.org/10.5281/zenodo.17407408")

**Permanent DOI (latest version):** [10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389 "Khomyakov, V. (2025). Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics. Zenodo. https://doi.org/10.5281/zenodo.15719389")

**This page is continuously updated with new versions.**

**[Official Abstract](https://digitalphysics.ru/cognitive-projection-abstract.html)**

Reproducibility
---------------

The complete simulation code, datasets, and plotting scripts are publicly available at  
[https://github.com/Khomyakov-Vladimir/subjective-physics-simulation](https://github.com/Khomyakov-Vladimir/subjective-physics-simulation)

Repository Contents:

*   Python scripts for entropy generation and reaction time simulation.
*   Bootstrap confidence interval computation tools.
*   Example figures in PDF format.
*   Instructions for reproducing all results.

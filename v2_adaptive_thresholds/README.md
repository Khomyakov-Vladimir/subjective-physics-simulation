# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the simulation code and figures for the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/
├── main.py # Run the full simulation (standard + adaptive)
├── observer.py # Observer with multiple adaptation modes
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
│ ├── adaptive_threshold_entropy.pdf
│ ├── adaptive_threshold_norm.pdf
│ ├── adaptive_trace_distance_entropy.pdf
│ ├── adaptive_trace_distance_norm.pdf
├── data/
│ ├── adaptive_metrics_entropy.npz
│ └── adaptive_metrics_norm.npz
```

## Reproducibility

To reproduce the figures used in the article, run:

## Adaptive Mode

To run the dynamic adaptation of perception threshold ε(t), execute:

```bash
python main.py
```

The script generates separate plots for:

Threshold evolution

Entropy dynamics

Norm and trace-distance scaling

The default adaptation uses entropy feedback. To test norm-based adaptation, modify main.py (see comments).

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
```
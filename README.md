# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the simulation code and figures for the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/
├── main.py
├── observer.py
├── README.md
├── LICENSE
├── .zenodo.json
├── .github/
│   └── workflows/
│       └── run-simulation.yml
├── figures/
│   ├── entropy_vs_epsilon.pdf
│   ├── norm_vs_time.pdf
│   └── trace_distance_vs_epsilon.pdf
```

## Reproducibility

To reproduce the figures used in the article, run:

```bash
python main.py
```

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
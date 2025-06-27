# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the simulation code and figures for the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

## Directory Structure

```
subjective-physics-simulation/
├── main.py                             # Run the full simulation and generate plots (PDF)
├── observer.py                         # Simulation logic for cognitive projection observer
├── README.md                           # This file
├── LICENSE                             # MIT License
├── CITATION.cff                        # Citation metadata
├── .zenodo.json
├── figures/                            # Output directory (entropy curves, plots, JSON)
│   ├── entropy_vs_epsilon.pdf          # Plot 1: Entropy vs epsilon
│   ├── norm_vs_time.pdf                # Plot 2: Norm vs Time
│   └── trace_distance_vs_epsilon.pdf   # Plot 3: Trace distance vs epsilon
```

## Reproducibility

To reproduce the figures used in the article, run:

```bash
python main.py
```

---

## ⚠️ Version Notice

This implementation corresponds to version 1 (v1) of the model.  
It provides the first reproducible simulation of entropy generation from coarse-grained cognitive projection.  
However, certain assumptions (projection formalism, dimensionality) were clarified in [version 2](../v2_adaptive_thresholds/README.md).  
For current results, please refer to v2.

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
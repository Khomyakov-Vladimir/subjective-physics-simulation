# Subjective Physics Simulation: Cognitive Projection and Entropy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719389.svg)](https://doi.org/10.5281/zenodo.15719389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains simulation code and data accompanying the article:

**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**  
by Vladimir Khomyakov (Zenodo DOI: [10.5281/zenodo.15719389](https://doi.org/10.5281/zenodo.15719389))

---

## 🔁 Version Structure

| Version | Directory                     | Description |
|---------|-------------------------------|-------------|
| v1      | `v1_entropy_hierarchy/`       | Static threshold model, hierarchical observer chain |
| v2      | `v2_adaptive_thresholds/`     | Adaptive threshold ε(t), feedback by entropy or norm |

Each subdirectory contains its own code, figures, data, and citation metadata.  

You can reproduce all results by navigating into the respective folder and running:

```bash
python main.py
```

---

## ✅ Statement of Correction

The current version (v2) refines and extends the initial formulation published as v1.  
While the first version captured the core idea of observer-induced entropy via cognitive thresholds,  
certain assumptions — such as the definition of the projection operator $F_\epsilon$ and the dimensional structure of the process —  
were clarified and formalized in v2.

Version v2 supersedes v1 as the authoritative reference.  
The earlier conceptual series *“Cognitive Simulator and Subjective Physics” [v1–v7]* is now deprecated and should be treated as a heuristic precursor.

---

For DOI versioning and archival history, see:

- [Latest version](https://doi.org/10.5281/zenodo.15719389)
- [Version 2 only](https://doi.org/10.5281/zenodo.15751229)

---

## 📜 License

MIT License (see individual LICENSE files per version).

## 📖 Citation

Use the corresponding BibTeX entry from each version's `README.md` or `CITATION.cff`.

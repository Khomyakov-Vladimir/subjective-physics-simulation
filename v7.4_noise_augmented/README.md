# Subjective Physics Simulation: Cognitive Projection and Entropy (v7.4)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16478500.svg)](https://doi.org/10.5281/zenodo.16478500)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This subdirectory contains additional simulation code for:

**Version 7.4** of the article  
**"Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics"**

This version introduces **noise-augmented simulations** of cognitive reconstruction under uncertainty in final observations.

---

## 🆕 Additions in Version 7.4

### Scripts

- `noise_dynamics_simulation.py`  
  Generates multiple cognitive trajectories under **perturbed final conditions** \( B' = B + \delta \).  
  📈 Output: `figures/noise_dynamics.pdf`

- `retrodiction_noise_variation.py`  
  Simulates retrodicted trajectories from multiple **noisy final observations**.  
  📊 Output: `figures/cog_reconstruction_noise.pdf`

### Figures

- `figures/noise_dynamics.pdf`  
  Cognitive trajectories from various noisy boundary conditions.

- `figures/cog_reconstruction_noise.pdf`  
  Entropy-reducing reconstructions under variable final beliefs.

---

## 💻 How to Run

Install required dependencies:

```bash
pip install numpy matplotlib scipy
```

Then run:

```bash
python noise_dynamics_simulation.py
python retrodiction_noise_variation.py
```

The generated PDF figures will be saved to the `../figures/` directory.

---

## License

MIT License. See `LICENSE` for full terms.

## Citation

If using this module in your research, please cite:

Khomyakov, V. (2025). *Cognitive Projection and Observer Entropy: A Minimal Model of Subjective Physics (v7.4)*. Zenodo. DOI: [https://doi.org/10.5281/zenodo.16478500](https://doi.org/10.5281/zenodo.16478500)

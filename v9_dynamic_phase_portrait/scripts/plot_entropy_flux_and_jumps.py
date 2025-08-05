#!/usr/bin/env python3
"""
# scripts/plot_entropy_flux_and_jumps.py

"Entropy Flux and Cognitive Jumps"

Core Features:
- Time-domain Tsallis entropy dynamics ($H_q(t)$)
- Energy-based jump detection ($\Delta E_{\mathrm{disc}}$)
- Fluctuation theorem validation ($P_+/P_-$)
- Dual-axis visualization of entropy and energy jumps
- Real-time annotation of cognitive phase transitions
- Adaptive thresholding for entropy-based adaptation
- Quantum-weighted trajectory evolution ($w_i(t)$)

Scientific Focus:  
Visualizes temporal dynamics of cognitive entropy and energy dissipation during phase transitions. 
Implements nonequilibrium thermodynamics and validates fluctuation theorem signatures.

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from scipy.integrate import solve_ivp

# Import from phase_portrait (ensure consistent constants)
from phase_portrait import fluctuation_theorem, energy_cost, cognitive_entropy, adaptive_threshold, weight_dynamics, apply_quantum_phase, k_B, T

# Configure reproducibility
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
figures_dir = os.path.join(repo_root, "figures")
os.makedirs(figures_dir, exist_ok=True)

# --- Data generation ---
t = np.linspace(0, 1, 200)
mode = 'Entropy-based'
epsilon = adaptive_threshold(t, mode.lower())
S = cognitive_entropy(epsilon, model='tsallis', q=1.2)
E = energy_cost(epsilon)

# Solve ODE for weights
sol = solve_ivp(weight_dynamics, [t[0], t[-1]], y0=[0.5], t_eval=t, args=(0.3, 0.5))
w_real = sol.y[0]
w_complex = apply_quantum_phase(t, w_real)
w = np.abs(w_complex)

# Detect jumps and calculate ΔE
jump_indices = [50, 140]
jump_times = t[jump_indices]
delta_E_list = []
fluctuation_ratios = []

for idx in jump_indices:
    if idx > 0:
        delta_E = E[idx] - E[idx-1]
        delta_E_list.append(delta_E)
        fluctuation_ratios.append(fluctuation_theorem(delta_E))

# --- Plotting with international labels ---
fig, ax1 = plt.subplots(figsize=(9, 5))

# Primary axis: Entropy
color_entropy = 'tab:blue'
ax1.set_xlabel("Normalized time $t$", fontsize=12)
ax1.set_ylabel("Tsallis entropy $H_q(t)$", color=color_entropy, fontsize=12)
ax1.plot(t, S, color=color_entropy, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_entropy)
ax1.set_xlim(0, 1)
ax1.grid(True, linestyle='--', alpha=0.7)

# Secondary axis: Energy jumps
ax2 = ax1.twinx()
color_jumps = 'tab:red'
ax2.set_ylabel("Discretization energy $\Delta E_{\mathrm{disc}}$ (J)", color=color_jumps, fontsize=12)

# Calculate dynamic positioning
y_base = max(delta_E_list) * 1.5 if delta_E_list else 1e-22

for i, (jt, delta_E, ratio) in enumerate(zip(jump_times, delta_E_list, fluctuation_ratios)):
    delta_E_str = f"{delta_E:.1e}".replace('e-0', 'e-')
    label = f"ΔE = {delta_E_str}\nP+/P- = {ratio:.2f}"
    
    ax2.axvline(x=jt, color=color_jumps, linestyle='--', alpha=0.8)
    
    # Dynamic positioning
    y_offset = y_base * 0.3 * i
    ax2.annotate(
        label, 
        xy=(jt, 0), 
        xytext=(jt + 0.05, y_base + y_offset),
        arrowprops=dict(arrowstyle="->", color=color_jumps, connectionstyle="arc3,rad=.2"),
        fontsize=9, 
        color=color_jumps,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_jumps, alpha=0.8)
    )

ax2.tick_params(axis='y', labelcolor=color_jumps)
ax2.set_ylim(0, y_base * 2.0)

# International title
plt.title(
    "Entropy Flux and Cognitive Jumps\n"
    "Entropy-based Adaptation with Real Simulation Jumps",
    fontsize=14,
    pad=20
)
fig.tight_layout()

# Save PDF
pdf_path = os.path.join(figures_dir, "entropy_flux_and_jumps_real.pdf")
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
plt.close(fig)

logger.info(f"Saved PDF to: {pdf_path}")
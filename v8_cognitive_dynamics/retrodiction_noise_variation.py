#!/usr/bin/env python3
"""
# scripts/retrodiction_noise_variation.py

Noise-Augmented Retrodiction Trajectories

Generates and visualizes multiple reconstructed trajectories under
different noisy observations of final state B_obs = B + δ.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_bvp
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
A_TRUE = 2.5
B_DESIRED = 5.0
K = 1.2
GAMMA = 0.6
T_MIN, T_MAX = 0.0, 1.0
NOISE_STD = 0.2
N_SAMPLES = 15

# ODE system
def cognitive_ode(t, y, k, gamma, B):
    dydt = y[1]
    d2ydt2 = -gamma * y[1] - 2 * k * (y[0] - B)
    return [dydt, d2ydt2]

# Boundary conditions
def bc(ya, yb, B_target):
    return np.array([ya[1], yb[0] - B_target])

# Solve BVP
def solve_bvp_trajectory(B_target, k, gamma, A_guess=None):
    t = np.linspace(T_MIN, T_MAX, 100)
    if A_guess is None:
        A_guess = B_target / 2
    y_guess = np.zeros((2, t.size))
    y_guess[0] = A_guess + (B_target - A_guess) * t ** 2
    y_guess[1] = 2 * (B_target - A_guess) * t
    sol = solve_bvp(
        lambda t, y: cognitive_ode(t, y, k, gamma, B_target),
        lambda ya, yb: bc(ya, yb, B_target),
        t, y_guess
    )
    return sol if sol.success else None

# Save figure
def save_figure(fig, filename, output_dir="figures"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    figures_dir = os.path.join(repo_root, output_dir)
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    logger.info(f"Saved PDF to: {path}")
    plt.close(fig)

# Main workflow
def run_simulation(show_plot=True):
    np.random.seed(42)
    t_plot = np.linspace(T_MIN, T_MAX, 100)
    noisy_Bs = B_DESIRED + np.random.normal(0, NOISE_STD, N_SAMPLES)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, B_obs in enumerate(noisy_Bs):
        A_guess = B_obs / K
        sol = solve_bvp_trajectory(B_obs, K, GAMMA, A_guess)
        if sol:
            ax.plot(t_plot, sol.sol(t_plot)[0], lw=2, alpha=0.5)

    ax.axhline(B_DESIRED, color='gray', linestyle='--', label='B (Target)')
    ax.set_title("Retrodicted Trajectories Under Noisy Final Observations")
    ax.set_xlabel("Normalized Time $t$")
    ax.set_ylabel("Cognitive State $y(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, "cog_reconstruction_noise.pdf")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    run_simulation(show_plot=True)

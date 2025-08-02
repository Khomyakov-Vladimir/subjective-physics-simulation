#!/usr/bin/env python3
"""
# scripts/noise_dynamics_simulation.py

Noise-Augmented Retrodiction Simulation

Generates cognitive trajectories under noisy final conditions B' = B + δ.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_bvp

# Constants
B_ORIGINAL = 5.0
K = 1.2
GAMMA = 0.6
T_MIN, T_MAX = 0, 1
N_SAMPLES = 15
NOISE_STD = 0.3

def cognitive_ode(t, y, k, gamma, B):
    dydt = y[1]
    d2ydt2 = -gamma * y[1] - 2*k*(y[0] - B)
    return [dydt, d2ydt2]

def bc(ya, yb, B):
    return np.array([ya[1], yb[0] - B])

def solve_trajectory(B_target, A_guess):
    t = np.linspace(T_MIN, T_MAX, 100)
    y_guess = np.zeros((2, t.size))
    y_guess[0] = A_guess + (B_target - A_guess) * t**2
    y_guess[1] = 2 * (B_target - A_guess) * t
    sol = solve_bvp(
        lambda t, y: cognitive_ode(t, y, K, GAMMA, B_target),
        lambda ya, yb: bc(ya, yb, B_target),
        t, y_guess
    )
    return sol if sol.success else None

def main():
    t_plot = np.linspace(T_MIN, T_MAX, 100)
    np.random.seed(42)
    B_values = B_ORIGINAL + np.random.normal(0, NOISE_STD, N_SAMPLES)

    fig, ax = plt.subplots(figsize=(8, 6))
    for B_i in B_values:
        A_guess = B_i / K
        sol = solve_trajectory(B_i, A_guess)
        if sol:
            ax.plot(t_plot, sol.sol(t_plot)[0], alpha=0.5, lw=2)

    ax.set_title("Cognitive Trajectories Under Noisy Final State $B + \\delta$")
    ax.set_xlabel("Normalized Time $t$")
    ax.set_ylabel("Cognitive State $y(t)$")
    ax.grid(True, alpha=0.3)

    # Save to PDF
    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "noise_dynamics.pdf")
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
# scripts/cognitive_retrodiction_simulation.py

Cognitive Retrodiction Simulation

Simulates cognitive retrodiction using differential equations model 
with boundary value problems for state trajectories.

Author: Vladimir Khomyakov
Date: July 2025
Version: 2.4
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.gridspec as gridspec
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
A_TRUE = 2.5        # True initial state
B_DESIRED = 5.0     # Desired target state (potential center)
K = 1.2             # Stiffness coefficient
GAMMA = 0.6         # Damping coefficient
NOISE_STD = 0.1     # Observation noise standard deviation
T_MIN, T_MAX = 0, 1 # Time boundaries

# --- Differential Equation System ---
def cognitive_ode(t, y, k, gamma, B):
    """Cognitive geodesic equation (2nd order ODE system)"""
    dydt = y[1]
    d2ydt2 = -gamma * y[1] - 2*k*(y[0] - B)
    return [dydt, d2ydt2]

# --- Boundary Conditions ---
def bc(ya, yb, B_target):
    """Boundary conditions for BVP"""
    return np.array([
        ya[1],           # dy/dt(0) = 0 (zero initial velocity)
        yb[0] - B_target  # y(1) = B_target (observed final state)
    ])

# --- Cognitive Intervention ---
def cognitive_intervention(B_original):
    """Apply intervention to the target state"""
    return B_original * 1.25  # 25% increase

# --- Solve Boundary Value Problem ---
def solve_cognitive_bvp(B_target, k, gamma, A_guess=None):
    """Solve boundary value problem for cognitive trajectory"""
    t = np.linspace(T_MIN, T_MAX, 100)
    
    # Improved initial guess
    if A_guess is None:
        A_guess = B_target / 2  # Default guess if not provided
    
    # Quadratic function satisfying boundary conditions:
    # y(0) = A_guess, dy/dt(0) = 0, y(1) = B_target
    y_guess = np.zeros((2, t.size))
    y_guess[0] = A_guess + (B_target - A_guess) * t**2
    y_guess[1] = 2 * (B_target - A_guess) * t
    
    # Solve BVP
    solution = solve_bvp(
        lambda t, y: cognitive_ode(t, y, k, gamma, B_target),
        lambda ya, yb: bc(ya, yb, B_target),
        t, y_guess
    )
    
    if not solution.success:
        logger.error(f"BVP solution failed for B={B_target:.2f}")
        logger.error(f"Error message: {solution.message}")
        logger.error(f"Status: {solution.status}")
        return None
    
    logger.info(f"BVP solved successfully for B={B_target:.2f}")
    return solution

# --- Generate Observation with Noise ---
def generate_observation(A_true, k, gamma, B_desired):
    """Generate noisy observation of final state"""
    # Solve IVP to get true final state
    sol_ivp = solve_ivp(
        lambda t, y: cognitive_ode(t, y, k, gamma, B_desired),
        [T_MIN, T_MAX], 
        [A_true, 0.0],  # Initial state and velocity
        t_eval=[T_MAX]
    )
    if not sol_ivp.success:
        logger.error("IVP solution failed for observation generation")
        return None
    
    B_true = sol_ivp.y[0, -1]
    B_obs = B_true + np.random.normal(0, NOISE_STD)
    logger.info(f"Generated observation: B_true={B_true:.4f}, B_obs={B_obs:.4f}")
    return B_obs

# --- Visualization Utilities ---
def save_figure(fig: plt.Figure, filename: str, output_dir: str = "figures"):
    """Save figure to PDF in repository structure"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    figures_path = os.path.join(repo_root, output_dir)
    os.makedirs(figures_path, exist_ok=True)
    pdf_path = os.path.join(figures_path, filename)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    logger.info(f"Saved PDF to: {pdf_path}")
    plt.close(fig)

# --- Run Full Simulation ---
def run_simulation(show_plot: bool = True):
    """Main simulation workflow"""
    # Generate noisy observation of final state
    B_obs = generate_observation(A_TRUE, K, GAMMA, B_DESIRED)
    if B_obs is None:
        logger.error("Failed to generate observation. Exiting.")
        return
    
    logger.info(f"True initial state: A = {A_TRUE:.4f}")
    logger.info(f"Observed final state: B_obs = {B_obs:.4f}")

    # --- Solve without intervention ---
    # Use linear model for initial guess: A ≈ B/k
    A_initial_guess = B_obs / K
    sol_no_intv = solve_cognitive_bvp(B_obs, K, GAMMA, A_guess=A_initial_guess)
    if sol_no_intv is None:
        logger.error("Failed to solve BVP without intervention")
        return
    
    A_hat = sol_no_intv.sol(0)[0]
    logger.info(f"Retrodicted state (no intv): A = {A_hat:.4f}")

    # --- Apply cognitive intervention ---
    B_intv = cognitive_intervention(B_obs)
    logger.info(f"Intervened target state: B_intv = {B_intv:.4f}")
    sol_intv = solve_cognitive_bvp(B_intv, K, GAMMA, A_guess=A_hat)
    if sol_intv is None:
        logger.error("Failed to solve BVP with intervention")
        return
    
    A_prime = sol_intv.sol(0)[0]
    logger.info(f"Retrodicted state (with intv): A' = {A_prime:.4f}")
    logger.info(f"Intervention effect: ΔA = {A_prime - A_hat:.4f}")

    # --- Time values for plotting ---
    t_plot = np.linspace(T_MIN, T_MAX, 100)
    
    # =================================================================
    # Visualization 1: State and Velocity Trajectories (state_trajectories.pdf)
    # =================================================================
    fig1 = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
    
    # State trajectory plot
    ax1 = plt.subplot(gs[0])
    ax1.plot(t_plot, sol_no_intv.sol(t_plot)[0], 'b-', lw=2.5, label='No Intervention')
    ax1.plot(t_plot, sol_intv.sol(t_plot)[0], 'r--', lw=2.5, label='With Intervention')
    ax1.axhline(B_DESIRED, color='gray', linestyle=':', alpha=0.7, label='Desired State (B)')
    ax1.set_xlabel('Normalized Time (t)')
    ax1.set_ylabel('Cognitive State (y)')
    ax1.set_title('Cognitive State Trajectories')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Velocity plot
    ax2 = plt.subplot(gs[1])
    ax2.plot(t_plot, sol_no_intv.sol(t_plot)[1], 'b-', lw=2.5, label='No Intervention')
    ax2.plot(t_plot, sol_intv.sol(t_plot)[1], 'r--', lw=2.5, label='With Intervention')
    ax2.set_xlabel('Normalized Time (t)')
    ax2.set_ylabel('Cognitive Velocity ($\dot{y}$)')
    ax2.set_title('Cognitive Velocity Dynamics')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig1, "state_trajectories.pdf")
    if show_plot: 
        plt.show()

    # =================================================================
    # Visualization 2: Phase Portrait (phase_portrait.pdf)
    # =================================================================
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(sol_no_intv.sol(t_plot)[0], sol_no_intv.sol(t_plot)[1], 
             'b-', lw=2.5, label='No Intervention')
    ax3.plot(sol_intv.sol(t_plot)[0], sol_intv.sol(t_plot)[1], 
             'r--', lw=2.5, label='With Intervention')
    ax3.set_xlabel('Cognitive State (y)')
    ax3.set_ylabel('Cognitive Velocity ($\dot{y}$)')
    ax3.set_title('Phase Portrait of Cognitive Dynamics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    save_figure(fig2, "phase_portrait.pdf")
    if show_plot: 
        plt.show()

    # =================================================================
    # Visualization 3: Initial State Comparison (retrodicted_states.pdf)
    # =================================================================
    fig3, ax4 = plt.subplots(figsize=(7, 5))
    states = [A_TRUE, A_hat, A_prime]
    labels = ['True A', 'A (No Intv.)', "A' (With Intv.)"]
    colors = ['gray', 'blue', 'red']
    
    # Create bar plot with values on top
    bars = ax4.bar(labels, states, color=colors)
    ax4.set_ylabel('Value of Initial State (A)')
    ax4.set_title('Comparison of Retrodicted Initial States')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    save_figure(fig3, "retrodicted_states.pdf")
    if show_plot: 
        plt.show()

    # =================================================================
    # Visualization 4: Potential Landscape (NEW)
    # =================================================================
    fig4, ax5 = plt.subplots(figsize=(8, 6))
    y_vals = np.linspace(min(A_TRUE, A_hat, A_prime)-1, max(B_DESIRED, B_obs, B_intv)+1, 200)
    
    # Potential function: V(y) = (y - B)^2
    V_no_intv = (y_vals - B_DESIRED)**2
    V_intv = (y_vals - B_intv)**2
    
    ax5.plot(y_vals, V_no_intv, 'b-', label='Potential (No Intv)')
    ax5.plot(y_vals, V_intv, 'r--', label='Potential (With Intv)')
    
    # Mark important points
    ax5.axvline(B_DESIRED, color='blue', linestyle=':', alpha=0.5)
    ax5.axvline(B_intv, color='red', linestyle=':', alpha=0.5)
    ax5.scatter([A_hat, A_prime], [(A_hat - B_DESIRED)**2, (A_prime - B_intv)**2], 
               s=100, c=['blue', 'red'], edgecolors='black', zorder=5)
    
    ax5.set_xlabel('Cognitive State (y)')
    ax5.set_ylabel('Potential V(y; B)')
    ax5.set_title('Cognitive Potential Landscape')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    save_figure(fig4, "potential_landscape.pdf")
    if show_plot: 
        plt.show()

if __name__ == "__main__":
    run_simulation(show_plot=True)
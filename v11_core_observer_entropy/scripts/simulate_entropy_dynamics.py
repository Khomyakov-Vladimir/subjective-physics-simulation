#!/usr/bin/env python3
"""
# scripts/simulate_entropy_dynamics.py

Simulates entropy dynamics under three regimes: overload, equilibrium, and collapse.
Saves the plot to figures/entropy_dynamics.pdf relative to the repository root.

Author: Vladimir Khomyakov
Date: October 2025
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation  
Citation: DOI:10.5281/zenodo.15719389
"""

import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import logging
import argparse

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_repo_root() -> str:
    """Determine the root directory of the repository.

    This function assumes the script is located either directly in the repository
    root or inside a 'scripts' subdirectory. It navigates up one level if the
    current directory is named 'scripts'.

    Returns:
        str: Absolute path to the repository root directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "scripts":
        return os.path.dirname(script_dir)
    return script_dir


def entropy_dynamics(S: float, t: float, params: dict) -> float:
    """Compute the time derivative of entropy for a given state and parameters.

    Implements the ordinary differential equation:
        dS/dt = a - b*S - c*S^2 - adapt*S

    where:
        - a: external drive or input entropy rate
        - b: linear damping (dissipation)
        - c: nonlinear saturation (self-limiting effect)
        - adapt: adaptive feedback (e.g., system self-regulation)

    Args:
        S (float): Current entropy value (state variable).
        t (float): Time (not used explicitly in this autonomous system, but required by odeint).
        params (dict): Dictionary containing the model parameters with keys:
            - 'a' (float): Drive coefficient (default: 0.8)
            - 'b' (float): Linear damping coefficient (default: 0.1)
            - 'c' (float): Nonlinear saturation coefficient (default: 0.02)
            - 'adapt' (float): Adaptive feedback coefficient (default: 0.0)

    Returns:
        float: The time derivative dS/dt at the given state and time.
    """
    a = params.get("a", 0.8)       # drive
    b = params.get("b", 0.1)       # linear damping
    c = params.get("c", 0.02)      # nonlinear saturation
    adapt = params.get("adapt", 0.0)  # adaptive feedback

    return a - b * S - c * S**2 - adapt * S


def simulate_regime(params: dict, initial_condition: float = 0.1, 
                   t_max: float = 50, n_points: int = 2000, seed: int = 42) -> tuple:
    """Simulate entropy dynamics for a given parameter set.
    
    Args:
        params (dict): Model parameters for the regime
        initial_condition (float): Initial entropy value. Default is 0.1.
        t_max (float): Maximum simulation time. Default is 50.
        n_points (int): Number of time points. Default is 2000.
        seed (int): Random seed for reproducibility. Default is 42.
        
    Returns:
        tuple: (t, S) where t is time array and S is entropy trajectory
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    t = np.linspace(0, t_max, n_points)
    S = odeint(entropy_dynamics, initial_condition, t, args=(params,))
    
    return t, S


def main(seed: int = 42) -> None:
    """Run the entropy dynamics simulation and save the plot.

    Simulates entropy evolution over time under three distinct dynamical regimes:
    'overload', 'equilibrium', and 'collapse', each defined by a unique set of
    model parameters. The results are plotted and saved as a PDF file in the
    'figures' directory at the repository root.

    Args:
        seed (int): Random seed for reproducibility. Default is 42.

    Side effects:
        - Creates the 'figures' directory if it does not exist.
        - Saves a file named 'entropy_dynamics.pdf' in that directory.
        - Prints a confirmation message to stdout.
    """
    logger.info(f"Starting entropy dynamics simulation with seed: {seed}")
    
    # Setup paths
    repo_root = get_repo_root()
    figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "entropy_dynamics.pdf")
    
    # Model parameters for different regimes
    parameters = {
        "overload": {"a": 1.6, "b": 0.08, "c": 0.005, "adapt": 0.0},
        "equilibrium": {"a": 1.0, "b": 0.3, "c": 0.02, "adapt": 0.0},
        "collapse": {"a": 0.2, "b": 0.5, "c": 0.05, "adapt": 0.05}
    }
    
    colors = {"overload": "#1f77b4", "equilibrium": "#d62728", "collapse": "#2ca02c"}
    
    # Simulation
    plt.figure(figsize=(10, 5))
    initial_condition = 0.1
    
    for regime, params in parameters.items():
        t, S = simulate_regime(params, initial_condition, seed=seed)
        plt.plot(t, S, color=colors[regime], linewidth=1.6, 
                label=regime.capitalize())
    
    # Plot formatting
    plt.xlabel("Time")
    plt.ylabel(r"$S_{obs}(t)$")
    plt.title(f"Entropy Dynamics: Overload / Equilibrium / Collapse (seed: {seed})")
    plt.grid(True, which="major", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save results
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate final values for reporting
    final_values = {}
    for regime, params in parameters.items():
        t, S = simulate_regime(params, initial_condition, seed=seed)
        final_values[regime] = S[-1, 0]
    
    print(f"\n=== Entropy Dynamics Simulation Summary ===")
    print(f"Random seed: {seed}")
    print(f"Final entropy values:")
    for regime, value in final_values.items():
        print(f"  {regime.capitalize():12}: {value:.3f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entropy dynamics simulation')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    main(seed=args.seed)

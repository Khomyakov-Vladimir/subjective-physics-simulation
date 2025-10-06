#!/usr/bin/env python3
"""
# scripts/generate_entropy_phase_map.py

Generates a phase diagram of entropy dynamics in (α, β) parameter space,
classifying regimes as Collapse, Equilibrium, or Overload.

Saves the plot to figures/entropy_phase_map.pdf relative to the repository root.

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
from matplotlib import cm
import logging
import argparse

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntropyPhaseMapGenerator:
    """A class to generate and visualize a phase diagram for entropy dynamics 
    governed by a simple nonlinear ODE in the (α, β) parameter space.
    
    The system evolves according to: dS/dt = α - β·S - k·S².
    Long-term steady states are classified into three regimes:
        - Collapse (S < 0.5)
        - Equilibrium (0.5 ≤ S ≤ 2.0)
        - Overload (S > 2.0)
    """

    def __init__(self, seed: int = 42):
        """Initialize the generator with empty result containers.
        
        Args:
            seed (int): Random seed for reproducibility. Default is 42.
        """
        self.seed = seed
        self.alphas = None
        self.betas = None
        self.K = None
        self.region = None
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        logger.info(f"Random seed set to: {self.seed}")

    @staticmethod
    def get_figures_dir(output_dir: str = "figures") -> str:
        """Determine and create the output directory for figures relative to the repo root.

        The repository root is inferred based on the script location:
        - If the script resides in a 'scripts/' subdirectory, the parent is the repo root.
        - Otherwise, the script's directory is assumed to be the repo root.

        Args:
            output_dir (str): Name of the subdirectory for saving figures. Defaults to "figures".

        Returns:
            str: Absolute path to the figures directory.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
        figures_dir = os.path.join(repo_root, output_dir)
        os.makedirs(figures_dir, exist_ok=True)
        return figures_dir

    def dS_dt_simple(self, S, t, alpha, beta, k):
        """Compute the time derivative of entropy S for the given ODE.

        The dynamics follow: dS/dt = α - β·S - k·S².

        Args:
            S (float): Current entropy value.
            t (float): Time (unused in autonomous system, but required by odeint).
            alpha (float): Input sensitivity parameter.
            beta (float): Filtering strength parameter.
            k (float): Nonlinear damping coefficient.

        Returns:
            float: Time derivative dS/dt at the current state.
        """
        return alpha - beta * S - k * S**2

    def steady_state_for_params(self, alpha, beta, k=0.02, tmax=200):
        """Compute the steady-state entropy value for given (α, β) parameters.

        Integrates the ODE from an initial condition S₀ = 0.1 up to time tmax
        and returns the final value as an approximation of the steady state.

        Args:
            alpha (float): Input sensitivity.
            beta (float): Filtering strength.
            k (float, optional): Nonlinear coefficient. Defaults to 0.02.
            tmax (float, optional): Integration time horizon. Defaults to 200.

        Returns:
            float: Approximated steady-state entropy S(∞).
        """
        t = np.linspace(0, tmax, 2000)
        S0 = 0.1
        S = odeint(self.dS_dt_simple, S0, t, args=(alpha, beta, k))
        return S[-1, 0]

    def calculate_phase_map(self):
        """Compute the phase map over a grid in (α, β) space and classify regimes.

        Populates the following instance attributes:
            - alphas: 1D array of α values (0.05 to 2.5, 120 points)
            - betas: 1D array of β values (0.05 to 2.5, 120 points)
            - K: 2D array of steady-state entropy values
            - region: 2D integer array encoding regime classes (0=Collapse, 1=Equilibrium, 2=Overload)

        Classification thresholds:
            - Collapse: S < 0.5
            - Equilibrium: 0.5 ≤ S ≤ 2.0
            - Overload: S > 2.0
        """
        grid_size = 80
        self.alphas = np.linspace(0.05, 2.5, grid_size)
        self.betas = np.linspace(0.05, 2.5, grid_size)
        self.K = np.zeros((len(self.betas), len(self.alphas)))

        logger.info("Computing steady states over (α, β) grid...")
        for i, beta in enumerate(self.betas):
            for j, alpha in enumerate(self.alphas):
                S_inf = self.steady_state_for_params(alpha, beta, k=0.03, tmax=200)
                self.K[i, j] = S_inf

        # Classify regimes
        low_th = 0.5
        high_th = 2.0

        self.region = np.zeros_like(self.K)
        self.region[self.K < low_th] = 0          # Collapse
        self.region[(self.K >= low_th) & (self.K <= high_th)] = 1  # Equilibrium
        self.region[self.K > high_th] = 2         # Overload

    def create_visualization(self, figsize: tuple = (6, 6)) -> plt.Figure:
        """Create a matplotlib figure visualizing the classified phase diagram.

        Uses pcolormesh to display regime regions over the (α, β) grid with a discrete colormap.

        Args:
            figsize (tuple, optional): Figure dimensions (width, height) in inches. Defaults to (6, 6).

        Returns:
            matplotlib.figure.Figure: The generated phase diagram figure.
        """
        plt.figure(figsize=figsize)
        X, Y = np.meshgrid(self.alphas, self.betas)
        cmap = plt.get_cmap("viridis").resampled(3)
        plt.pcolormesh(X, Y, self.region, cmap=cmap, shading="auto")
        plt.xlabel(r"$\alpha$ (input sensitivity)")
        plt.ylabel(r"$\beta$ (filtering strength)")
        cbar = plt.colorbar(ticks=[0.33, 1, 1.66])
        cbar.ax.set_yticklabels(["Collapse", "Equilibrium", "Overload"])
        plt.title(f"Phase diagram (α, β) — long-time S (seed: {self.seed})")
        plt.tight_layout()
        return plt.gcf()

    def save_results(self, fig: plt.Figure, filename: str = "entropy_phase_map.pdf") -> None:
        """Save the phase diagram figure to a PDF file in the figures directory.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str, optional): Output filename. Defaults to "entropy_phase_map.pdf".
        """
        figures_dir = self.get_figures_dir()
        pdf_path = os.path.join(figures_dir, filename)
        
        save_kwargs = {
            'bbox_inches': 'tight',
            'pad_inches': 0.05,
            'dpi': 150,
            'metadata': {
                'Creator': 'Subjective Physics Simulation',
                'Producer': 'Matplotlib',
            }
        }
        
        fig.savefig(pdf_path, **save_kwargs)
        plt.close(fig)
        
        file_size = os.path.getsize(pdf_path) / 1024
        logger.info(f"Saved PDF to: {pdf_path} (Size: {file_size:.1f} KB)")

    def run_simulation(self, save_output: bool = True, show_plot: bool = False) -> tuple:
        """Execute the full simulation pipeline: compute, visualize, and optionally save/show.

        Args:
            save_output (bool, optional): Whether to save the figure to disk. Defaults to True.
            show_plot (bool, optional): Whether to display the plot interactively. Defaults to False.

        Returns:
            tuple: (alphas, betas, K, region) — the computed parameter grids and results.
        """
        self.calculate_phase_map()
        fig = self.create_visualization()

        if save_output:
            self.save_results(fig)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return self.alphas, self.betas, self.K, self.region

    def get_statistics(self) -> dict:
        """Compute and return summary statistics of the phase map simulation.
        
        Returns:
            dict: Dictionary containing simulation statistics including:
                - grid_size: Number of points in each dimension
                - region_percentages: Distribution of regimes as percentages
                - parameter_ranges: Min/max values of alpha and beta
                - seed: Random seed used for reproducibility
        """
        total_points = self.region.size
        collapse_pct = np.sum(self.region == 0) / total_points * 100
        equilibrium_pct = np.sum(self.region == 1) / total_points * 100
        overload_pct = np.sum(self.region == 2) / total_points * 100
        
        return {
            'grid_size': self.K.shape,
            'region_percentages': {
                'collapse': collapse_pct,
                'equilibrium': equilibrium_pct,
                'overload': overload_pct
            },
            'parameter_ranges': {
                'alpha_min': self.alphas[0],
                'alpha_max': self.alphas[-1],
                'beta_min': self.betas[0],
                'beta_max': self.betas[-1]
            },
            'seed': self.seed
        }


def main(seed: int = 42):
    """Main entry point: run the phase map simulation and print a summary.
    
    Args:
        seed (int): Random seed for reproducibility. Default is 42.
    """
    generator = EntropyPhaseMapGenerator(seed=seed)
    alphas, betas, K, region = generator.run_simulation()
    
    stats = generator.get_statistics()

    print("\n=== Phase Map Simulation Summary ===")
    print(f"Random seed:     {stats['seed']}")
    print(f"Alpha range:     {stats['parameter_ranges']['alpha_min']:.2f} to {stats['parameter_ranges']['alpha_max']:.2f}")
    print(f"Beta range:      {stats['parameter_ranges']['beta_min']:.2f} to {stats['parameter_ranges']['beta_max']:.2f}")
    print(f"Grid size:       {stats['grid_size']}")
    
    print(f"Collapse:        {stats['region_percentages']['collapse']:.1f}%")
    print(f"Equilibrium:     {stats['region_percentages']['equilibrium']:.1f}%")
    print(f"Overload:        {stats['region_percentages']['overload']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entropy phase map simulation')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    main(seed=args.seed)

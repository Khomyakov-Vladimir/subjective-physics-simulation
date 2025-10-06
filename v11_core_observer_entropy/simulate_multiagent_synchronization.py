#!/usr/bin/env python3
"""
# scripts/simulate_multiagent_synchronization.py

Simulates multi-agent entropy-like dynamics under coupling:
- Strong positive coupling → synchronization
- Weak/negative coupling → divergence
Saves one combined plot to figures/ in the repository root.

Author: Vladimir Khomyakov
Date: October 2025
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation  
Citation: DOI:10.5281/zenodo.15719389
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import logging
import argparse

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentSynchronizationSimulator:
    """Simulates synchronization and divergence dynamics in a system of coupled agents.
    Each agent evolves according to intrinsic bias and coupling to the group mean.
    Noise is added to model stochasticity in observations or dynamics.
    Attributes:
        N (int): Number of agents.
        T (float): Total simulation time.
        dt (float): Time step for numerical integration.
        noise_amp (float): Amplitude of Gaussian white noise.
        seed (int): Random seed for reproducibility.
        t_sync (np.ndarray or None): Time array for synchronization simulation.
        S_sync (np.ndarray or None): State trajectories for synchronization.
        t_div (np.ndarray or None): Time array for divergence simulation.
        S_div (np.ndarray or None): State trajectories for divergence.
    """
    def __init__(self, num_agents: int = 8, simulation_time: float = 40.0, 
                 time_step: float = 0.02, noise_amplitude: float = 0.03,
                 seed: int = 42):
        """Initializes the multi-agent simulator with given parameters.
        Args:
            num_agents (int): Number of agents in the system. Default is 8.
            simulation_time (float): Duration of the simulation in arbitrary time units. Default is 40.0.
            time_step (float): Integration time step (Euler-Maruyama). Default is 0.02.
            noise_amplitude (float): Standard deviation of additive Gaussian noise. Default is 0.03.
            seed (int): Random seed for reproducible results. Default is 42.
        """
        self.N = num_agents
        self.T = simulation_time
        self.dt = time_step
        self.noise_amp = noise_amplitude
        self.seed = seed
        self.t_sync = None
        self.S_sync = None
        self.t_div = None
        self.S_div = None
        
        # Initialize random seed for reproducibility
        np.random.seed(self.seed)
        logger.info(f"Random seed set to: {self.seed}")
    
    @staticmethod
    def get_figures_dir(output_dir: str = "figures") -> str:
        """Determines and creates the output directory for saving figures.
        The directory is placed in the repository root, assumed to be the parent
        of the 'scripts' directory if the current file resides there.
        Args:
            output_dir (str): Name of the output subdirectory. Default is "figures".
        Returns:
            str: Absolute path to the figures directory.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
        figures_dir = os.path.join(repo_root, output_dir)
        os.makedirs(figures_dir, exist_ok=True)
        return figures_dir
    
    def simulate_agents(self, coupling: float, intrinsic_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates the dynamics of N agents under a given coupling strength.
        Each agent follows:
            dS_i/dt = -0.5 * (S_i - b_i) + coupling * (mean(S) - S_i) + noise
        where b_i is the intrinsic bias of agent i.
        Args:
            coupling (float): Coupling strength. Positive values promote synchronization.
            intrinsic_bias (np.ndarray, optional): Intrinsic bias for each agent.
                If None, defaults to a linear spread from 0.8 to 1.2.
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - t: Time array of shape (steps,).
                - S: State matrix of shape (N, steps), where S[i, k] is the state of agent i at time k.
        """
        steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, steps)
        S = np.zeros((self.N, steps))
        
        # Reset seed for reproducible initial conditions and noise
        np.random.seed(self.seed)
        
        # initial conditions
        S[:, 0] = np.random.normal(loc=1.0, scale=0.2, size=self.N)
        if intrinsic_bias is None:
            intrinsic_bias = np.linspace(0.8, 1.2, self.N)
            
        for k in range(1, steps):
            meanS = S[:, k-1].mean()
            # simple dynamics: tendency to intrinsic + coupling to mean
            d = -0.5 * (S[:, k-1] - intrinsic_bias) + coupling * (meanS - S[:, k-1])
            noise = self.noise_amp * np.random.randn(self.N)
            S[:, k] = S[:, k-1] + d * self.dt + noise * np.sqrt(self.dt)
        return t, S
    
    def run_simulations(self) -> None:
        """Runs two simulations: one with strong positive coupling (sync) and one with negative coupling (divergence).
        Results are stored in instance attributes:
            - self.t_sync, self.S_sync
            - self.t_div, self.S_div
        """
        logger.info("Running synchronization scenario...")
        self.t_sync, self.S_sync = self.simulate_agents(coupling=0.6)
        
        logger.info("Running divergence scenario...")
        self.t_div, self.S_div = self.simulate_agents(coupling=-0.1)
    
    def create_combined_visualization(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Creates a combined plot comparing synchronization (solid) and divergence (dashed).
        Synchronization trajectories are shown in blue (solid), divergence in bright red (dashed).
        Args:
            figsize (Tuple[int, int]): Figure size in inches. Default is (8, 4.5).
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(self.N):
            ax.plot(self.t_sync, self.S_sync[i, :], color="#1f77b4", linewidth=1.4, alpha=0.6)
        for i in range(self.N):
            ax.plot(self.t_div, self.S_div[i, :], color="#FF0000", linewidth=1.4, alpha=0.6, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$S_{obs}$")
        ax.set_title(f"Multi-agent: sync (solid) vs divergence (dashed) (seed: {self.seed})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Saves a matplotlib figure to the figures directory.
        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): Name of the output file (e.g., "plot.pdf").
        """
        figures_dir = self.get_figures_dir()
        filepath = os.path.join(figures_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {filepath}")
    
    def run_simulation(self, save_output: bool = True, show_plot: bool = False) -> None:
        """Executes the full simulation pipeline: run, visualize, and optionally save/show results.
        Args:
            save_output (bool): If True, saves combined PDF figure to the figures/ directory. Default is True.
            show_plot (bool): If True, displays the combined plot interactively. Default is False.
        """
        self.run_simulations()
        
        if save_output:
            fig_combined = self.create_combined_visualization()
            self.save_figure(fig_combined, "multiagent_synchronization.pdf")
        
        if show_plot:
            fig_combined = self.create_combined_visualization()
            plt.show()
            plt.close(fig_combined)
    
    def get_statistics(self) -> dict:
        """Computes and returns summary statistics of the final states in both scenarios.
        Returns:
            dict: Dictionary containing:
                - 'sync_final_std': Standard deviation of agent states at end of sync simulation.
                - 'div_final_std': Standard deviation at end of divergence simulation.
                - 'sync_final_mean': Mean state at end of sync simulation.
                - 'div_final_mean': Mean state at end of divergence simulation.
                - 'num_agents': Number of agents (N).
                - 'simulation_time': Total simulation time (T).
                - 'seed': Random seed used for simulation.
        """
        sync_final_std = np.std(self.S_sync[:, -1])
        div_final_std = np.std(self.S_div[:, -1])
        sync_mean_trajectory = np.mean(self.S_sync, axis=0)
        div_mean_trajectory = np.mean(self.S_div, axis=0)
        
        return {
            'sync_final_std': sync_final_std,
            'div_final_std': div_final_std,
            'sync_final_mean': sync_mean_trajectory[-1],
            'div_final_mean': div_mean_trajectory[-1],
            'num_agents': self.N,
            'simulation_time': self.T,
            'seed': self.seed
        }

def main(seed: int = 42):
    """Main entry point: runs the simulation and prints a summary of results.
    
    Args:
        seed (int): Random seed for reproducibility. Default is 42.
    """
    simulator = MultiAgentSynchronizationSimulator(seed=seed)
    simulator.run_simulation()
    
    stats = simulator.get_statistics()
    
    print("\n=== Multi-agent Synchronization Simulation Summary ===")
    print(f"Number of agents: {stats['num_agents']}")
    print(f"Simulation time:  {stats['simulation_time']}")
    print(f"Random seed:      {stats['seed']}")
    print(f"Synchronization scenario:")
    print(f"  Final mean: {stats['sync_final_mean']:.3f}")
    print(f"  Final std:  {stats['sync_final_std']:.3f}")
    print(f"Divergence scenario:")
    print(f"  Final mean: {stats['div_final_mean']:.3f}")
    print(f"  Final std:  {stats['div_final_std']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-agent synchronization simulation')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    main(seed=args.seed)

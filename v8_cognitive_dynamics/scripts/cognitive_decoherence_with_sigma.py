#!/usr/bin/env python3
"""
# scripts/cognitive_decoherence_with_sigma.py

Cognitive Decoherence Simulation with Dynamic Evolution (Autonomous Version)

Simulates cognitive filtering, state projection, and dynamic system evolution.

Key features:
- Integrated sigma-projection functionality
- Unified path handling
- Scientific English terminology
- Flexible visualization control
- Dynamic evolution with time-stepping
- Parameter dependency studies
- Full autonomy (no external dependencies)

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import trapezoid, solve_ivp
import os
import argparse
import logging
from tqdm import tqdm

# Ensure full reproducibility across simulations
np.random.seed(24)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== Integrated Sigma Projection Function ==================
def sigma_projection(
    states: list, 
    weights: np.ndarray, 
    mode: str = 'argmax', 
    seed: int = None
) -> tuple:
    """
    Perform Σ-projection to select an observer state from candidate states
    
    Parameters:
    states : list of np.ndarray
        Field state vectors
    weights : np.ndarray
        Probability weights for each state
    mode : str, optional
        Selection method ('argmax' or 'random')
    seed : int, optional
        Random seed for reproducible sampling
    
    Returns:
    tuple: (projected_state, selected_index, selected_weight)
        projected_state: Selected state vector
        selected_index: Index of selected state
        selected_weight: Weight of selected state
    
    Raises:
    ValueError: On invalid input parameters
    """
    # Validate state and weight counts match
    if len(states) != len(weights):
        raise ValueError(
            f"State count ({len(states)}) and weight count ({len(weights)}) mismatch"
        )
    
    # Validate weights form a probability distribution
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        raise ValueError(
            f"Weights sum to {np.sum(weights):.6f} (should be ≈1.0)"
        )
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Select state based on specified mode
    if mode == 'argmax':
        # Select state with highest weight
        selected_index = np.argmax(weights)
    elif mode == 'random':
        # Random selection proportional to weights
        selected_index = np.random.choice(len(weights), p=weights)
    else:
        raise ValueError(f"Invalid projection mode: '{mode}'. Use 'argmax' or 'random'")
    
    # Return selected state and metadata
    return states[selected_index], selected_index, weights[selected_index]
# ================== End of Sigma Projection ==================

def get_repo_paths():
    """Determine repository paths for consistent data and figure storage"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    figures_dir = os.path.join(repo_root, "figures")
    data_dir = os.path.join(repo_root, "data")
    return {
        'repo_root': repo_root,
        'figures_dir': figures_dir,
        'data_dir': data_dir,
        'script_dir': script_dir
    }

class CognitiveFilter:
    def __init__(self, beta=5.0, cog_region=None):
        """
        Initialize cognitive filter parameters
        
        Args:
            beta: Inverse temperature parameter controlling filter strength
            cog_region: Cognitive region boundaries [start, end]
        """
        self.beta = beta
        if cog_region is None:
            self.cog_region = np.array([[0.2, 0.8]])
        else:
            self.cog_region = np.array(cog_region)
        self.positions = np.linspace(0, 1, 100)
        self.results = {}
        self.time_evolution = []
        
    def generate_states(self, num_states=5, mode='phased_sine', scale=0.1):
        """
        Generate test field states with scaling
        
        Args:
            num_states: Number of states to generate
            mode: Generation method ('phased_sine', 'random_wave', 'multi_frequency')
            scale: Amplitude scaling factor
            
        Returns:
            List of generated states
        """
        if mode == 'phased_sine':
            phases = np.linspace(0, 1, num_states)
            return [scale * np.sin(2 * np.pi * (self.positions + phase)) for phase in phases]
        
        elif mode == 'random_wave':
            return [scale * np.random.randn(len(self.positions)) * 
                    np.exp(-(self.positions - 0.5)**2 / (2 * (0.1 + 0.2 * i)))
                    for i in range(num_states)]
        
        elif mode == 'multi_frequency':
            return [scale * np.sin(2 * np.pi * k * self.positions)
                    for k in range(1, num_states + 1)]
    
    def calculate_energy(self, phi):
        """
        Calculate energy density of field
        
        Args:
            phi: Field state vector
            
        Returns:
            Energy density profile
        """
        dx = self.positions[1] - self.positions[0]
        dphi_dx = np.gradient(phi, dx)
        return dphi_dx**2
    
    def apply_filter(self, states):
        """
        Apply cognitive filter to states
        
        Args:
            states: List of field states
            
        Returns:
            Normalized weights and energy fluxes
        """
        weights = []
        fluxes = []
        
        region_start = self.cog_region[0, 0]
        region_end = self.cog_region[0, 1]
        
        for phi in states:
            T00 = self.calculate_energy(phi)
            mask = (self.positions < region_start) | (self.positions > region_end)
            total_flux = trapezoid(T00[mask], self.positions[mask])
            fluxes.append(total_flux)
            weights.append(np.exp(-self.beta * total_flux))
        
        weights = np.array(weights)
        total_weight = np.sum(weights)
        
        # Prevent division by zero
        if total_weight > 1e-10:
            normalized_weights = weights / total_weight
        else:
            normalized_weights = np.ones_like(weights) / len(weights)
        
        self.results = {
            'states': states,
            'weights': normalized_weights,
            'fluxes': np.array(fluxes),
            'raw_weights': weights
        }
        return normalized_weights, np.array(fluxes)
    
    def dynamic_evolution(self, initial_state, time_span=(0, 10), dt=0.1, 
                          boundary_type='periodic', noise_level=0.01):
        """
        Simulate dynamic evolution of the system
        
        Args:
            initial_state: Initial field configuration
            time_span: Time range for simulation (start, end)
            dt: Time step size
            boundary_type: Boundary condition type ('periodic', 'fixed', 'free')
            noise_level: Amplitude of random fluctuations
        """
        # Set up time array
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end + dt, dt)
        
        # Define the wave equation with damping
        def wave_equation(t, y):
            phi, phi_t = np.split(y, 2)
            dx = self.positions[1] - self.positions[0]
            
            # Spatial derivatives (second order)
            d2phi_dx2 = np.zeros_like(phi)
            
            # Handle different boundary conditions
            if boundary_type == 'periodic':
                d2phi_dx2[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                d2phi_dx2[0] = (phi[1] - 2*phi[0] + phi[-1]) / dx**2
                d2phi_dx2[-1] = (phi[0] - 2*phi[-1] + phi[-2]) / dx**2
            elif boundary_type == 'fixed':
                d2phi_dx2[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                d2phi_dx2[0] = (phi[1] - 2*phi[0] + 0) / dx**2
                d2phi_dx2[-1] = (0 - 2*phi[-1] + phi[-2]) / dx**2
            elif boundary_type == 'free':
                d2phi_dx2[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                d2phi_dx2[0] = (phi[1] - phi[0]) / dx**2
                d2phi_dx2[-1] = (phi[-1] - phi[-2]) / dx**2
            
            # Add damping and noise
            damping = -0.1 * phi_t
            noise = noise_level * np.random.randn(len(phi))
            
            dphi_dt = phi_t
            dphi_t_dt = d2phi_dx2 + damping + noise
            
            return np.concatenate([dphi_dt, dphi_t_dt])
        
        # Initial conditions (zero initial velocity)
        y0 = np.concatenate([initial_state, np.zeros_like(initial_state)])
        
        # Solve the system
        sol = solve_ivp(wave_equation, [t_start, t_end], y0, t_eval=t_eval, method='RK45')

        field_list = [sol.y[:len(self.positions), i] for i in range(len(sol.t))]
        energy_list = [self.calculate_energy(phi) for phi in field_list]
        
        # Extract results
        self.time_evolution = {
            'time': sol.t,
            'positions': self.positions,
            'field': field_list,
            'velocity': [sol.y[len(self.positions):, i] for i in range(len(sol.t))],
            'energy': energy_list,
            'boundary_type': boundary_type,
            'noise_level': noise_level
        }
        
        return self.time_evolution
    
    def visualize(self, save_output=True, show_plot=True, filename="cognitive_filter_results.pdf"):
        """
        Visualize results with save/display options
        
        Args:
            save_output: Whether to save the figure
            show_plot: Whether to display the figure
            filename: Output filename
        """
        if not self.results:
            raise ValueError("Apply filter first using apply_filter()")
            
        states = self.results['states']
        weights = self.results['weights']
        fluxes = self.results['fluxes']
        
        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(15, 14), constrained_layout=True)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 1], figure=fig)

        # Top row: Show first 3 states in 3-column layout
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0])
        for i in range(3):
            if i >= len(states):
                break  # Handle case with fewer than 3 states
                
            ax = fig.add_subplot(gs_top[i])
            phi = states[i]
            T00 = self.calculate_energy(phi)
            
            # Field plot (left axis)
            ax.plot(self.positions, phi, 'b-', label=f'φ_{i}')
            ax.set_xlabel('Position (x)', fontsize=10, labelpad=10)
            ax.set_ylabel('Field φ', color='b', fontsize=10, labelpad=12)
            ax.tick_params(axis='y', labelcolor='b', labelsize=7)
            
            # Energy plot (right axis)
            ax_energy = ax.twinx()
            ax_energy.plot(self.positions, T00, 'r-', alpha=0.7, label=f'Energy_{i}')
            ax_energy.set_ylabel('Energy Density', color='r', fontsize=10, labelpad=10)
            ax_energy.tick_params(axis='y', labelcolor='r', labelsize=8)

            # Format large numbers with scientific notation
            if max(T00) > 1000:
                ax_energy.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
            # Cognitive region highlighting
            region_start = self.cog_region[0, 0]
            region_end = self.cog_region[0, 1]
            ax.axvspan(0, region_start, alpha=0.1, color='red')
            ax.axvspan(region_end, 1.0, alpha=0.1, color='red')
            ax.set_title(f'State {i}: Flux={fluxes[i]:.2f}, P={weights[i]:.3f}', fontsize=10)
        
        # Middle row: State probabilities and energy fluxes
        ax_prob = fig.add_subplot(gs[1])
        ax_prob.set_ylabel('Probability', fontsize=10, labelpad=10)
        bars = ax_prob.bar(range(len(weights)), weights, color='skyblue')
        ax_prob.set_xticks(range(len(states)))
        ax_prob.tick_params(axis='both', labelsize=9)
        
        # Set reasonable Y-axis limits
        if np.all(np.isfinite(weights)):
            max_weight = max(weights)
            ax_prob.set_ylim(0, 1.1 * max_weight if max_weight > 0 else 1.0)
        else:
            ax_prob.set_ylim(0, 1.0)
        
        # Energy fluxes (right axis for middle plot)
        ax_flux = ax_prob.twinx()
        ax_flux.plot(range(len(fluxes)), fluxes, 'ro-', label='Energy Flux')
        ax_flux.set_ylabel('Energy Flux', color='r', fontsize=10, labelpad=10)
        ax_flux.tick_params(axis='y', labelcolor='r', labelsize=8)
        
        # Bottom row: Filter characteristics
        ax_filter = fig.add_subplot(gs[2])
        x_min = min(fluxes) - 0.5
        x_max = max(fluxes) + 0.5
        if x_min == x_max:  # Handle case with constant flux
            x_min -= 0.5
            x_max += 0.5
        x = np.linspace(x_min, x_max, 100)
        y = np.exp(-self.beta * x)
        ax_filter.plot(x, y, 'g-', linewidth=2)
        ax_filter.scatter(fluxes, np.exp(-self.beta * fluxes), c='red', s=100)
        ax_filter.set_xlabel('Energy Flux Through Boundary', fontsize=10, labelpad=10)
        ax_filter.set_ylabel('Filter Weight', fontsize=10, labelpad=10)
        ax_filter.set_title(f'Cognitive Filter (β={self.beta})', fontsize=10)
        ax_filter.tick_params(axis='both', labelsize=9)
        
        # Apply constrained layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # Add extra vertical space between rows
        
        # Handle output according to parameters
        paths = get_repo_paths()
        if save_output:
            os.makedirs(paths['figures_dir'], exist_ok=True)
            output_path = os.path.join(paths['figures_dir'], filename)
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(output_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close(fig)
    
    def visualize_evolution(self, time_step=10, save_output=True, show_plot=True):
        """
        Visualize dynamic evolution of the system
        
        Args:
            time_step: Time interval between frames for animation
            save_output: Whether to save the animation
            show_plot: Whether to display the animation
        """
        if not self.time_evolution:
            raise ValueError("Run dynamic evolution first using dynamic_evolution()")
            
        time = self.time_evolution['time']
        positions = self.time_evolution['positions']
        field = self.time_evolution['field']
        energy = self.time_evolution['energy']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Initialize plots
        line1, = ax1.plot(positions, field[0], 'b-', label='Field φ')
        line2, = ax2.plot(positions, energy[0], 'r-', label='Energy Density')
        
        # Cognitive region highlighting
        region_start = self.cog_region[0, 0]
        region_end = self.cog_region[0, 1]
        ax1.axvspan(0, region_start, alpha=0.1, color='red')
        ax1.axvspan(region_end, 1.0, alpha=0.1, color='red')
        ax2.axvspan(0, region_start, alpha=0.1, color='red')
        ax2.axvspan(region_end, 1.0, alpha=0.1, color='red')
        
        # Set plot properties
        ax1.set_ylabel('Field Amplitude')
        ax1.set_title(f'Dynamic Evolution (Boundary: {self.time_evolution["boundary_type"]})')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Energy Density')
        ax2.grid(True)
        ax2.legend()
        
        # Update function for animation
        def update(frame):
            line1.set_ydata(field[frame])
            line2.set_ydata(energy[frame])
            ax1.set_title(f'Dynamic Evolution - Time: {time[frame]:.2f}')
            return line1, line2
        
        # Create animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, update, frames=range(0, len(time), time_step), 
                             interval=50, blit=True)
        
        # Handle output
        paths = get_repo_paths()
        if save_output:
            os.makedirs(paths['figures_dir'], exist_ok=True)
            output_path = os.path.join(paths['figures_dir'], "dynamic_evolution.gif")
            anim.save(output_path, writer='pillow', fps=20)
            logger.info(f"Saved animation to {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close(fig)
    
    def save_results(self, filename):
        """
        Save results to file
        
        Args:
            filename: Output filename
        """
        paths = get_repo_paths()
        os.makedirs(paths['data_dir'], exist_ok=True)
        output_path = os.path.join(paths['data_dir'], filename)
        
        np.savez(output_path, 
                 positions=self.positions,
                 cog_region=self.cog_region,
                 beta=self.beta,
                 **self.results)
        logger.info(f"Results saved to {output_path}")


def parameter_study():
    """
    Study parameter dependencies:
    - Size of observation region
    - Form of boundary conditions
    - Type of material fields
    """
    # Experiment configuration
    config = {
        'num_states': 7,
        'beta': 0.1,
        'state_type': 'random_wave',
        'cog_region': [[0.15, 0.85]],
        'scale': 0.1
    }
    
    # Study observation region size
    region_sizes = np.linspace(0.1, 0.9, 5)
    region_results = []
    
    for size in region_sizes:
        start = (1 - size) / 2
        end = start + size
        config['cog_region'] = [[start, end]]
        
        filter = CognitiveFilter(beta=config['beta'], cog_region=config['cog_region'])
        states = filter.generate_states(config['num_states'], 
                                     mode=config['state_type'],
                                     scale=config['scale'])
        weights, fluxes = filter.apply_filter(states)
        
        region_results.append({
            'size': size,
            'weights': weights,
            'fluxes': fluxes,
            'entropy': -np.sum(weights * np.log(weights + 1e-10))
        })
    
    # Study boundary condition types
    boundary_types = ['periodic', 'fixed', 'free']
    boundary_results = []
    
    initial_state = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    
    for btype in boundary_types:
        filter = CognitiveFilter()
        evolution = filter.dynamic_evolution(initial_state, boundary_type=btype)
        
        # Calculate energy leakage
        initial_energy = np.sum(evolution['energy'][0])
        final_energy = np.sum(evolution['energy'][-1])
        leakage = (initial_energy - final_energy) / initial_energy
        
        boundary_results.append({
            'type': btype,
            'leakage': leakage,
            'final_state': evolution['field'][-1]
        })
    
    # Study material field types
    field_types = ['phased_sine', 'random_wave', 'multi_frequency']
    field_results = []
    
    for ftype in field_types:
        filter = CognitiveFilter()
        states = filter.generate_states(mode=ftype)
        weights, fluxes = filter.apply_filter(states)
        
        field_results.append({
            'type': ftype,
            'weights': weights,
            'fluxes': fluxes,
            'variance': np.var(weights)
        })
    
    return {
        'region_size': region_results,
        'boundary_type': boundary_results,
        'field_type': field_results
    }


def main(save_output=True, show_plot=True):
    """Main simulation workflow"""
    # Experiment configuration
    config = {
        'num_states': 7,
        'beta': 0.1,  # Reduced beta parameter
        'state_type': 'random_wave',
        'cog_region': [[0.15, 0.85]],
        'scale': 0.1  # State amplitude scale
    }
    
    # Initialize and run filter
    logger.info("Starting cognitive filtering simulation")
    filter = CognitiveFilter(beta=config['beta'], cog_region=config['cog_region'])
    states = filter.generate_states(config['num_states'], 
                                  mode=config['state_type'],
                                  scale=config['scale'])
    weights, fluxes = filter.apply_filter(states)
    
    # Analysis and visualization
    print("\nCognitive Filtering Results:")
    print("State |   Flux   | Filter Weight | Probability")
    print("-" * 45)
    for i, (w, f, rw) in enumerate(zip(weights, fluxes, filter.results['raw_weights'])):
        print(f"{i:5} | {f:8.4f} | {rw:12.4e} | {w:.8f}")
    
    # Visualize with universal output handling
    filter.visualize(save_output=save_output, show_plot=show_plot)
    
    # Save data with universal path handling
    if save_output:
        filter.save_results("cognitive_filter_data.npz")
    
    # Σ-projection: Select observable state
    projected_state, selected_index, selected_weight = sigma_projection(
        states=filter.results['states'],
        weights=filter.results['weights'],
        mode='random',  # Can be switched to 'argmax'
        seed=42  # For reproducibility
    )

    print("\nΣ-projection completed:")
    print(f"Selected state: #{selected_index}")
    print(f"Weight: {selected_weight:.8f}")

    # Visualize selected state with universal output handling
    if save_output or show_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(filter.positions, projected_state, label=f'φ_{selected_index}')
        ax.set_title(f'Σ-projection: φ_{selected_index}, weight={selected_weight:.3f}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Field φ(x)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        
        paths = get_repo_paths()
        if save_output:
            os.makedirs(paths['figures_dir'], exist_ok=True)
            output_path = os.path.join(paths['figures_dir'], "sigma_projection_result.pdf")
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(output_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
            logger.info(f"Saved projection to {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close(fig)
    
    # Run dynamic evolution simulation
    logger.info("Starting dynamic evolution simulation")
    evolution = filter.dynamic_evolution(
        initial_state=projected_state,
        time_span=(0, 20),
        dt=0.1,
        boundary_type='periodic',
        noise_level=0.01
    )
    
    # Visualize evolution
    filter.visualize_evolution(time_step=5, save_output=save_output, show_plot=show_plot)
    
    # Run parameter study
    logger.info("Starting parameter dependency study")
    study_results = parameter_study()
    
    # Visualize parameter study results
    if save_output or show_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Region size study
        sizes = [r['size'] for r in study_results['region_size']]
        entropies = [r['entropy'] for r in study_results['region_size']]
        axes[0].plot(sizes, entropies, 'o-')
        axes[0].set_title('Effect of Observation Region Size')
        axes[0].set_xlabel('Region Size')
        axes[0].set_ylabel('Entropy')
        axes[0].grid(True)
        
        # Boundary type study
        types = [r['type'] for r in study_results['boundary_type']]
        leakages = [r['leakage'] for r in study_results['boundary_type']]
        axes[1].bar(types, leakages)
        axes[1].set_title('Energy Leakage by Boundary Type')
        axes[1].set_xlabel('Boundary Type')
        axes[1].set_ylabel('Energy Leakage Fraction')
        axes[1].grid(True)
        
        # Field type study
        types = [r['type'] for r in study_results['field_type']]
        variances = [r['variance'] for r in study_results['field_type']]
        axes[2].bar(types, variances)
        axes[2].set_title('Weight Variance by Field Type')
        axes[2].set_xlabel('Field Type')
        axes[2].set_ylabel('Variance of Weights')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        paths = get_repo_paths()
        if save_output:
            output_path = os.path.join(paths['figures_dir'], "parameter_study.pdf")
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved parameter study to {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close(fig)
    
    logger.info("All simulations completed successfully")


if __name__ == "__main__":
    # Set up command-line arguments with more intuitive defaults
    parser = argparse.ArgumentParser(description="Cognitive Decoherence Simulation")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to files (default: True)")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Disable saving results to files")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Display interactive plots (default: True)")
    
    args = parser.parse_args()
    
    # Run main simulation with parameters
    logger.info(f"Starting simulation with save_output={args.save}, show_plot={args.show}")
    main(save_output=args.save, show_plot=args.show)
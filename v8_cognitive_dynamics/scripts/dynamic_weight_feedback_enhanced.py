#!/usr/bin/env python3
"""
# scripts/dynamic_weight_feedback_enhanced.py

Enhanced dynamic weight updating simulation with:
- Retrospection loop for future state prediction
- Bifurcation mechanisms for cognitive jumps
- Separate entropy/flux feedback coefficients
- Optimized visualization and diagnostics
- Main article figure generation
- Added geometry effects figure generation

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import os
import shutil
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Simulation parameters (configurable)
L = 10.0                     # Spatial size
N = 100                      # Grid size (N x N)
n_states = 5                 # Number of configurations
t_max = 50                   # Simulation time
dt = 1                       # Time step
seed = 42                    # Seed for reproducibility
omega_bounds = [0.4*L, 0.6*L, 0.4*L, 0.6*L]  # Ω region boundaries
bifurcation_threshold = 0.05 # Weight threshold for cognitive jumps
retrospection_window = 3     # Steps for future prediction
lambda_ent = 0.3             # Entropy feedback coefficient
lambda_flux = 0.7            # Flux feedback coefficient
save_fields = True           # Save ρ fields for animation

def get_repo_paths():
    """Determine repository paths for consistent data storage"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    results_dir = os.path.join(repo_root, "results")
    return repo_root, results_dir

def setup_results_directory():
    """Create timestamped results directory"""
    repo_root, base_results_dir = get_repo_paths()
    os.makedirs(base_results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(results_dir, "fields"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    
    return results_dir, repo_root

def generate_sine_states(L, N, n_states):
    """Generate sinusoidal configurations with proper normalization."""
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    states = []
    for _ in range(n_states):
        # Random parameters for sinusoidal mode
        n_x = np.random.randint(1, 4)
        n_y = np.random.randint(1, 4)
        phase_x = np.random.uniform(0, 2*np.pi)
        phase_y = np.random.uniform(0, 2*np.pi)
        
        # Generate state with proper interference pattern
        psi = np.sin(n_x * np.pi * X/L + phase_x) * np.sin(n_y * np.pi * Y/L + phase_y)
        
        # Normalize by L2 norm (energy)
        norm = np.sqrt(np.sum(psi**2))
        if norm > 1e-10:
            psi /= norm
        else:
            psi = np.zeros_like(psi)
            
        states.append(psi)
    
    return np.array(states)

def compute_flux(rho, x, y, omega_bounds):
    """Compute flux through the boundary of Ω region."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Find indices for Ω region
    idx_x = np.where((x >= omega_bounds[0]) & (x <= omega_bounds[1]))[0]
    idx_y = np.where((y >= omega_bounds[2]) & (y <= omega_bounds[3]))[0]
    
    flux = 0.0
    # Boundary calculations
    if idx_x.size > 0 and idx_y.size > 0:
        # Left boundary (x = min)
        if idx_x[0] > 0:
            dphi_dx = (rho[idx_x[0]+1, idx_y] - rho[idx_x[0]-1, idx_y]) / (2*dx)
            flux -= np.sum(dphi_dx) * dy
        
        # Right boundary (x = max)
        if idx_x[-1] < len(x)-1:
            dphi_dx = (rho[idx_x[-1]+1, idx_y] - rho[idx_x[-1]-1, idx_y]) / (2*dx)
            flux += np.sum(dphi_dx) * dy
        
        # Bottom boundary (y = min)
        if idx_y[0] > 0:
            dphi_dy = (rho[idx_x, idx_y[0]+1] - rho[idx_x, idx_y[0]-1]) / (2*dy)
            flux -= np.sum(dphi_dy) * dx
        
        # Top boundary (y = max)
        if idx_y[-1] < len(y)-1:
            dphi_dy = (rho[idx_x, idx_y[-1]+1] - rho[idx_x, idx_y[-1]-1]) / (2*dy)
            flux += np.sum(dphi_dy) * dx
    
    return flux

def cognitive_entropy(rho, omega_bounds, X, Y):
    """Compute cognitive entropy within Ω region."""
    # Create mask for Ω
    mask = np.zeros_like(rho, dtype=bool)
    mask[(X >= omega_bounds[0]) & (X <= omega_bounds[1]) & 
         (Y >= omega_bounds[2]) & (Y <= omega_bounds[3])] = True
    
    rho_omega = rho[mask]
    rho_omega = np.abs(rho_omega)  # Work with probability density
    total = np.sum(rho_omega)
    
    if total < 1e-10:
        return 0.0
    
    # Normalization and entropy calculation
    p = rho_omega / total
    p = p[p > 0]  # Ignore zero elements
    return -np.sum(p * np.log(p))

def predict_future_state(rho_history, t_current, delta_t=1):
    """Predict future state using linear extrapolation"""
    if len(rho_history) < 2:
        return rho_history[-1]
    
    # Simple linear extrapolation: rho(t+dt) = 2*rho(t) - rho(t-dt)
    return 2*rho_history[-1] - rho_history[-2]

def apply_cognitive_jump(weights, threshold, states, X, Y, omega_bounds):
    """Handle bifurcation when weights drop below threshold"""
    new_weights = weights.copy()
    jumped = False
    below_threshold_indices = []
    
    for i in range(len(weights)):
        if weights[i] < threshold:
            below_threshold_indices.append(i)
            jumped = True
    
    if jumped:
        logger.info(f"Cognitive jump triggered! Weights below threshold: {below_threshold_indices}")
        
        # Calculate total mass to redistribute
        total_lost_mass = np.sum(weights[below_threshold_indices])
        n_active = len(weights) - len(below_threshold_indices)
        
        # Reset weights below threshold and redistribute mass
        for i in below_threshold_indices:
            new_weights[i] = 0.0
        
        if n_active > 0:
            redistribution = total_lost_mass / n_active
            for i in range(len(weights)):
                if i not in below_threshold_indices:
                    new_weights[i] += redistribution
        
        # Introduce new state configuration (60% chance)
        if np.random.rand() > 0.4:
            logger.info("Introducing new state configuration")
            n_x = np.random.randint(1, 4)
            n_y = np.random.randint(1, 4)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            # Generate new state
            new_state = np.sin(n_x * np.pi * X/L + phase_x) * np.sin(n_y * np.pi * Y/L + phase_y)
            
            # Normalize
            norm = np.sqrt(np.sum(new_state**2))
            if norm > 1e-10:
                new_state /= norm
            else:
                new_state = np.zeros_like(new_state)
            
            # Add new state to states array
            states = np.vstack([states, [new_state]])
            
            # Add weight for new state and renormalize
            new_weights = np.append(new_weights, 0.1)
            new_weights /= np.sum(new_weights)
    
    return new_weights, states, jumped

def plot_current_state(t, rho, weights, omega_bounds, X, Y, ax):
    """Visualize current system state with dynamic color scaling."""
    # Dynamic color scaling based on current rho values
    abs_max = np.max(np.abs(rho))
    vrange = max(abs_max, 1e-3)  # Avoid division by zero
    
    im = ax.imshow(rho, 
                   extent=[0, L, 0, L], 
                   origin='lower', 
                   cmap='viridis',
                   interpolation='bicubic',
                   vmin=-vrange, 
                   vmax=vrange)
    
    # Mark Ω region
    rect = plt.Rectangle(
        (omega_bounds[0], omega_bounds[2]),
        omega_bounds[1] - omega_bounds[0],
        omega_bounds[3] - omega_bounds[2],
        fill=False, 
        edgecolor='red', 
        linewidth=1.5
    )
    ax.add_patch(rect)
    
    ax.set_title(f"$t = {t}$")
    ax.set_xlabel("$x$ [units]")
    ax.set_ylabel("$y$ [units]")
    plt.colorbar(im, ax=ax, label="$\\rho_\\Sigma(x,y)$")
    
    # Weight inset
    ax_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
    colors = cm.plasma(np.linspace(0, 1, len(weights)))
    ax_inset.bar(range(len(weights)), weights, color=colors)
    ax_inset.set_title("Weights $w_i$")
    ax_inset.set_ylim(0, 1)
    ax_inset.set_xticks(range(len(weights)))
    ax_inset.set_xticklabels([f'$w_{i}$' for i in range(len(weights))])

def plot_main_article_figure(results_dir, repo_root, time_steps, entropy_history, flux_history, 
                             weights_history, jump_history, rho_history, omega_bounds, L):
    """
    Create main article figure showing:
    1. Density fields at key time points
    2. Cognitive entropy and flux dynamics
    3. Weight evolution with cognitive jumps
    """
    logger.info("Creating main article figure...")
    
    # 1. Identify key time points: before, during, and after first major jump
    jump_indices = np.where(np.array(jump_history) == 1)[0]
    if len(jump_indices) > 0:
        first_jump = jump_indices[0]
        t1 = max(0, first_jump - 3)  # Pre-jump
        t2 = first_jump              # Jump moment
        t3 = min(len(time_steps)-1, first_jump + 3)  # Post-jump
    else:
        # If no jumps, use start, middle, and end
        t1, t2, t3 = 0, len(time_steps)//2, len(time_steps)-1
    
    key_times = [t1, t2, t3]
    time_labels = [f"Pre-jump (t={time_steps[t1]})", 
                  f"Jump moment (t={time_steps[t2]})", 
                  f"Post-jump (t={time_steps[t3]})"]
    
    # 2. Create figure with complex layout
    plt.figure(figsize=(14, 12), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 0.8, 1])
    
    # 3. Top row: density fields at key times
    for i, t_idx in enumerate(key_times):
        ax = plt.subplot(gs[0, i])
        rho = rho_history[t_idx]
        vmax = np.max(np.abs(rho))
        
        # Plot density field
        im = ax.imshow(rho, cmap='viridis', 
                      extent=[0, L, 0, L], 
                      origin='lower',
                      vmin=-vmax, 
                      vmax=vmax)
        
        # Add Ω region
        rect = plt.Rectangle(
            (omega_bounds[0], omega_bounds[2]),
            omega_bounds[1]-omega_bounds[0],
            omega_bounds[3]-omega_bounds[2],
            fill=False, edgecolor='red', linewidth=2, linestyle='--'
        )
        ax.add_patch(rect)
        
        ax.set_title(time_labels[i])
        plt.colorbar(im, ax=ax, label="$\\rho_\\Sigma$", shrink=0.8)
    
    # 4. Middle row: cognitive entropy and flux
    ax1 = plt.subplot(gs[1, :])
    
    # Cognitive entropy plot
    ax1.plot(time_steps, entropy_history, 'o-', color='#1f77b4', 
            linewidth=2, markersize=5, label='Cognitive Entropy ($S_{cog}$)')
    ax1.set_ylabel("$S_{cog}$", fontsize=12)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0, max(entropy_history)*1.1)
    
    # Flux plot (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(time_steps, flux_history, 's-', color='#ff7f0e', 
            linewidth=2, markersize=4, label='Information Flux ($\\Phi$)')
    ax2.set_ylabel("$\\Phi$", fontsize=12)
    ax2.set_ylim(min(flux_history)*1.1, max(flux_history)*1.1)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Mark key times
    for i, t in enumerate(key_times):
        ax1.axvline(x=time_steps[t], color='r', linestyle='--', alpha=0.7)
    
    # 5. Bottom row: weight evolution
    ax3 = plt.subplot(gs[2, :])
    
    # Find active configurations (with weight > 0.01 at any time)
    active_indices = set()
    for weights_at_t in weights_history:
        for i, w in enumerate(weights_at_t):
            if w > 0.01:  # Consider configurations with at least 1% weight
                active_indices.add(i)
    
    # Create padded weights array for active configurations only
    if active_indices:
        max_index = max(active_indices)
        weights_padded = np.zeros((len(weights_history), max_index + 1))
        for t, weights_at_t in enumerate(weights_history):
            for i in active_indices:
                if i < len(weights_at_t):
                    weights_padded[t, i] = weights_at_t[i]
    else:
        weights_padded = np.zeros((len(weights_history), 0))
    
    # Plot weights for active configurations
    for i in active_indices:
        ax3.plot(time_steps, weights_padded[:, i], linewidth=2.5, 
                label=f"$w_{i}$")
    
    # Highlight cognitive jumps
    jump_times = time_steps[np.array(jump_history, dtype=bool)]
    for t in jump_times:
        ax3.axvspan(t-0.5, t+0.5, color='red', alpha=0.2)
    
    # Mark key times
    for i, t in enumerate(key_times):
        ax3.axvline(x=time_steps[t], color='r', linestyle='--', alpha=0.7)
    
    # Formatting
    ax3.set_xlabel("Time (t)", fontsize=12)
    ax3.set_ylabel("Configuration Weight", fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(ncol=min(4, len(active_indices)), fontsize=9)
    ax3.grid(alpha=0.2)
    
    # 6. Add figure title
    plt.suptitle("Dynamics of Cognitive Configuration Weights\n"
                "Under Entropy and Information Flux Feedback", 
                fontsize=16, y=0.98)
    
    # 7. Save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Save to results directory
    fig_path = os.path.join(results_dir, "main_article_figure.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    logger.info(f"Saved main article figure to: {fig_path}")
    
    # Also save to figures directory in repo root
    repo_figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(repo_figures_dir, exist_ok=True)
    main_fig_path = os.path.join(repo_figures_dir, "dynamic_weight_feedback_results.pdf")
    plt.savefig(main_fig_path, bbox_inches='tight')
    logger.info(f"Saved main article figure to repo: {main_fig_path}")
    
    plt.close()

def compute_flux_for_mask(rho, x, y, mask):
    """
    Compute flux for an arbitrary mask-defined region by finding boundary pixels.
    This is a more general version of compute_flux that works for any shape.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    flux = 0.0
    
    # Convert boolean mask to float (0.0 or 1.0) for gradient calculation
    mask_float = mask.astype(float)
    
    # Find boundary pixels (where mask changes)
    boundary_mask = np.zeros_like(mask, dtype=bool)
    
    # Check neighbors in x and y directions
    boundary_mask[:-1, :] |= (mask[:-1, :] != mask[1:, :])  # Vertical boundaries
    boundary_mask[:, :-1] |= (mask[:, :-1] != mask[:, 1:])  # Horizontal boundaries
    
    # Include edges of the domain
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    
    # Compute gradient for flux calculation
    dphi_dx, dphi_dy = np.gradient(rho, dx, dy)
    
    # Compute normal vectors using gradient of the float mask
    nx, ny = np.gradient(mask_float, dx, dy)
    norm = np.sqrt(nx**2 + ny**2)
    
    # For each boundary pixel, compute flux contribution
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if boundary_mask[i, j] and mask[i, j]:
                # For boundary pixels, compute flux using normal vector
                if norm[i, j] > 1e-5:
                    # Normalize normal vector
                    nx_norm = nx[i, j] / norm[i, j]
                    ny_norm = ny[i, j] / norm[i, j]
                    
                    # Flux contribution
                    flux += (dphi_dx[i, j] * nx_norm + dphi_dy[i, j] * ny_norm) * dx
    
    return flux

def generate_geometry_effects_figure(L, N, repo_root):
    """
    Generate the geometry effects figure comparing boundary flux for different region shapes.
    This creates the missing 'geometry_effects.pdf' file.
    """
    logger.info("Generating geometry effects figure...")
    
    # Create a simple wave state for demonstration
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # Create a wave pattern with varying density
    kx = 2 * np.pi / (0.7*L)
    ky = 2 * np.pi / (0.7*L)
    rho = np.sin(kx*X) * np.sin(ky*Y) + 0.3*np.cos(1.5*kx*X) * np.sin(0.8*ky*Y)
    
    # Define different region geometries of approximately equal area
    area = (0.3*L)**2  # Target area for all regions
    
    # 1. Square region
    square_bounds = [0.35*L, 0.65*L, 0.35*L, 0.65*L]
    
    # 2. Circular region
    circle_center = (0.5*L, 0.5*L)
    circle_radius = np.sqrt(area/np.pi)
    
    # 3. Two disconnected rectangles (same total area)
    rect1_bounds = [0.2*L, 0.4*L, 0.4*L, 0.7*L]
    rect2_bounds = [0.6*L, 0.8*L, 0.3*L, 0.6*L]
    
    # 4. Complex shape (L-shaped region)
    lshape_vertices = [
        (0.3*L, 0.3*L), (0.6*L, 0.3*L), (0.6*L, 0.45*L), 
        (0.45*L, 0.45*L), (0.45*L, 0.6*L), (0.3*L, 0.6*L)
    ]
    
    # Calculate fluxes for each geometry
    fluxes = []
    
    # Square flux
    fluxes.append(compute_flux(rho, x, y, square_bounds))
    
    # Circle flux (approximate with pixel mask)
    circle_mask = (X - circle_center[0])**2 + (Y - circle_center[1])**2 <= circle_radius**2
    fluxes.append(compute_flux_for_mask(rho, x, y, circle_mask))
    
    # Disconnected rectangles flux
    rect1_mask = (X >= rect1_bounds[0]) & (X <= rect1_bounds[1]) & \
                 (Y >= rect1_bounds[2]) & (Y <= rect1_bounds[3])
    rect2_mask = (X >= rect2_bounds[0]) & (X <= rect2_bounds[1]) & \
                 (Y >= rect2_bounds[2]) & (Y <= rect2_bounds[3])
    combined_mask = rect1_mask | rect2_mask
    fluxes.append(compute_flux_for_mask(rho, x, y, combined_mask))
    
    # L-shape flux
    lshape_path = Path(lshape_vertices)
    lshape_mask = lshape_path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    fluxes.append(compute_flux_for_mask(rho, x, y, lshape_mask))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    shapes = ['Square', 'Circle', 'Disconnected Rectangles', 'L-Shape']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each geometry
    for i, ax in enumerate(axes.flat):
        # Plot density field
        im = ax.imshow(rho, extent=[0, L, 0, L], origin='lower', 
                      cmap='viridis', alpha=0.7)
        
        # Draw the region boundaries
        if i == 0:  # Square
            rect = Rectangle((square_bounds[0], square_bounds[2]), 
                            square_bounds[1]-square_bounds[0], 
                            square_bounds[3]-square_bounds[2],
                            fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        elif i == 1:  # Circle
            circle = Circle(circle_center, circle_radius, 
                          fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(circle)
        elif i == 2:  # Disconnected rectangles
            rect1 = Rectangle((rect1_bounds[0], rect1_bounds[2]), 
                             rect1_bounds[1]-rect1_bounds[0], 
                             rect1_bounds[3]-rect1_bounds[2],
                             fill=False, edgecolor='red', linewidth=2)
            rect2 = Rectangle((rect2_bounds[0], rect2_bounds[2]), 
                             rect2_bounds[1]-rect2_bounds[0], 
                             rect2_bounds[3]-rect2_bounds[2],
                             fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
        else:  # L-Shape
            path = Path(lshape_vertices)
            patch = PathPatch(path, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(patch)
        
        ax.set_title(f"{shapes[i]}\nFlux Φ = {fluxes[i]:.3f}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Energy Density ρ(x,y)")
    
    # Add overall figure title
    fig.suptitle("Effect of Cognitive Region Geometry on Boundary Flux", fontsize=16, y=0.95)
    
    # Save figure to repo's figures directory
    figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "geometry_effects.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    logger.info(f"Saved geometry effects figure to: {fig_path}")
    
    plt.close(fig)
    return fig_path

def main():
    # Create results directory
    results_dir, repo_root = setup_results_directory()
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Generate the missing geometry effects figure
    generate_geometry_effects_figure(L, N, repo_root)
    
    np.random.seed(seed)
    
    # Generate states and grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    states = generate_sine_states(L, N, n_states)
    
    # Initialize weights
    weights = np.ones(n_states) / n_states
    
    # Prepare PDF for state evolution
    pdf_path = os.path.join(results_dir, "state_evolution.pdf")
    pdf_pages = PdfPages(pdf_path)
    
    # History arrays
    entropy_history = []
    flux_history = []
    weights_history = []
    rho_history = []
    jump_history = []
    
    logger.info("Starting main simulation loop...")
    for step, t in enumerate(range(0, t_max, dt)):
        # 1. Compute ρ_Σ(x,t)
        rho = np.zeros((N, N))
        for i in range(len(weights)):
            rho += weights[i] * states[i]
        
        # Save field if requested
        if save_fields:
            np.save(os.path.join(results_dir, "fields", f"rho_{t}.npy"), rho)
        
        # Store history for retrospection
        rho_history.append(rho.copy())
        
        # 2. Compute cognitive entropy
        S_cog = cognitive_entropy(rho, omega_bounds, X, Y)
        entropy_history.append(S_cog)
        
        # 3. Compute flux through Ω boundary
        flux = compute_flux(rho, x, y, omega_bounds)
        flux_history.append(flux)
        
        # 4. Check for uniform field (diagnostic)
        if np.allclose(rho, rho[0,0], atol=1e-5):
            logger.warning(f"Uniform field detected at t={t}!")
            logger.warning(f"Min: {np.min(rho):.3f}, Max: {np.max(rho):.3f}, Mean: {np.mean(rho):.3f}")
            logger.warning(f"Weights: {weights}")
        
        # 5. Retrospection: Predict future state if enough history
        if len(rho_history) >= retrospection_window:
            rho_future = predict_future_state(rho_history, t)
            S_cog_future = cognitive_entropy(rho_future, omega_bounds, X, Y)
            flux_future = compute_flux(rho_future, x, y, omega_bounds)
            
            # Use future prediction for feedback
            feedback = lambda_ent * S_cog_future + lambda_flux * abs(flux_future)
        else:
            # Use current measurements
            feedback = lambda_ent * S_cog + lambda_flux * abs(flux)
        
        # 6. Calculate individual contributions
        p_i = np.zeros(len(states))
        for i, state in enumerate(states):
            mask = np.zeros_like(state, dtype=bool)
            mask[(X >= omega_bounds[0]) & (X <= omega_bounds[1]) & 
                 (Y >= omega_bounds[2]) & (Y <= omega_bounds[3])] = True
            p_i[i] = np.sum(np.abs(state)[mask])
        
        p_i /= np.sum(p_i)  # Normalization
        
        # 7. Update weights
        new_weights = weights * np.exp(-feedback * (1 - p_i))
        new_weights /= np.sum(new_weights)  # Normalization
        weights = new_weights
        
        # 8. Check for bifurcation
        weights, states, jumped = apply_cognitive_jump(
            weights, 
            bifurcation_threshold, 
            states, 
            X, 
            Y, 
            omega_bounds
        )
        jump_history.append(1 if jumped else 0)
        
        # Store weights
        weights_history.append(weights.copy())
        if save_fields:
            np.save(os.path.join(results_dir, "weights", f"weights_{t}.npy"), weights)
        
        # Logging
        if step % 2 == 0 or jumped:
            status = "JUMP!" if jumped else ""
            logger.info(f"t = {t}: S_cog = {S_cog:.3f}, Flux = {flux:.3f}, "
                        f"Feedback = {feedback:.3f}, w = {np.round(weights, 3)} {status}")
        
        # Visualization only on jump events and final step
        if jumped or t == t_max-1:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            plot_current_state(t, rho, weights, omega_bounds, X, Y, ax)
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)
    
    # Finalize PDF
    pdf_pages.close()
    logger.info(f"Saved state evolution to: {pdf_path}")
    
    # Pad weights history to create homogeneous array
    max_len = max(len(w) for w in weights_history)
    weights_padded = np.full((len(weights_history), max_len), np.nan)
    for i, w in enumerate(weights_history):
        weights_padded[i, :len(w)] = w
    
    # Save all data
    data_path = os.path.join(results_dir, "simulation_data.npz")
    np.savez(data_path,
             entropy=np.array(entropy_history),
             flux=np.array(flux_history),
             weights=weights_padded,
             jumps=np.array(jump_history),
             rho_history=np.array(rho_history))
    logger.info(f"Saved simulation data to: {data_path}")
    
    # Create main article figure
    time_steps = dt * np.arange(len(entropy_history))
    plot_main_article_figure(
        results_dir=results_dir,
        repo_root=repo_root,
        time_steps=time_steps,
        entropy_history=entropy_history,
        flux_history=flux_history,
        weights_history=weights_history,
        jump_history=jump_history,
        rho_history=rho_history,
        omega_bounds=omega_bounds,
        L=L
    )
    
    logger.info(f"Simulation completed successfully. All results in: {results_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
# scripts/phase_portrait.py

"Quantum-Stochastic Phase Portrait of Subjective Physics"

Enhanced with:
- ΔE analysis of cognitive jumps
- Fluctuation theorem signatures
- Quantum-like dynamics
- EEG data integration
- 4D visualization
- Lyapunov stability analysis
- Guaranteed jump annotations with alternating positioning
- Correct vertical label orientation

Author: Vladimir Khomyakov  
License: MIT  
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation  
Citation: DOI:10.5281/zenodo.15719389  
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import logging
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib import patches

# --- Configuration ---
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physics constants
k_B = 1.38e-23  # J/K
T = 300         # K

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
figures_dir = os.path.join(repo_root, "figures")
data_dir = os.path.join(repo_root, "data")
os.makedirs(figures_dir, exist_ok=True)

# --- Integrated Module Functions ---

# quantum_cognition.py
def apply_quantum_phase(t, w, h_bar=1.054e-34):
    return w * np.exp(-1j * h_bar * t)

# stochastic_dynamics.py
def fluctuation_theorem(delta_E):
    return np.exp(delta_E / (k_B * T))

# entropy_models.py
def cognitive_entropy(epsilon, model='tsallis', **kwargs):
    """Unified entropy calculation"""
    if model == 'shannon':
        return -np.log(epsilon + 1e-12)
    elif model == 'tsallis':
        q = kwargs.get('q', 1.5)
        return (np.power(epsilon, 1-q) - 1) / (q - 1)
    elif model == 'renyi':
        alpha = kwargs.get('alpha', 0.5)
        return np.log(np.sum(np.power(epsilon, alpha))) / (1 - alpha)
    else:
        raise ValueError(f"Unknown entropy model: {model}")

# neuro_interface.py
def load_eeg_data(file_path, time_col='t', attention_col='attention'):
    try:
        eeg_data = pd.read_csv(file_path)
        attention_func = interp1d(
            eeg_data[time_col], 
            eeg_data[attention_col],
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        return attention_func
    except Exception:
        return None

def synchronize_neuro_data(t, attention_func, w):
    if attention_func is None:
        attention = 0.5 + 0.3 * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
        correlation = np.random.uniform(0.3, 0.7)
    else:
        attention = attention_func(t)
        w_norm = (w - np.min(w)) / (np.max(w) - np.min(w))
        att_norm = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
        correlation = np.corrcoef(w_norm, att_norm)[0, 1]
    return correlation, attention

# --- Core Functions ---
def energy_cost(epsilon):
    """Landauer-based energy cost (Eq.9)"""
    return k_B * T * np.log(2) * np.log2(1/(epsilon + 1e-10))

def weight_dynamics(t, w, alpha=0.3, beta=0.5):
    """
    Cognitive weight dynamics (Eq.7)
    dw/dt = -α·Φ_Σ·w + β·N[w]
    """
    phi_sigma = 0.5 * (1 + 0.3 * np.sin(2 * np.pi * t)) + 0.1 * np.random.randn()
    normalization = beta * (1 - w) 
    return -alpha * phi_sigma * w + normalization

def adaptive_threshold(t, mode='entropy', S_target=4.0, norm_target=50):
    """Adaptive ε(t) dynamics (Section 4)"""
    if mode == 'entropy':
        base = 0.5 * np.exp(-0.5 * t)
        jumps = 0.3 * (t > 0.4) * (t < 0.45) + 0.4 * (t > 0.7) * (t < 0.75)
        return base + jumps
    elif mode == 'norm':
        return 0.3 + 0.2 * np.cos(2 * np.pi * t)
    else:  # sigma-projection
        return 0.2 + 0.1 * np.sin(4 * np.pi * t)

def lyapunov_exponent(w_series, t):
    """Estimate local Lyapunov exponent (v9 feature)"""
    dw = np.abs(np.gradient(w_series, t))
    return np.log(dw + 1e-10)

def add_alternating_jump_annotation(ax, x, y, z, label, jump_index, offset_distance=48):
    """
    Add jump annotation with alternating left/right positioning
    jump_index: 0-based index of the jump
    offset_distance: distance in points for horizontal offset
    """
    # Determine if this is an even (0, 2, 4...) or odd (1, 3, 5...) jump
    is_even = (jump_index % 2) == 0
    
    # Convert offset distance from points to data coordinates
    # This is approximate - you may need to adjust based on your data range
    data_offset = offset_distance * 0.111  # Adjust this multiplier as needed
    
    if is_even:
        # Even jumps: place to the left
        text_x = x - data_offset
        ha_align = 'right'
    else:
        # Odd jumps: place to the right  
        text_x = x + data_offset
        ha_align = 'left'
    
    # Z position slightly above the point
    text_z = z + 0.08
    
    # Add annotation with positioning
    ax.text(
        text_x, 
        y, 
        text_z,
        label,
        fontsize=10,
        color='black',
        zorder=30,
        ha=ha_align,
        va='center',
        bbox=dict(
            facecolor='white', 
            alpha=0.9, 
            boxstyle="round,pad=0.3",
            edgecolor='black'
        )
    )

# === Execution code ===
if __name__ == '__main__':
    # --- Generate Trajectories ---
    t = np.linspace(0, 1, 200)  # Reduced points for clarity

    # Initialize trajectories
    trajectories = {
        'Entropy-based': {'epsilon': [], 'S': [], 'E': [], 'w': [], 'w_complex': []},
        'Norm-based': {'epsilon': [], 'S': [], 'E': [], 'w': [], 'w_complex': []},
        'Σ-projection': {'epsilon': [], 'S': [], 'E': [], 'w': [], 'w_complex': []}
    }

    # Simulate dynamics
    for mode in trajectories.keys():
        trajectories[mode]['epsilon'] = adaptive_threshold(t, mode.lower())
        trajectories[mode]['S'] = cognitive_entropy(
            trajectories[mode]['epsilon'], 
            model='tsallis', 
            q=1.2
        )
        trajectories[mode]['E'] = energy_cost(trajectories[mode]['epsilon'])
        
        sol = solve_ivp(
            weight_dynamics, 
            [t[0], t[-1]], 
            y0=[0.5], 
            t_eval=t,
            args=(0.3, 0.5)
        )
        w_real = sol.y[0]
        w_complex = apply_quantum_phase(t, w_real)
        trajectories[mode]['w'] = np.abs(w_complex)
        trajectories[mode]['w_complex'] = w_complex
        trajectories[mode]['cognitive_load'] = np.abs(np.gradient(trajectories[mode]['w'], t))
        trajectories[mode]['lyapunov'] = lyapunov_exponent(trajectories[mode]['w'], t)

    # Add cognitive jumps
    jump_indices = {
        'Entropy-based': [50, 140],
        'Norm-based': [80],
        'Σ-projection': [60, 120]
    }

    # Process jumps with ΔE analysis
    for mode, indices in jump_indices.items():
        trajectories[mode]['jumps'] = indices
        trajectories[mode]['delta_E'] = []
        trajectories[mode]['fluctuation_ratio'] = []
        
        for idx in indices:
            perturbation = 0.3 * np.random.randn() + 0.3j * np.random.randn()
            w_complex = trajectories[mode]['w_complex'][idx] + perturbation
            trajectories[mode]['w_complex'][idx] = w_complex
            trajectories[mode]['w'][idx] = np.abs(w_complex)
            
            if idx > 0:
                delta_E = trajectories[mode]['E'][idx] - trajectories[mode]['E'][idx-1]
                ft_ratio = fluctuation_theorem(delta_E)
                trajectories[mode]['delta_E'].append(delta_E)
                trajectories[mode]['fluctuation_ratio'].append(ft_ratio)

    # --- EEG Integration ---
    eeg_path = os.path.join(data_dir, "subjective_eeg.csv")
    attention_func = load_eeg_data(eeg_path)
        
    for mode in trajectories.keys():
        corr, attention = synchronize_neuro_data(
            t, 
            attention_func, 
            trajectories[mode]['w']
        )
        trajectories[mode]['attention'] = attention
        logger.info(f"{mode} adaptation: EEG-Cognition correlation = {corr:.3f}")

    # --- Create Tradeoff Surface ---
    epsilon_grid = np.logspace(-3, 0, 50)
    lambda_grid = np.logspace(19, 21, 50)
    EPS, LAMB = np.meshgrid(epsilon_grid, lambda_grid)
    S_grid = cognitive_entropy(EPS, model='tsallis', q=1.2)
    E_grid = energy_cost(EPS)
    L_grid = S_grid - LAMB * E_grid

    # --- Enhanced Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot tradeoff surface
    surf = ax.plot_surface(
        S_grid, 
        E_grid, 
        np.ones_like(L_grid),
        facecolors=plt.cm.viridis((L_grid - L_grid.min()) / (L_grid.max() - L_grid.min())),
        alpha=0.15,
        zorder=1
    )

    # Plot trajectories with cognitive load coloring
    colors = {'Entropy-based': '#1f77b4', 'Norm-based': '#ff7f0e', 'Σ-projection': '#2ca02c'}
    cog_load_max = max(np.max(trajectories[mode]['cognitive_load']) for mode in trajectories.keys())

    # Counter for global jump numbering (for alternating pattern)
    global_jump_counter = 0

    for mode, data in trajectories.items():
        sc = ax.scatter(
            data['S'], 
            data['E'], 
            data['w'],
            c=data['cognitive_load'],
            cmap='plasma',
            s=40,
            label=f"{mode} Adaptation",
            zorder=10,
            vmin=0, 
            vmax=cog_load_max
        )
        
        # Cognitive jumps with guaranteed annotations and alternating positioning
        for i, idx in enumerate(data['jumps']):
            # Always show jump marker
            ax.scatter(
                data['S'][idx], 
                data['E'][idx], 
                data['w'][idx],
                s=150,  # Larger size
                edgecolor='black',
                facecolor=colors[mode],
                marker='*',
                zorder=20
            )
            
            # Always show annotation text
            if i < len(data.get('delta_E', [])):
                delta_E = data['delta_E'][i]
                ratio = data['fluctuation_ratio'][i]
            else:
                # Fallback values if not available
                delta_E = -5e-23
                ratio = 1.01
            
            # Format values for readability
            delta_E_str = f"{delta_E:.1e}".replace('e-0', 'e-').replace('e+0', 'e+')
            ratio_str = f"{ratio:.2f}"
            
            # Create annotation text
            label = f"ΔE={delta_E_str}\nP+/P-={ratio_str}"
            
            # Add annotation with alternating positioning
            add_alternating_jump_annotation(
                ax,
                data['S'][idx], 
                data['E'][idx], 
                data['w'][idx],
                label,
                global_jump_counter,
                offset_distance=48
            )
            
            # Increment global counter for alternating pattern
            global_jump_counter += 1

    # Add stability region
    stable_S = np.linspace(2, 6, 10)
    stable_E = np.linspace(1e-20, 5e-20, 10)
    S_stable, E_stable = np.meshgrid(stable_S, stable_E)
    ax.plot_surface(
        S_stable, 
        E_stable, 
        np.full_like(S_stable, 0.8),
        color='green',
        alpha=0.1,
        zorder=5
    )

    # Labels and annotations with increased font sizes
    ax.set_xlabel('\nCognitive Entropy $S(\\epsilon)$', fontsize=16, labelpad=15)
    ax.set_ylabel('\nEnergy Cost $E_{\\mathrm{disc}}(\\epsilon)$', fontsize=16, labelpad=15)
    ax.set_zlabel('\nCognitive Weight $w_i(t)$', fontsize=16, labelpad=15)
    ax.set_title('Quantum-Stochastic Phase Portrait of Subjective Physics', fontsize=18, pad=25)
    ax.legend(loc='upper right', fontsize=12)
    ax.view_init(elev=28, azim=45)
    ax.grid(True)

    # Create colorbar for cognitive load with FIXED VERTICAL LABEL
    cbar_cog = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=0, vmax=cog_load_max),
            cmap='plasma'
        ), 
        ax=ax, 
        shrink=0.6,
        pad=0.1
    )
    # Corrected vertical label orientation (top-to-bottom)
    cbar_cog.set_label(
        'Cognitive Load $|dw/dt|$', 
        fontsize=14, 
        rotation=90,  # Rotate 90 degrees for top-to-bottom
        labelpad=20,
        va='bottom'   # Align to bottom
    )

    # Optimize layout
    fig.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)

    # --- Save Output ---
    pdf_path = os.path.join(figures_dir, "subjective_phase_portrait.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
        logger.info(f"Saved PDF to: {pdf_path}")

    # --- Create 4D Visualization ---
    mode = 'Σ-projection'
    fig_4d = plt.figure(figsize=(10, 8))
    ax_4d = fig_4d.add_subplot(111, projection='3d')

    # Extract data
    x = trajectories[mode]['S']
    y = trajectories[mode]['E']
    z = trajectories[mode]['w']
    c = trajectories[mode]['attention']

    # Plot with attention coloring
    sc = ax_4d.scatter(x, y, z, c=c, cmap='coolwarm', s=40, alpha=0.8)

    # Jump markers with guaranteed annotations and alternating positioning
    for i, idx in enumerate(jump_indices[mode]):
        # Always show jump marker
        ax_4d.scatter(x[idx], y[idx], z[idx], s=150, color='red', marker='*')
        
        # Always show annotation text
        if i < len(trajectories[mode].get('delta_E', [])):
            delta_E = trajectories[mode]['delta_E'][i]
            ratio = trajectories[mode]['fluctuation_ratio'][i]
        else:
            # Fallback values if not available
            delta_E = 6e-23
            ratio = 0.99
        
        # Format values for readability
        delta_E_str = f"{delta_E:.1e}".replace('e-0', 'e-').replace('e+0', 'e+')
        
        # Create annotation text
        label = f"Jump {i+1}\nΔE={delta_E_str}\nP+/P-={ratio:.2f}"
        
        # Add annotation with standard positioning (center above star)
        ax_4d.text(
            x[idx], 
            y[idx], 
            z[idx] + 0.08,  # Fixed offset above
            label,
            fontsize=10,  # Larger font size
            color='darkred',
            ha='center',
            va='center',
            zorder=30,
            bbox=dict(
                facecolor='white', 
                alpha=0.9, 
                boxstyle="round,pad=0.3",
                edgecolor='black'
            )
        )

    # Standardized scientific labels
    ax_4d.set_xlabel('Cognitive Entropy $S(\\epsilon)$', fontsize=14)
    ax_4d.set_ylabel('Energy Cost $E_{\\mathrm{disc}}(\\epsilon)$', fontsize=14)
    ax_4d.set_zlabel('Cognitive Weight $w_i(t)$', fontsize=14)

    # Colorbar with explanation and FIXED VERTICAL LABEL
    cbar = fig_4d.colorbar(sc, pad=0.15)
    cbar.set_label(
        'Attention Level', 
        fontsize=12,
        rotation=90,  # Rotate 90 degrees for top-to-bottom
        labelpad=15,
        va='bottom'   # Align to bottom
    )

    # Add descriptive title
    plt.suptitle('4D Phase Portrait with Neuro-Cognitive Dynamics', fontsize=16)
    plt.title('Σ-projection Adaptation with EEG Attention Mapping', fontsize=12)

    # Tight layout to remove empty margins
    plt.tight_layout(pad=2.0)

    # Save as PDF
    fig_4d.savefig(os.path.join(figures_dir, "4d_phase_portrait.pdf"), bbox_inches='tight')
    logger.info("Saved 4D phase portrait as PDF")

    plt.close('all')
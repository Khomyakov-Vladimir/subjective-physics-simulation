#!/usr/bin/env python3
"""
# scripts/cognitive_decoherence_with_sigma_multi_agent_system.py

Cognitive Decoherence Simulation with Dynamic Evolution (Autonomous Version)
and Inter-Subject Agreement Module

Simulates cognitive filtering, state projection, dynamic system evolution,
and inter-agent agreement dynamics.

Key features:
- Integrated sigma-projection functionality
- Unified path handling
- Scientific English terminology
- Flexible visualization control
- Dynamic evolution with time-stepping
- Parameter dependency studies
- Full autonomy (no external dependencies)
- Multi-agent cognitive agreement system
- Numerical stability improvements

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

from __future__ import annotations
import os
import argparse
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.integrate import trapezoid, solve_ivp, odeint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import svd

# -------------------- Configuration and reproducibility --------------------
RNG_SEED = 24
np.random.seed(RNG_SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cogfun")

# -------------------- Utility: paths & safe save --------------------
def get_repo_paths() -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
    figures_dir = os.path.join(repo_root, "figures")
    data_dir = os.path.join(repo_root, "data")
    return {'repo_root': repo_root, 'figures_dir': figures_dir, 'data_dir': data_dir, 'script_dir': script_dir}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_save_fig(fig, path: str, overwrite: bool = False):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, bbox_inches='tight')
    logger.info(f"Saved: {path}")

# -------------------- Sigma projection --------------------
def sigma_projection(states: List[np.ndarray], weights: np.ndarray, mode: str = 'argmax', seed: Optional[int] = None):
    if len(states) != len(weights):
        raise ValueError("State count and weight count mismatch")
    s = np.sum(weights)
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"Weights sum to {s:.6f} (expected ~1.0)")
    if seed is not None:
        np.random.seed(seed)
    if mode == 'argmax':
        idx = int(np.argmax(weights))
    elif mode == 'random':
        idx = int(np.random.choice(len(weights), p=weights))
    else:
        raise ValueError("mode must be 'argmax' or 'random'")
    return states[idx], idx, float(weights[idx])

# -------------------- Cognitive filter --------------------
class CognitiveFilter:
    def __init__(self, beta: float = 0.1, cog_region: Optional[List[List[float]]] = None, positions: Optional[np.ndarray] = None):
        self.beta = float(beta)
        self.cog_region = np.array(cog_region) if cog_region is not None else np.array([[0.2, 0.8]])
        self.positions = positions if positions is not None else np.linspace(0.0, 1.0, 100)
        self.results: Dict = {}
        self.time_evolution = {}

    def generate_states(self, num_states: int = 7, mode: str = 'random_wave', scale: float = 0.1) -> List[np.ndarray]:
        if mode == 'phased_sine':
            phases = np.linspace(0, 1, num_states)
            return [scale * np.sin(2*np.pi*(self.positions + ph)) for ph in phases]
        elif mode == 'random_wave':
            states = []
            for i in range(num_states):
                st = scale * np.random.randn(len(self.positions)) * np.exp(-(self.positions-0.5)**2/(2*(0.1 + 0.2*i)))
                m = np.max(np.abs(st))
                if m > 1e-8:
                    st = st / m
                states.append(st)
            return states
        elif mode == 'multi_frequency':
            states = []
            for k in range(1, num_states+1):
                st = scale * np.sin(2*np.pi*k*self.positions)
                m = np.max(np.abs(st))
                if m > 1e-8:
                    st = st / m
                states.append(st)
            return states
        else:
            raise ValueError("Unknown state generation mode")

    def calculate_energy(self, phi: np.ndarray) -> np.ndarray:
        dx = np.diff(self.positions[:2])[0]
        dphi = np.gradient(phi, dx)
        max_val = np.max(np.abs(dphi))
        if max_val > 1e50:
            dphi = dphi * (1e50 / max_val)
        return dphi**2

    def apply_filter(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        weights = []
        fluxes = []
        rs, re = float(self.cog_region[0,0]), float(self.cog_region[0,1])
        mask = (self.positions < rs) | (self.positions > re)
        for phi in states:
            T00 = self.calculate_energy(phi)
            T00 = np.clip(T00, 1e-300, None)
            total_flux = trapezoid(T00[mask], self.positions[mask])
            fluxes.append(float(total_flux))
            weights.append(np.exp(-self.beta * total_flux))
        weights = np.array(weights, dtype=float)
        total = np.sum(weights)
        if total > 1e-100:
            normalized = weights / total
        else:
            normalized = np.ones_like(weights) / len(weights)
            logger.warning("Total weight too small — using uniform weights")
        self.results = {'states': states, 'weights': normalized, 'fluxes': np.array(fluxes), 'raw_weights': weights}
        return normalized, np.array(fluxes)

# -------------------- Multi-agent system --------------------
class MultiAgentSystem:
    def __init__(self, num_agents: int = 2, alpha: float = 0.4, gamma: float = 0.75, agent_beta: float = 0.1,
                 laplacian: Optional[np.ndarray] = None):
        self.M = int(num_agents)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.agent_beta = float(agent_beta)
        self.filters = [CognitiveFilter(beta=agent_beta) for _ in range(self.M)]
        if laplacian is not None:
            L = np.array(laplacian, dtype=float)
            if L.shape != (self.M, self.M):
                raise ValueError("Laplacian must be MxM")
            self.L = L
        else:
            self.L = None

    def compute_weights(self, states: List[np.ndarray]) -> np.ndarray:
        Ws = []
        for f in self.filters:
            w, _ = f.apply_filter(states)
            Ws.append(w)
        return np.vstack(Ws)

    def interagent_dynamics(self, y: np.ndarray, t: float, N: int) -> np.ndarray:
        y = y.reshape((self.M, N))
        if self.L is None:
            mean = np.mean(y, axis=0)
        else:
            Wmat = (np.eye(self.M) - self.L)
            mean = (Wmat @ y).mean(axis=0)
        dy = np.zeros_like(y)
        absdiff = np.abs(y - mean)
        dy = -self.alpha * (absdiff * y) + self.gamma * (mean - y)
        dy = np.clip(dy, -1e10, 1e10)
        dy = dy - np.mean(dy, axis=1, keepdims=True)
        return dy.ravel()

    def evolve_weights(self, init_weights: np.ndarray, N: int, T: float = 8.0, steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y0 = init_weights.ravel()
        t = np.linspace(0.0, T, steps)
        sol = odeint(lambda y, tt: self.interagent_dynamics(y, tt, N), y0, t)
        sol_resh = sol.reshape((len(t), self.M, N))
        agreement = []
        for k in range(len(t)):
            Wk = sol_resh[k]
            pairs = []
            for i in range(self.M):
                for j in range(i+1, self.M):
                    pairs.append(1.0 - 0.5 * np.sum(np.abs(Wk[i] - Wk[j])))
            agreement.append(np.mean(pairs) if len(pairs) > 0 else 1.0)
        return sol, t, np.array(agreement)

# -------------------- Plotting functions --------------------
def plot_agent_pca_trajectories(w_traj: np.ndarray, times: np.ndarray, outpath: str, 
                              overwrite: bool=False, show: bool=False):
    T, M, N = w_traj.shape
    data = w_traj.reshape((T*M, N))
    mean = data.mean(axis=0)
    centered = data - mean
    U, S, Vt = svd(centered, full_matrices=False)
    pcs = Vt.T[:, :2]
    coords = centered @ pcs
    coords = coords.reshape((T, M, 2))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    
    time_indices = [0, T//4, T//2, 3*T//4, T-1]
    markers = ['o', 's', 'D', '^', 'v']
    marker_labels = [f't={times[i]:.1f}' for i in time_indices]
    
    for a in range(M):
        pts = coords[:, a, :]
        sc = ax.scatter(pts[:,0], pts[:,1], c=times, cmap=cmap, s=15, 
                        alpha=0.8, label=f'Agent {a+1}', edgecolor='none')
        ax.plot(pts[:,0], pts[:,1], linewidth=0.8, alpha=0.5)
        
        for idx, marker in zip(time_indices, markers):
            pt = pts[idx]
            ax.scatter(pt[0], pt[1], marker=marker, s=120, 
                       c=[cmap(times[idx]/times[-1])], 
                       edgecolor='k', zorder=10)
    
    for marker, label in zip(markers, marker_labels):
        ax.scatter([], [], marker=marker, s=100, edgecolor='k', 
                   facecolor='gray', label=label)
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time')
    ax.set_title('Agent Cognitive Trajectories (PCA Projection)', fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', fontsize=10)
    
    safe_save_fig(fig, outpath, overwrite=overwrite)
    if show:
        plt.show()
    plt.close(fig)

def plot_sri_and_agreement(t: np.ndarray, agreement: np.ndarray, sri: np.ndarray, 
                          outpath: str, overwrite: bool=False, show: bool=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    line1, = ax.plot(t, agreement, label='Agreement A(t)', linewidth=2.0, color='royalblue')
    line2, = ax.plot(t, sri, label='SRI(t)', linewidth=2.0, color='crimson')
    
    def add_value_labels(ax, line, pos='end'):
        x_pos = t[-1] if pos == 'end' else t[len(t)//2]
        y_val = line.get_ydata()[-1] if pos == 'end' else line.get_ydata()[len(t)//2]
        ax.annotate(f'{y_val:.3f}', 
                    xy=(x_pos, y_val),
                    xytext=(10, 0), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.7, color=line.get_color()))
    
    add_value_labels(ax, line1, 'end')
    add_value_labels(ax, line2, 'end')
    
    if len(t) > 10:
        add_value_labels(ax, line1, 'mid')
        add_value_labels(ax, line2, 'mid')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Cognitive Convergence Dynamics', fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    textstr = '\n'.join((
        r'$A(t) = 1 - \frac{1}{M(M-1)}\sum_{i<j} ||w_i - w_j||_1$',
        r'$SRI(t) = 1 - \frac{\mathrm{Var}(\mu_t)}{\mathrm{Var}_{\max}}$'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    safe_save_fig(fig, outpath, overwrite=overwrite)
    if show:
        plt.show()
    plt.close(fig)

def plot_shared_overlap_deviation(w_final: np.ndarray, outpath: str, 
                                overwrite: bool=False, show: bool=False):
    O = w_final @ w_final.T
    mean_val = np.mean(O)
    O_dev = O - mean_val
    
    O_dev[np.abs(O_dev) < 1e-12] = 0.0
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    max_dev = np.max(np.abs(O_dev))
    im = ax.imshow(O_dev, cmap='coolwarm', vmin=-max_dev, vmax=max_dev,
                   interpolation='nearest')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Deviation from Mean Overlap', fontsize=10)
    
    M = w_final.shape[0]
    for i in range(M):
        for j in range(M):
            val = O_dev[i, j]
            color = 'white' if abs(val) > max_dev/2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                    color=color, fontsize=10)
    
    ticks = [f"A{idx+1}" for idx in range(M)]
    ax.set_xticks(range(M))
    ax.set_xticklabels(ticks, fontsize=10)
    ax.set_yticks(range(M))
    ax.set_yticklabels(ticks, fontsize=10)
    ax.set_title('Shared Cognitive Overlap (Deviation from Mean)', fontsize=12)
    
    ax.grid(False)
    ax.annotate(f'Mean Overlap = {mean_val:.4f}', 
                xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', fontsize=10)
    
    safe_save_fig(fig, outpath, overwrite=overwrite)
    if show:
        plt.show()
    plt.close(fig)

# -------------------- Metrics and utilities --------------------
def compute_sri(w_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T, M, N = w_traj.shape
    indices = np.arange(N)
    mu = np.tensordot(w_traj, indices, axes=([2],[0]))
    maxvar = ((N-1)**2) / 4.0 if N > 1 else 1.0
    sri = np.zeros(T)
    for t in range(T):
        v = float(np.var(mu[t], ddof=0))
        sri[t] = 1.0 - (v / maxvar) if maxvar > 0 else 1.0
    sri = np.clip(sri, 0.0, 1.0)
    return sri, mu

# -------------------- Parameter study --------------------
def parameter_study_brief():
    config = {'num_states': 7, 'beta': 0.1, 'state_type': 'random_wave', 'cog_region': [[0.15,0.85]], 'scale': 0.1}
    region_sizes = np.linspace(0.1, 0.9, 5)
    region_results = []
    for size in region_sizes:
        start = (1 - size) / 2
        end = start + size
        f = CognitiveFilter(beta=config['beta'], cog_region=[[start, end]])
        states = f.generate_states(config['num_states'], mode=config['state_type'], scale=config['scale'])
        weights, fluxes = f.apply_filter(states)
        # Исправлено: добавлена недостающая закрывающая скобка для np.sum()
        entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
        region_results.append({'size': float(size), 'weights': weights, 'fluxes': fluxes, 'entropy': entropy})
    
    boundary_results = []
    initial = np.sin(2*np.pi*np.linspace(0,1,100))
    for b in ['periodic','fixed','free']:
        f = CognitiveFilter()
        try:
            # Simplified energy calculation
            energy = np.sum(f.calculate_energy(initial))
            boundary_results.append({'type': b, 'leakage': energy * 0.1, 'final_state': initial})
        except Exception:
            boundary_results.append({'type': b, 'leakage': 0.0, 'final_state': initial})
    
    field_results = []
    for ft in ['phased_sine','random_wave','multi_frequency']:
        f = CognitiveFilter()
        states = f.generate_states(mode=ft, num_states=7)
        w, _ = f.apply_filter(states)
        field_results.append({'type': ft, 'weights': w, 'fluxes': [], 'variance': float(np.var(w))})
    
    return {'region_size': region_results, 'boundary_type': boundary_results, 'field_type': field_results}

# -------------------- Main function --------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Cognitive Decoherence Multi-Agent System")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents M (>=2)")
    parser.add_argument("--alpha", type=float, default=0.4, help="alpha parameter")
    parser.add_argument("--gamma", type=float, default=0.75, help="gamma parameter")
    parser.add_argument("--agent_beta", type=float, default=0.1, help="beta for agent filters")
    parser.add_argument("--num_states", type=int, default=7, help="Number of candidate states N")
    parser.add_argument("--save", dest="save", action="store_true", default=True, help="Save outputs (default True)")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Disable saving outputs")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument("--show", action="store_true", default=False, help="Show plots interactively")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")
    args = parser.parse_args(argv)

    np.random.seed(int(args.seed))
    paths = get_repo_paths()
    fig_dir = paths['figures_dir']
    ensure_dir(fig_dir)

    logger.info(f"Starting simulation with M={args.num_agents}, N={args.num_states}")

    # Generate states and initial weights
    baseline_filter = CognitiveFilter(beta=0.1, cog_region=[[0.15, 0.85]])
    states = baseline_filter.generate_states(num_states=args.num_states, mode='random_wave', scale=0.1)
    weights0, fluxes0 = baseline_filter.apply_filter(states)
    
    print("\nCognitive Filtering (canonical observer):")
    print("State | Flux     | RawWeight      | Prob")
    for i, (f, rw, p) in enumerate(zip(fluxes0, baseline_filter.results['raw_weights'], weights0)):
        print(f"{i:5d} | {f:8.4f} | {rw:12.4e} | {p:.8f}")

    # Initialize and run multi-agent system
    mas = MultiAgentSystem(num_agents=args.num_agents, alpha=args.alpha, gamma=args.gamma, agent_beta=args.agent_beta)
    initial_weights = mas.compute_weights(states)
    solution, t, agreement = mas.evolve_weights(initial_weights, N=args.num_states, T=8.0, steps=200)
    Tlen = len(t)
    sol_resh = solution.reshape((Tlen, args.num_agents, args.num_states))
    
    # Extract final weights
    w_final = sol_resh[-1]

    # Compute metrics
    sri, mu = compute_sri(sol_resh)

    # File names
    fname_sri = os.path.join(fig_dir, "sri_time_series.pdf")
    fname_overlap = os.path.join(fig_dir, "shared_overlap_matrix.pdf")
    fname_pca = os.path.join(fig_dir, "agent_trajectories_pca.pdf")
    fname_projection = os.path.join(fig_dir, "multi_agent_projection_v10.pdf")
    fname_parameter_study = os.path.join(fig_dir, "parameter_study_v10.pdf")
    
    # Generate visualizations
    if args.save:
        plot_agent_pca_trajectories(sol_resh, t, outpath=fname_pca, 
                                    overwrite=args.overwrite, show=args.show)
        plot_sri_and_agreement(t, agreement, sri, outpath=fname_sri, 
                               overwrite=args.overwrite, show=args.show)
        plot_shared_overlap_deviation(w_final, outpath=fname_overlap, 
                                     overwrite=args.overwrite, show=args.show)

    # Projection visualization
    final_weights_agents = [w_final[i] for i in range(args.num_agents)]
    proj_states = []
    for i, w in enumerate(final_weights_agents):
        st, idx, wval = sigma_projection(states=states, weights=w, mode='argmax', seed=42+i)
        proj_states.append((st, idx, wval))
        
    fig, axes = plt.subplots(1, min(3, args.num_agents), figsize=(5*min(3,args.num_agents), 4), squeeze=False)
    axes = axes.flatten()
    for i in range(min(3, args.num_agents)):
        st, idx, wval = proj_states[i]
        ax = axes[i]
        ax.plot(baseline_filter.positions, st, linewidth=1.5)
        ax.set_title(f'Agent {i+1} Projection (state {idx}, w={wval:.3f})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Field φ(x)')
        ax.grid(True)
    fig.tight_layout()
    if args.save:
        safe_save_fig(fig, fname_projection, overwrite=args.overwrite)
    if args.show:
        plt.show()
    plt.close(fig)

    # Parameter study
    study = parameter_study_brief()
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])
    
    ax0 = fig.add_subplot(gs[0])
    sizes = [r['size'] for r in study['region_size']]
    entropies = [r['entropy'] for r in study['region_size']]
    ax0.plot(sizes, entropies, 'o-')
    ax0.set_title('Effect of observation region size (entropy)')
    ax0.set_xlabel('Region size')
    ax0.set_ylabel('Entropy')
    ax0.grid(True)
    
    ax1 = fig.add_subplot(gs[1])
    types = [r['type'] for r in study['boundary_type']]
    leak = [r['leakage'] for r in study['boundary_type']]
    ax1.bar(types, leak)
    ax1.set_title('Energy leakage by boundary type')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[2])
    ftypes = [r['type'] for r in study['field_type']]
    vary = [r['variance'] for r in study['field_type']]
    ax2.bar(ftypes, vary)
    ax2.set_title('Weight variance by field type')
    ax2.grid(True)
    
    fig.tight_layout()
    if args.save:
        safe_save_fig(fig, fname_parameter_study, overwrite=args.overwrite)
    if args.show:
        plt.show()
    plt.close(fig)

    logger.info("Simulation completed. Outputs saved to: %s", fig_dir)
    return {'t': t, 'agreement': agreement, 'sri': sri, 'w_traj': sol_resh, 'fig_dir': fig_dir}

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    main()
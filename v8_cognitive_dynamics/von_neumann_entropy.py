# von_neumann_entropy.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os

# Create the folder if it does not exist
os.makedirs("figures", exist_ok=True)

try:
    plt.style.use('seaborn')
except OSError:
    plt.style.use('ggplot')

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

def initial_density_matrix(n):
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    return rho

def von_neumann_entropy(rho, eps=1e-12):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.maximum(eigvals, eps)
    return -np.sum(eigvals * np.log(eigvals))

def spectral_threshold_filter(rho, epsilon):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals_filtered = np.where(eigvals < epsilon, 0.0, eigvals)
    if np.sum(eigvals_filtered) > 1e-8:
        eigvals_filtered /= np.sum(eigvals_filtered)
    rho_filtered = eigvecs @ np.diag(eigvals_filtered) @ eigvecs.conj().T
    return rho_filtered

def trace_distance(rho1, rho2):
    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.abs(eigvals)))

def evolve_state(rho, H, dt):
    U = expm(-1j * H * dt)
    return U @ rho @ U.conj().T

# Parameters
epsilon_0 = 0.5
lambda_ = 1.0
T = 50
n = 4
dt = 0.1

# Initialization
rho = initial_density_matrix(n)
A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
H = A + A.conj().T

# Data storage
epsilons = []
entropies = []
trace_distances = []

# Simulation loop
for t in range(T):
    entropy = von_neumann_entropy(rho)
    epsilon_t = epsilon_0 * np.exp(-lambda_ * entropy)
    rho_filtered = spectral_threshold_filter(rho, epsilon_t)
    dist = trace_distance(rho_filtered, rho)

    epsilons.append(epsilon_t)
    entropies.append(entropy)
    trace_distances.append(dist)

    rho = evolve_state(rho_filtered, H, dt)

# Function to save plots separately
def save_single_plot(y, xlabel, ylabel, title, filename, color, marker):
    plt.figure(figsize=(6,4))
    plt.plot(y, color=color, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=300)
    plt.savefig(f"figures/{filename}.pdf")
    plt.show()

# Save each plot
save_single_plot(epsilons, "Time Step", r"Threshold $\epsilon$", 
                 "Adaptive Threshold Over Time", "epsilon_over_time", "tab:blue", "o")

save_single_plot(entropies, "Time Step", "Von Neumann Entropy", 
                 "Entropy Dynamics", "entropy_over_time", "tab:green", "s")

save_single_plot(trace_distances, "Time Step", "Trace Distance", 
                 "Perceptual Deviation", "trace_distance_over_time", "tab:red", "^")

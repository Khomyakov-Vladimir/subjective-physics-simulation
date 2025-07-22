# compare_entropy_fixed_adaptive.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def initial_density_matrix(n):
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    rho = A @ A.conj().T
    return rho / np.trace(rho)

def von_neumann_entropy(rho, eps=1e-12):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.maximum(eigvals, eps)
    return -np.sum(eigvals * np.log(eigvals))

def spectral_threshold_filter(rho, epsilon):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.where(eigvals < epsilon, 0.0, eigvals)
    eigvals /= np.sum(eigvals)
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

def evolve_state(rho, H, dt):
    U = expm(-1j * H * dt)
    return U @ rho @ U.conj().T

# Parameters
epsilon_0 = 0.5
lambda_ = 1.0
fixed_epsilon = 0.2
T = 50
dt = 0.1
n = 4

# Initialization
rho_adaptive = initial_density_matrix(n)
rho_fixed = np.copy(rho_adaptive)
A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
H = A + A.conj().T

entropy_adaptive = []
entropy_fixed = []

# Simulation loop
for t in range(T):
    e_adapt = von_neumann_entropy(rho_adaptive)
    eps_t = epsilon_0 * np.exp(-lambda_ * e_adapt)
    rho_adaptive = spectral_threshold_filter(rho_adaptive, eps_t)
    entropy_adaptive.append(e_adapt)
    rho_adaptive = evolve_state(rho_adaptive, H, dt)

    e_fixed = von_neumann_entropy(rho_fixed)
    rho_fixed = spectral_threshold_filter(rho_fixed, fixed_epsilon)
    entropy_fixed.append(e_fixed)
    rho_fixed = evolve_state(rho_fixed, H, dt)

# Plotting
plt.figure(figsize=(6,4))
plt.plot(entropy_fixed, label="Fixed Threshold ($\\epsilon=0.2$)", color="orange", marker="o", markersize=8)
plt.plot(entropy_adaptive, label="Adaptive Threshold", color="royalblue", marker="s", markersize=3)
plt.xlabel("Time Step")
plt.ylabel("Entropy")
plt.title("Entropy Dynamics: Adaptive vs Fixed Threshold")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/entropy_comparison.pdf")
plt.savefig("figures/entropy_comparison.png")
plt.show()

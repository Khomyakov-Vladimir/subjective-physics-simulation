# main.py
# v2 adaptive threshold simulation script
# Run the full simulation and generate plots (PDF)
# Zenodo DOI: https://doi.org/10.5281/zenodo.15719389

# Note: See README.md for usage

from observer import ObserverDomain
import numpy as np
import matplotlib.pyplot as plt
import os

# === Параметры ===
M = 1000
D = 20
sigma = 1.0
N = 10000
steps = 20

initial_epsilon = 0.3
eta = 0.1
mode = "norm"  # "entropy" or "norm"

entropy_target = 0.02
norm_target = 1.5

# === Директории ===
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

obs = ObserverDomain()
entropies, epsilons, norms, trace_distances = [], [], [], []

np.random.seed(42)
x = np.cumsum(sigma * np.random.randn(N, M), axis=0)
prev_proj = None
epsilon = initial_epsilon

# === Основной цикл ===
for t in range(steps):
    x_t = x[t * (N // steps):(t + 1) * (N // steps)]
    F_xt = obs.project(x_t, epsilon, D)

    entropy = obs.compute_entropy(F_xt, epsilon)
    norm = np.mean(np.linalg.norm(F_xt, axis=1))
    td = np.mean(np.linalg.norm(F_xt - prev_proj, axis=1)) if prev_proj is not None else 0.0
    prev_proj = F_xt.copy()

    print(f"[{mode.upper()}][Step {t+1:2d}] ε = {epsilon:.4f} | S = {entropy:.4f} | Norm = {norm:.4f} | TD = {td:.4f}")

    epsilons.append(epsilon)
    entropies.append(entropy)
    norms.append(norm)
    trace_distances.append(td)

    # === Адаптация ===
    if mode == "entropy":
        epsilon = obs.adapt_epsilon_by_entropy(epsilon, entropy, entropy_target, eta)
    elif mode == "norm":
        epsilon = obs.adapt_epsilon_by_norm(epsilon, norm, norm_target, eta)

# === Сохранение данных ===
np.savez(f"data/adaptive_metrics_{mode}.npz",
         epsilons=np.array(epsilons),
         entropies=np.array(entropies),
         norms=np.array(norms),
         trace_distances=np.array(trace_distances))

# === Построение графиков ===
steps_range = np.arange(steps)
suffix = f"_{mode}"

def save_plot(data, ylabel, title, filename, marker='o', color=None):
    plt.figure()
    plt.plot(steps_range, data, marker=marker, color=color)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}{suffix}.pdf")
    plt.close()

save_plot(epsilons, "Perceptual Threshold ε", "Adaptive Threshold Over Time", "adaptive_threshold")
save_plot(entropies, "Cognitive Entropy", "Cognitive Entropy Over Time", "adaptive_entropy", marker='s', color='orange')
save_plot(norms, "Norm of Projection", "Norm Dynamics", "adaptive_norm", marker='^', color='green')
save_plot(trace_distances, "Average Trace Distance", "Trace Distance Dynamics", "adaptive_trace_distance", marker='x', color='purple')

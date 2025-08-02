# simulate_entropy_rt_full.py

import numpy as np
from scipy.stats import dirichlet, norm, spearmanr, pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ================================
# ======== CONFIGURATION =========
# ================================

N_TRIALS = 100000  # Number of simulated trials
DIRICHLET_PARAMS = [2, 2]  # Dirichlet distribution parameters
NOISE_STD = 50  # Standard deviation of Gaussian noise
BASE_RT = 200  # Baseline reaction time (ms)
SLOPE = 150  # Entropy scaling factor
SEED = 42  # Random seed for reproducibility
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
OUTPUT_CSV = "data/simulated_data.csv"
FIGURES_DIR = "figures"
# ================================

np.random.seed(SEED)

# Containers for results
entropies = []
reaction_times = []

def compute_entropy(p):
    """Compute Shannon entropy given a probability vector."""
    return -np.sum(p * np.log(p))

# Simulation loop
print(f"Running {N_TRIALS:,} simulations...")
for _ in tqdm(range(N_TRIALS)):
    p_i = dirichlet.rvs(DIRICHLET_PARAMS)[0]
    entropy_i = compute_entropy(p_i)
    noise = norm.rvs(scale=NOISE_STD)
    rt_i = BASE_RT + SLOPE * entropy_i + noise

    entropies.append(entropy_i)
    reaction_times.append(rt_i)

entropies = np.array(entropies)
reaction_times = np.array(reaction_times)

# Save raw data to CSV
df = pd.DataFrame({
    "Entropy": entropies,
    "ReactionTime": reaction_times
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved raw data to {OUTPUT_CSV}")

# Print summary statistics
print("Entropy mean = {:.3f}, std = {:.3f}".format(entropies.mean(), entropies.std()))
print("Reaction Time mean = {:.1f} ms, std = {:.1f} ms".format(reaction_times.mean(), reaction_times.std()))

# Correlation calculations
spearman_corr, spearman_p = spearmanr(entropies, reaction_times)
pearson_corr, _ = pearsonr(entropies, reaction_times)

print("Spearman correlation: {:.3f} (p = {:.1e})".format(spearman_corr, spearman_p))
print("Pearson correlation: {:.3f}".format(pearson_corr))

# Bootstrap confidence intervals for Spearman correlation
print("Bootstrapping Spearman correlation confidence interval...")
boot_corrs = []
for _ in tqdm(range(N_BOOTSTRAP)):
    idx = np.random.choice(len(entropies), len(entropies), replace=True)
    boot_corr, _ = spearmanr(entropies[idx], reaction_times[idx])
    boot_corrs.append(boot_corr)

boot_corrs = np.sort(boot_corrs)
ci_low = np.percentile(boot_corrs, 2.5)
ci_high = np.percentile(boot_corrs, 97.5)
print("Spearman 95% CI: [{:.3f}, {:.3f}]".format(ci_low, ci_high))

# Histogram of Reaction Times
plt.figure(figsize=(12, 4))
plt.hist(reaction_times, bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Frequency")
plt.title("Histogram of Simulated Reaction Times")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/histogram_rt.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/histogram_rt.pdf")

# Scatter plot: Entropy vs Reaction Time
plt.figure(figsize=(6, 5))
plt.scatter(entropies, reaction_times, alpha=0.2, s=1, rasterized=True)
plt.xlabel("Observer Entropy")
plt.ylabel("Reaction Time (ms)")
plt.title("Entropy vs. Reaction Time")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/scatter_entropy_rt.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/scatter_entropy_rt.pdf")

# CDF plot of Reaction Times
plt.figure(figsize=(6, 5))
sorted_rt = np.sort(reaction_times)
cdf = np.arange(1, len(sorted_rt) + 1) / len(sorted_rt)
plt.plot(sorted_rt, cdf, label="Empirical CDF")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Simulated Reaction Times")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/cdf_rt.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/cdf_rt.pdf")

print(f"Saved graphs in both PDF and PNG formats in the '{FIGURES_DIR}' directory: histogram_rt, scatter_entropy_rt, cdf_rt.")
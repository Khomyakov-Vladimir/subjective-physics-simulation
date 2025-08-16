# compare_dirichlet_params.py

import numpy as np
from scipy.stats import dirichlet, norm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ================================
# ========= CONFIGURATION ========
# ================================
N_TRIALS = 5000  # Number of trials per Dirichlet configuration
DIRICHLET_PARAM_LIST = [
    [1, 1],
    [2, 2],
    [5, 5],
    [10, 10]
]
NOISE_STD = 50
BASE_RT = 200
SLOPE = 150
SEED = 42
OUTPUT_DIR = "data"
FIGURES_DIR = "figures"
# ================================

np.random.seed(SEED)

# Container for all results
all_results = []

# Iterate over parameter sets
for params in DIRICHLET_PARAM_LIST:
    print(f"Simulating Dirichlet parameters: {params}")
    entropies = []
    rts = []
    
    for _ in tqdm(range(N_TRIALS)):
        p_i = dirichlet.rvs(params)[0]
        entropy_i = -np.sum(p_i * np.log(p_i))
        noise = norm.rvs(scale=NOISE_STD)
        rt_i = BASE_RT + SLOPE * entropy_i + noise
        
        entropies.append(entropy_i)
        rts.append(rt_i)

    df = pd.DataFrame({
        "Entropy": entropies,
        "ReactionTime": rts,
        "DirichletParams": str(params)
    })
    all_results.append(df)
    
    # Save per-configuration CSV
    csv_path = f"{OUTPUT_DIR}/simulated_data_dirichlet_{params[0]}_{params[1]}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")

# Concatenate all results
full_df = pd.concat(all_results, ignore_index=True)
full_df.to_csv(f"{OUTPUT_DIR}/simulated_data_all_configs.csv", index=False)
print("Saved full dataset.")

# Plot histogram comparison of entropies
plt.figure(figsize=(10,6))
for params in DIRICHLET_PARAM_LIST:
    subset = full_df[full_df["DirichletParams"]==str(params)]
    plt.hist(subset["Entropy"], bins=50, alpha=0.5, label=f"{params}")
plt.xlabel("Observer Entropy")
plt.ylabel("Frequency")
plt.title("Entropy Distribution Across Dirichlet Parameters")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/entropy_hist_comparison.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/entropy_hist_comparison.pdf")
plt.show()

# Plot CDF comparison of reaction times
plt.figure(figsize=(10,6))
for params in DIRICHLET_PARAM_LIST:
    subset = full_df[full_df["DirichletParams"]==str(params)]
    sorted_rt = np.sort(subset["ReactionTime"])
    cdf = np.arange(1, len(sorted_rt)+1)/len(sorted_rt)
    plt.plot(sorted_rt, cdf, label=f"{params}")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Reaction Times Across Dirichlet Parameters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rt_cdf_comparison.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/rt_cdf_comparison.pdf")
plt.show()

# Boxplot of Entropy
plt.figure(figsize=(8,6))
full_df["Dirichlet Label"] = full_df["DirichletParams"].astype(str)
full_df.boxplot(column="Entropy", by="Dirichlet Label")
plt.xlabel("Dirichlet Parameters")
plt.ylabel("Entropy")
plt.title("Entropy Distribution per Dirichlet Config")
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/entropy_boxplot.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/entropy_boxplot.pdf")
plt.show()

# Boxplot of Reaction Time
plt.figure(figsize=(8,6))
full_df.boxplot(column="ReactionTime", by="Dirichlet Label")
plt.xlabel("Dirichlet Parameters")
plt.ylabel("Reaction Time (ms)")
plt.title("Reaction Time Distribution per Dirichlet Config")
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rt_boxplot.png", dpi=300)
plt.savefig(f"{FIGURES_DIR}/rt_boxplot.pdf")
plt.show()

print("All comparison plots saved.")

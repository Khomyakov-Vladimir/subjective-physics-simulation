import numpy as np
import matplotlib.pyplot as plt
import os

# Directories
data_dir = os.path.join("data")
figures_dir = os.path.join("figures")
os.makedirs(figures_dir, exist_ok=True)

# Load data
entropy_data = np.load(os.path.join(data_dir, "adaptive_metrics_entropy.npz"))
norm_data = np.load(os.path.join(data_dir, "adaptive_metrics_norm.npz"))

# Adaptation steps
steps = np.arange(1, len(entropy_data['epsilons']) + 1)

# Entropy-based feedback
eps_entropy = entropy_data['epsilons']
S_entropy = entropy_data['entropies']

# Norm-based feedback
eps_norm = norm_data['epsilons']
S_norm = norm_data['entropies']

# Create figure
fig, ax1 = plt.subplots(figsize=(10,6))

# Colors
color_entropy = 'tab:blue'
color_norm = 'tab:red'

# First Y axis: Perceptual threshold ε(t)
ax1.set_xlabel('Adaptation Step')
ax1.set_ylabel('Perceptual Threshold ε(t)', color='gray')

lns1 = ax1.plot(steps, eps_entropy, color=color_entropy, linestyle='--', marker='o', label='ε (Entropy Feedback)')
lns2 = ax1.plot(steps, eps_norm, color=color_norm, linestyle='--', marker='s', label='ε (Norm Feedback)')

ax1.tick_params(axis='y', labelcolor='gray')
ax1.set_ylim(0, max(np.max(eps_entropy), np.max(eps_norm)) * 1.2)

# Second Y axis: Cognitive entropy S(ε)
ax2 = ax1.twinx()
ax2.set_ylabel('Cognitive Entropy S(ε)', color='black')

lns3 = ax2.plot(steps, S_entropy, color=color_entropy, linestyle='-', marker='o', label='S (Entropy Feedback)')
lns4 = ax2.plot(steps, S_norm, color=color_norm, linestyle='-', marker='s', label='S (Norm Feedback)')

ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0, max(np.max(S_entropy), np.max(S_norm)) * 1.2)

# Title
plt.title('Adaptive Perceptual Dynamics (Simulation Data)')

# Legend
lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='center right')

# Grid
ax1.grid(True, linestyle='--', alpha=0.5)

# Save figure
output_path = os.path.join(figures_dir, "adaptive_perceptual_dynamics.pdf")
plt.tight_layout()
plt.savefig(output_path)

# Print confirmation
print(f"The graph has been successfully saved to the file: {output_path}")

# Show plot
plt.show()

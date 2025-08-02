import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.38e-23  # Boltzmann constant
T = 300         # Temperature (Kelvin)
ln2 = np.log(2)

# Define epsilon values
epsilon_values = np.logspace(-4, -0.1, 50)

# Compute S_prior and S_posterior
S_prior = np.log(1 + 1e4)  # Fixed fine-grained entropy
S_posterior = np.log(1 + 1 / epsilon_values)

# Delta S_cog
delta_S = S_prior - S_posterior

# Energetic cost per Landauer
E_min = k_B * T * ln2 * delta_S

# --- Plot Delta S ---
plt.figure(figsize=(7,5))
plt.plot(epsilon_values, delta_S, 'o-', color='blue')
plt.xscale('log')
plt.xlabel('Perceptual threshold ε')
plt.ylabel('ΔS_cog')
plt.title('Cognitive Entropy Reduction vs ε')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('figures/deltaS_vs_epsilon.pdf')
plt.close()

# --- Plot Energetic Cost ---
plt.figure(figsize=(7,5))
plt.plot(epsilon_values, E_min, 'o-', color='red')
plt.xscale('log')
plt.xlabel('Perceptual threshold ε')
plt.ylabel('E_min (Joules)')
plt.title('Minimal Energetic Cost per Landauer')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('figures/energy_vs_epsilon.pdf')
plt.close()

print("Calculation complete. Figures saved in 'figures/'.")

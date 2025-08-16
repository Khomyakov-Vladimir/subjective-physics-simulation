import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# Create output folders if they do not exist
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Constants
kB = 1.38e-23  # Boltzmann constant [J/K]
T = 300        # Temperature [K]

# Lambdas to compare
lambda_list = [1e19, 1e20, 1e21]
colors = ['green', 'purple', 'orange']

# Functions
def S(eps):
    """Cognitive entropy gain."""
    return np.log(1 + 1/eps)

def E_disc(eps):
    """Energetic dissipation cost (Landauer's bound)."""
    return kB * T * np.log(2) * np.log2(1/eps)

# Epsilon range
epsilon = np.logspace(-4, -0.1, 300)

# Precompute S and E_disc
S_vals = S(epsilon)
E_vals = E_disc(epsilon)

# Save comparison CSV
with open('data/tradeoff_data_lambda_comparison.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['lambda', 'epsilon', 'S(epsilon)', 'E_disc', 'lambda*E_disc', 'L(epsilon)'])
    for lam in lambda_list:
        L_vals = S_vals - lam * E_vals
        for eps, s, e, l in zip(epsilon, S_vals, E_vals, L_vals):
            writer.writerow([lam, eps, s, e, lam * e, l])

print("CSV file 'data/tradeoff_data_lambda_comparison.csv' saved.")

# Also save separate CSV for lambda=1e20
lambda_ref = 1e20
L_ref = S_vals - lambda_ref * E_vals

with open('data/tradeoff_data_lambda1e20.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epsilon', 'S(epsilon)', 'E_disc', 'lambda*E_disc', 'L(epsilon)'])
    for eps, s, e, l in zip(epsilon, S_vals, E_vals, L_ref):
        writer.writerow([eps, s, e, lambda_ref * e, l])

print("CSV file 'data/tradeoff_data_lambda1e20.csv' saved.")

# Plot comparison of L(epsilon) for all lambda
plt.figure(figsize=(10, 7))
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)

for lam, col in zip(lambda_list, colors):
    L_vals = S_vals - lam * E_vals
    idx_max = np.argmax(L_vals)
    eps_star = epsilon[idx_max]
    L_star = L_vals[idx_max]
    
    plt.plot(epsilon, L_vals, color=col, lw=2, linestyle='-', label=rf'$\lambda={lam:.0e}$')
    plt.scatter(eps_star, L_star, color=col, s=60, zorder=5)
    
    # Different annotation offsets per lambda
    # Here you can adjust the xytext:
    if lam == 1e19:
        # Green line: shift below (Y * 0.5)
        xytext = (eps_star * 1.2, L_star - abs(L_star)*0.9)
    elif lam == 1e20:
        # Purple line: shift below (Y * 0.5)
        xytext = (eps_star * 1.2, L_star - abs(L_star)*1.4)
    elif lam == 1e21:
        # Orange line: shift left (X * 0.5) and below (Y * 0.5)
        xytext = (eps_star * 0.1, L_star - abs(L_star)*7.4)

    plt.annotate(
        rf'$\epsilon^*={eps_star:.2e}$',
        xy=(eps_star, L_star),
        xytext=xytext,
        arrowprops=dict(arrowstyle='->', lw=1, color=col),
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
    )

plt.xlabel(r'Perceptual threshold $\epsilon$', fontsize=13)
plt.ylabel(r'Trade-off functional $\mathcal{L}(\epsilon)$', fontsize=13)
plt.title(r'Trade-off functional $\mathcal{L}(\epsilon)$ for different $\lambda$', fontsize=14)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()

# Save the figure for inclusion in the article
plt.savefig('figures/L_epsilon_lambda_comparison_plot.pdf')
print("Plot saved to 'figures/L_epsilon_lambda_comparison_plot.pdf'.")

plt.show()

print("\n=== Sample values for lambda=1e21 ===")
lambda_check = 1e21
for i, (eps, s, e) in enumerate(zip(epsilon, S_vals, E_vals)):
    if i % 30 == 0:
        L = s - lambda_check * e
        print(f"epsilon = {eps:.2e}\tS = {s:.3f}\tE_disc = {e:.3e}\tL = {L:.3e}")

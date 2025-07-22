# geodesic_dynamics_cognitive_action.py
# Version: 1.0 (canonical)
# This is the canonical implementation of cognitive geodesic dynamics
# as per the Subjective Physics Hypothesis.

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# --- Create output directories ---

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Cognitive Metric Tensor g_ij(δ) ---

def cognitive_metric_tensor(delta):
    """
    Defines the cognitive metric tensor g_ij as a symmetric 2x2 matrix
    dependent on the current state delta = (x, y).
    This metric encodes anisotropy in cognitive distinguishability.
    """
    x, y = delta
    kappa = 0.8    # curvature intensity
    alpha = 0.3    # off-diagonal coupling
    return np.array([
        [1.0, alpha * x * y],
        [alpha * x * y, 1.0 + kappa * x**2]
    ])

def inverse_metric_tensor(g):
    """Returns the inverse of the metric tensor g_ij."""
    return np.linalg.inv(g)

def partial_metric(delta, i, j, k, h=1e-5):
    """
    Numerically approximates the partial derivative ∂g_{jk}/∂x^i using central differences.
    """
    d1, d2 = delta.copy(), delta.copy()
    d1[i] += h
    d2[i] -= h
    g1 = cognitive_metric_tensor(d1)[j, k]
    g2 = cognitive_metric_tensor(d2)[j, k]
    return (g1 - g2) / (2 * h)

def christoffel_symbols(delta):
    """
    Computes Christoffel symbols Γ^i_{jk} for the cognitive metric using the standard formula:
        Γ^i_{jk} = 1/2 * g^{il}(∂g_{lj}/∂x^k + ∂g_{lk}/∂x^j − ∂g_{jk}/∂x^l)
    """
    dim = 2
    Gamma = np.zeros((dim, dim, dim))
    g = cognitive_metric_tensor(delta)
    g_inv = inverse_metric_tensor(g)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                sum_term = 0.0
                for l in range(dim):
                    term = (
                        partial_metric(delta, j, l, k) +
                        partial_metric(delta, k, l, j) -
                        partial_metric(delta, l, j, k)
                    )
                    sum_term += g_inv[i, l] * term
                Gamma[i, j, k] = 0.5 * sum_term

    return Gamma

# --- Cognitive Discrimination Potential V(δ) ---

def potential_V(delta):
    """
    Defines the scalar potential representing the cost of cognitive discrimination.
    """
    x, y = delta
    return 0.5 * (x**2 + 2 * y**2)

def grad_potential(delta, h=1e-5):
    """
    Computes the numerical gradient ∇V of the potential V(δ).
    """
    grad = np.zeros_like(delta)
    for i in range(len(delta)):
        d1, d2 = delta.copy(), delta.copy()
        d1[i] += h
        d2[i] -= h
        grad[i] = (potential_V(d1) - potential_V(d2)) / (2 * h)
    return grad

# --- Geodesic Simulation and Cognitive Action Computation ---

def simulate_geodesic_with_action(delta0, v0, dt=0.05, steps=100):
    """
    Simulates the geodesic trajectory in the cognitive space under a curved metric,
    with dynamical computation of the accumulated action S(t) along the path.

    Returns:
        trajectory: array of δ(t)
        action: scalar value of the final accumulated action
        data: list of dynamic quantities at each step (δ, v, L, S, etc.)
    """
    delta = delta0.copy()
    velocity = v0.copy()
    trajectory = [delta.copy()]
    action = 0.0
    data = []

    for _ in range(steps):
        Gamma = christoffel_symbols(delta)
        acc = np.zeros_like(delta)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acc[i] -= Gamma[i, j, k] * velocity[j] * velocity[k]

        acc -= grad_potential(delta)  # External potential force

        # Update dynamics (Euler integration)
        velocity += acc * dt
        delta += velocity * dt
        trajectory.append(delta.copy())

        # Compute energies and Lagrangian
        g = cognitive_metric_tensor(delta)
        kinetic = 0.5 * velocity @ g @ velocity
        potential = potential_V(delta)
        lagrangian = kinetic + potential
        action += lagrangian * dt

        data.append({
            "x": delta[0],
            "y": delta[1],
            "vx": velocity[0],
            "vy": velocity[1],
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "lagrangian": lagrangian,
            "action": action
        })

    return np.array(trajectory), action, data

# --- Initial Conditions and Geodesic Paths ---

initial_position = np.array([0.0, 0.0])  # Origin of cognitive distinction
initial_velocities = [
    np.array([1.0, 0.0]),
    np.array([0.8, 0.6]),
    np.array([0.5, 1.0])
]

trajectories = []

for i, v in enumerate(initial_velocities):
    traj, action, data = simulate_geodesic_with_action(
        initial_position, v, dt=0.05, steps=120
    )
    trajectories.append({
        "label": f"Path {i+1}",
        "trajectory": traj,
        "action": action,
        "dynamics": data
    })

# --- Plotting Cognitive Action S(t) over Time ---

plt.figure(figsize=(10, 6))

def linear_model(t, a, b):
    return a * t + b

fitted_lines = []

for path in trajectories:
    S_vals = [frame["action"] for frame in path["dynamics"]]
    t_vals = np.linspace(0, len(S_vals) * 0.05, len(S_vals))  # dt = 0.05

    popt, _ = curve_fit(linear_model, t_vals, S_vals)
    fitted_lines.append((path["label"], *popt))

    plt.plot(t_vals, S_vals, label=f'{path["label"]}, $S(t)$', linewidth=2)
    plt.plot(
        t_vals, linear_model(t_vals, *popt), '--',
        label=f'{path["label"]} fit: $S(t) ≈ {popt[0]:.3f}t + {popt[1]:.3f}$'
    )

plt.title("Cognitive Action $S(t)$ along Geodesic Trajectories", fontsize=14)
plt.xlabel("Time $t$ (arb. units)", fontsize=12)
plt.ylabel("Cognitive Action $S(t)$", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Save Action Plot to PDF ---

with PdfPages("figures/cognitive_action_vs_time.pdf") as pdf:
    pdf.savefig()
    # plt.close()

plt.show()

# --- Export Geodesic Dynamics to JSON ---

# Convert trajectory arrays to list format for JSON serialization
for traj in trajectories:
    traj["trajectory"] = traj["trajectory"].tolist()

with open("data/geodesic_paths_dynamics.json", "w") as jsonfile:
    json.dump(trajectories, jsonfile, indent=2)

# --- Print Linear Fit Results ---

print("Fitted linear parameters for S(t):")
for label, slope, intercept in fitted_lines:
    print(f"{label}: slope = {slope:.4f}, intercept = {intercept:.4f}")

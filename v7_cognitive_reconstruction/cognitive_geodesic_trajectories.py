# cognitive_geodesic_trajectories.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import json
import os

# --- Create output folders for figures and data ---

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# --- Cognitive Metric Tensor g_ij(δ) ---

def cognitive_metric_tensor(delta):
    """
    Returns the cognitive metric tensor g_ij for a given state delta = (x, y).
    The metric encodes anisotropic compatibility between distinctions.
    """
    x, y = delta
    kappa = 0.8   # curvature strength
    alpha = 0.3   # coupling parameter
    return np.array([
        [1.0, alpha * x * y],
        [alpha * x * y, 1.0 + kappa * x ** 2]
    ])

def inverse_metric_tensor(g):
    """Returns the inverse of a 2x2 metric tensor."""
    return np.linalg.inv(g)

def partial_metric(delta, i, j, k, h=1e-5):
    """
    Computes the numerical partial derivative ∂g_{jk}/∂x^i using central differences.
    """
    d1 = delta.copy()
    d2 = delta.copy()
    d1[i] += h
    d2[i] -= h
    g1 = cognitive_metric_tensor(d1)[j, k]
    g2 = cognitive_metric_tensor(d2)[j, k]
    return (g1 - g2) / (2 * h)

def christoffel_symbols(delta):
    """
    Computes the Christoffel symbols Γ^i_{jk} at a given state delta.
    These define how the metric structure affects geodesic motion.
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
    Defines the scalar potential representing attraction toward more stable cognitive states.
    """
    x, y = delta
    return 0.5 * (x**2 + 2 * y**2)

def grad_potential(delta, h=1e-5):
    """
    Computes the numerical gradient ∇V(δ) using central differences.
    """
    grad = np.zeros_like(delta)
    for i in range(len(delta)):
        d1 = delta.copy()
        d2 = delta.copy()
        d1[i] += h
        d2[i] -= h
        grad[i] = (potential_V(d1) - potential_V(d2)) / (2 * h)
    return grad

# --- Geodesic Simulation with Action Accumulation ---

def simulate_geodesic_with_action(delta0, v0, dt=0.05, steps=100):
    """
    Simulates geodesic motion under the cognitive metric and potential,
    computes the cumulative action along the path.

    Returns:
        trajectory: array of state vectors δ(t)
        action: total accumulated action
        dynamics: list of time-evolution data (δ, v, a, energy, etc.)
    """
    trajectory = [delta0.copy()]
    velocity = v0.copy()
    delta = delta0.copy()
    action = 0.0
    dynamics = []

    for step in range(steps):
        # Compute acceleration using geodesic equations
        Gamma = christoffel_symbols(delta)
        acc = np.zeros_like(delta)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acc[i] -= Gamma[i, j, k] * velocity[j] * velocity[k]

        # Add external force from potential
        acc -= grad_potential(delta)

        # Update state using Euler integration
        velocity += acc * dt
        delta += velocity * dt
        trajectory.append(delta.copy())

        # Compute cognitive action
        g = cognitive_metric_tensor(delta)
        kinetic = 0.5 * velocity @ g @ velocity
        potential = potential_V(delta)
        action += (kinetic + potential) * dt

        # Store system state
        dynamics.append({
            'step': step,
            'position': {'x': float(delta[0]), 'y': float(delta[1])},
            'velocity': {'x': float(velocity[0]), 'y': float(velocity[1])},
            'acceleration': {'x': float(acc[0]), 'y': float(acc[1])},
            'kinetic': float(kinetic),
            'potential': float(potential),
            'lagrangian': float(kinetic + potential),
            'action': float(action)
        })

    return np.array(trajectory), action, dynamics

# --- Initial Conditions ---

delta_start = np.array([0.0, 0.0])
velocities = [
    np.array([1.0, 0.0]),   # Path 1: direct trajectory
    np.array([0.8, 0.6]),   # Path 2: angled projection
    np.array([0.5, 1.0])    # Path 3: vertical distinction
]

# --- Run Simulations ---

trajectories = []
trajectories_data = []

for i, v in enumerate(velocities):
    print(f"Simulating path {i+1}...")
    traj, action, dynamics = simulate_geodesic_with_action(delta_start, v, dt=0.05, steps=120)
    trajectories.append(traj)
    trajectories_data.append({
        'path_id': i+1,
        'label': ['Direct trajectory', 'Angular projection', 'Vertical distinction'][i],
        'trajectory': traj.tolist(),
        'total_action': float(action),
        'dynamics': dynamics
    })

# --- Visualization of Trajectories ---

plt.figure(figsize=(8, 7.5))
colors = ['#1f77b4', '#2ca02c', '#d62728']

for i, (traj, td) in enumerate(zip(trajectories, trajectories_data)):
    plt.plot(traj[:, 0], traj[:, 1], 'o-', 
             markersize=4,
             linewidth=1.5,
             color=colors[i],
             label=f"{td['label']}, S={td['total_action']:.2f}")

plt.plot(*delta_start, 'ko', markersize=8, label='Initial point (δ₀)')
plt.title("Cognitive Distinction Trajectories in Metric Space\nwith Potential V(δ) and Action S", pad=15)
plt.xlabel("Cognitive coordinate δₓ", fontsize=12)
plt.ylabel("Cognitive coordinate δᵧ", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend(fontsize=10, framealpha=1, loc='upper left')
plt.tight_layout()

with PdfPages('figures/cognitive_trajectories.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
#   plt.close()

plt.show()

# --- Export Data ---

print("Exporting data...")

# CSV export for Path 1
csv_fields = ['step', 'position_x', 'position_y', 
              'velocity_x', 'velocity_y', 
              'acceleration_x', 'acceleration_y',
              'kinetic', 'potential', 'lagrangian', 'action']

with open("data/path1_dynamics.csv", "w", newline="", encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    
    for step_data in trajectories_data[0]['dynamics']:
        row = {
            'step': step_data['step'],
            'position_x': step_data['position']['x'],
            'position_y': step_data['position']['y'],
            'velocity_x': step_data['velocity']['x'],
            'velocity_y': step_data['velocity']['y'],
            'acceleration_x': step_data['acceleration']['x'],
            'acceleration_y': step_data['acceleration']['y'],
            'kinetic': step_data['kinetic'],
            'potential': step_data['potential'],
            'lagrangian': step_data['lagrangian'],
            'action': step_data['action']
        }
        writer.writerow(row)

# JSON export for all paths
with open("data/all_paths_dynamics.json", "w", encoding='utf-8') as jsonfile:
    json.dump(trajectories_data, jsonfile, indent=2, ensure_ascii=False)

print("Analysis complete. Results saved in 'figures' and 'data' folders.")

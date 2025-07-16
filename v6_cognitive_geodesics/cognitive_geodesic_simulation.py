# cognitive_geodesic_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Cognitive Space and Subjective Metric ---

def scalar_product(delta1, delta2):
    """
    Scalar product representing cognitive compatibility between two distinction vectors.
    A higher value indicates greater alignment in the space of differentiations.
    """
    return np.dot(delta1, delta2) / (np.linalg.norm(delta1) * np.linalg.norm(delta2) + 1e-10)

def cognitive_distance(delta1, delta2):
    """
    Cognitive distance as a measure of mismatch between two distinction vectors.
    Defined as 1 minus the normalized scalar product.
    """
    return 1 - scalar_product(delta1, delta2)

# --- Numerical Simulation of a Cognitive Geodesic ---

def simulate_geodesic(delta_start, delta_end, steps=100, learning_rate=0.05):
    """
    Simulates the trajectory of a cognitive geodesic between two distinction states
    using a basic gradient descent procedure.

    Parameters:
        delta_start (np.ndarray): Initial cognitive distinction vector (e.g., pre-measurement).
        delta_end (np.ndarray): Target cognitive distinction vector (e.g., post-measurement).
        steps (int): Number of simulation steps along the geodesic path.
        learning_rate (float): Step size in the gradient descent.

    Returns:
        np.ndarray: Sequence of vectors approximating the geodesic path.
    """
    trajectory = [delta_start]
    delta = delta_start.copy()

    for _ in range(steps):
        grad = (delta - delta_end)  # Gradient pointing towards the target
        delta = delta - learning_rate * grad
        trajectory.append(delta.copy())

    return np.array(trajectory)

# --- Initial and Final Distinction Vectors in 2D ---

delta_initial = np.array([1.0, 0.0])      # Before weak measurement
delta_final   = np.array([0.7, 0.7])      # After weak measurement

# --- Generate Geodesic Trajectory ---

trajectory = simulate_geodesic(delta_initial, delta_final, steps=50)

# --- Visualization of the Cognitive Geodesic Trajectory ---

plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Cognitive geodesic trajectory')
plt.plot(*delta_initial, 'go', label='Initial distinction $\\delta_{\\text{in}}$')
plt.plot(*delta_final, 'ro', label='Final distinction $\\delta_{\\text{post}}$')
plt.title("Cognitive Geodesic in the Distinction Space $\\mathcal{D}$")
plt.xlabel("Distinction component $\\delta_x$")
plt.ylabel("Distinction component $\\delta_y$")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()

# --- Save the Plot to PDF ---

with PdfPages('figures/cognitive_geodesic.pdf') as pdf:
    pdf.savefig()  # Save the current figure
    # plt.close()   # Optional: close figure after saving

# --- Display the Plot ---

plt.show()

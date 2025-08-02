# observer.py
# Simulation logic for cognitive projection observer
# Zenodo DOI: https://doi.org/10.5281/zenodo.15719389

import numpy as np

class ObserverDomain:
    """
    Cognitive projection observer with adaptive threshold ε(t).
    """

    def __init__(self, lambda_c=0.8, noise_level=0.1):
        self.lambda_c = lambda_c
        self.noise_level = noise_level

    def project(self, x, epsilon, D):
        """
        Apply projection F_ε: block-wise averaging and state grouping within ε.
        """
        N, M = x.shape
        block_size = M // D
        projected = np.zeros((N, D))

        for i in range(D):
            block = x[:, i * block_size:(i + 1) * block_size]
            avg = np.mean(block, axis=1)
            projected[:, i] = avg

        # Group similar states
        representative = []
        assigned = np.full(N, -1)

        for i in range(N):
            if assigned[i] >= 0:
                continue
            assigned[i] = len(representative)
            rep = projected[i]
            for j in range(i + 1, N):
                if assigned[j] == -1 and np.linalg.norm(rep - projected[j]) < epsilon:
                    assigned[j] = assigned[i]
            representative.append(rep)

        return np.array([representative[idx] for idx in assigned])

    def compute_entropy(self, projected, epsilon):
        """
        Estimate number of distinguishable classes and compute entropy.
        """
        N = projected.shape[0]
        classes = []
        assigned = np.full(N, -1)

        for i in range(N):
            if assigned[i] >= 0:
                continue
            assigned[i] = len(classes)
            ref = projected[i]
            for j in range(i + 1, N):
                if assigned[j] == -1 and np.linalg.norm(ref - projected[j]) < epsilon:
                    assigned[j] = assigned[i]
            classes.append(ref)

        C = len(classes)
        return np.log(C) if C > 0 else 0.0

    def adapt_epsilon_by_entropy(self, epsilon, entropy, entropy_target=0.02, eta=0.05):
        """
        Adapt ε based on entropy feedback.
        """
        delta = entropy_target - entropy
        return max(epsilon + eta * delta, 1e-5)

    def adapt_epsilon_by_norm(self, epsilon, norm, norm_target=1.5, eta=0.05):
        """
        Adapt ε based on projection norm feedback.
        """
        delta = norm - norm_target
        return max(epsilon + eta * delta, 1e-5)

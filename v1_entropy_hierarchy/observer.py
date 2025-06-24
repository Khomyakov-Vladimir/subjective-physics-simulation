# observer.py
# Simulation logic for cognitive projection observer
# Zenodo DOI: https://doi.org/10.5281/zenodo.15719389

import numpy as np

class CognitiveObserver:
    def __init__(self, epsilon=0.1, num_classes=2, decoherence_type=None, decoherence_gamma=0.0):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.decoherence_type = decoherence_type
        self.decoherence_gamma = decoherence_gamma

    def observe(self, state_label):
        d = self.num_classes
        state = np.full(d, self.epsilon)
        state[state_label] = 1.0
        state /= np.linalg.norm(state)

        if self.decoherence_type and self.decoherence_gamma > 0:
            state = self.apply_decoherence(state)

        return state

    def apply_decoherence(self, state):
        if self.decoherence_type == 'phase':
            return self._apply_phase_damping(state)
        return state

    def _apply_phase_damping(self, state):
        gamma = min(self.decoherence_gamma, 1.0)
        new_state = state.copy()
        new_state[1:] *= np.sqrt(1 - gamma)
        new_state /= np.linalg.norm(new_state)
        return new_state

    def entropy(self, a, b):
        rho = 0.5 * (np.outer(a, a) + np.outer(b, b))
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-12]
        return -np.sum(eigvals * np.log2(eigvals))

    def trace_distance(self, a, b):
        delta = np.outer(a, a) - np.outer(b, b)
        eigvals = np.linalg.eigvalsh(delta)
        return 0.5 * np.sum(np.abs(eigvals))

class HierarchicalObserver:
    def __init__(self, levels=3, base_epsilon=0.1, epsilon_decay=0.7, decoherence_type=None, decoherence_gamma=0.0, num_classes=2):
        self.observers = [
            CognitiveObserver(
                epsilon=base_epsilon * (epsilon_decay ** l),
                num_classes=num_classes,
                decoherence_type=decoherence_type,
                decoherence_gamma=decoherence_gamma * (1.2 ** l)
            )
            for l in range(levels)
        ]

    def observe(self, state_label):
        current_state = state_label
        metrics = []

        for obs in self.observers:
            if isinstance(current_state, int):
                vec_a = obs.observe(current_state)
            else:
                vec_a = current_state / np.linalg.norm(current_state)

            vec_b = obs.observe(np.argmax(np.abs(vec_a)))
            metrics.append({
                'entropy': obs.entropy(vec_a, vec_b),
                'trace_dist': obs.trace_distance(vec_a, vec_b),
                'state_vector': vec_b
            })
            current_state = vec_b

        return current_state, None, metrics

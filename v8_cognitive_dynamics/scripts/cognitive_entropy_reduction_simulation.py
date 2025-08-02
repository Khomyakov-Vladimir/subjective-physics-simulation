#!/usr/bin/env python3
"""
# scripts/cognitive_entropy_reduction_simulation.py

Cognitive Entropy Reduction Simulation (Refactored Russian Version)

Simulates ΔH = H(A) - H(A|B), where:
- H(A) is entropy before intervention
- H(A|B) is conditional entropy after cognitive intervention

Author: Vladimir Khomyakov
Date: July 2025
Version: 2.4
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/subjective-physics-simulation
Citation: DOI:10.5281/zenodo.15719389
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Tuple, List
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveEntropySimulator:
    def __init__(self, resolution: int = 100, p_range: Tuple[float, float] = (0.01, 0.99)):
        self.resolution = resolution
        self.p_min, self.p_max = p_range
        self.p_values = np.linspace(self.p_min, self.p_max, self.resolution)
        self.delta_h_values = []

    @staticmethod
    def shannon_entropy(p: float) -> float:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p) if 0 < p < 1 else 0.0

    def intervention_function(self, p1: float) -> float:
        # Интервенция усиливает уверенность
        return self.p_min + (self.p_max - self.p_min) * p1 ** 2

    def calculate_entropy_reduction(self) -> List[float]:
        self.delta_h_values = []
        for p1 in self.p_values:
            q1 = self.intervention_function(p1)
            h_a = self.shannon_entropy(p1)
            h_ab = self.shannon_entropy(q1)
            self.delta_h_values.append(h_a - h_ab)
        return self.delta_h_values

    def create_visualization(self, figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.p_values, self.delta_h_values,
                label=r"$\Delta H = H(A) - H(A|B)$", color='blue')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel(r"$p_1$ (Prior Probability of $A_1$)", fontsize=12)
        ax.set_ylabel(r"$\Delta H$ (Entropy Reduction)", fontsize=12)
        ax.set_title("Cognitive Reconstruction: Entropy Reduction After Intervention", fontsize=13)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        return fig

    def save_results(self, fig: plt.Figure,
                     output_dir: str = "figures",
                     filename: str = "cognitive_entropy_reduction_simulation.pdf") -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir
        figures_dir = os.path.join(repo_root, output_dir)
        os.makedirs(figures_dir, exist_ok=True)
        pdf_path = os.path.join(figures_dir, filename)

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        logger.info(f"Saved PDF to: {pdf_path}")

    def run_simulation(self, save_output: bool = True, show_plot: bool = True
                       ) -> Tuple[List[float], List[float]]:
        self.calculate_entropy_reduction()
        fig = self.create_visualization()

        if save_output:
            self.save_results(fig)

        if show_plot:
            plt.show()

        plt.close(fig)
        return self.p_values.tolist(), self.delta_h_values


def main():
    simulator = CognitiveEntropySimulator()
    p_vals, delta_vals = simulator.run_simulation()

    print("\n=== Simulation Summary ===")
    print(f"Max ΔH:   {max(delta_vals):.4f} bits")
    print(f"Mean ΔH:  {np.mean(delta_vals):.4f} bits")
    print(f"Points:   {len(p_vals)}")


if __name__ == "__main__":
    main()

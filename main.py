import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from observer import HierarchicalObserver

def main():
    levels = 5
    base_epsilons = [0.4, 0.2, 0.1, 0.05]
    num_runs = 20
    state_label = 0
    num_classes = 4

    output_dir = Path("./figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    entropy_vs_eps = []
    trace_vs_eps = []
    norm_time_data = {eps: np.zeros(levels) for eps in base_epsilons}

    for base_eps in base_epsilons:
        entropy_list = []
        trace_list = []
        norm_per_level_sum = np.zeros(levels)

        for _ in range(num_runs):
            ho = HierarchicalObserver(
                levels=levels,
                base_epsilon=base_eps,
                epsilon_decay=0.7,
                decoherence_type='phase',
                decoherence_gamma=0.1,
                num_classes=num_classes
            )
            _, _, metrics = ho.observe(state_label)

            for i, m in enumerate(metrics):
                entropy_list.append(m['entropy'])
                trace_list.append(m['trace_dist'])
                norm_per_level_sum[i] += np.linalg.norm(m['state_vector'])

        entropy_vs_eps.append(np.mean(entropy_list))
        trace_vs_eps.append(np.mean(trace_list))
        norm_time_data[base_eps] = norm_per_level_sum / num_runs

    # Plot 1
    plt.figure(figsize=(7, 5))
    for eps in base_epsilons:
        plt.plot(range(levels), norm_time_data[eps], marker='o', label=f'ε = {eps}')
    plt.xlabel("Time step (hierarchy level)")
    plt.ylabel("Norm |F_ε(xₜ)|")
    plt.title("Norm of Cognitive Projection over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "norm_vs_time.pdf")
    plt.close()

    # Plot 2
    plt.figure(figsize=(6, 4))
    plt.plot(base_epsilons, entropy_vs_eps, 'o-')
    plt.xlabel("Perceptual threshold ε")
    plt.ylabel("Cognitive entropy S(ε)")
    plt.title("Entropy vs Perceptual Threshold")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_vs_epsilon.pdf")
    plt.close()

    # Plot 3
    plt.figure(figsize=(6, 4))
    plt.plot(base_epsilons, trace_vs_eps, 's-')
    plt.xlabel("Perceptual threshold ε")
    plt.ylabel("Average trace distance")
    plt.title("Trace Distance vs Perceptual Threshold")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "trace_distance_vs_epsilon.pdf")
    plt.close()

    print(f"Plots saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()

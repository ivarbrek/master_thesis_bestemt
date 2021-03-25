import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def plot_alns_history(solution_costs: List[Tuple[int, int]]) -> None:
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    x, y = zip(*solution_costs)
    plt.scatter(x, y, s=7, alpha=0.4, c='black')
    # plt.yscale('log')
    plt.show()


def plot_operator_weights(operator_scores: Dict[str, List[float]]):
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    legend = []
    for operator, scores in operator_scores.items():
        plt.plot(scores)
        legend.append(operator)
    plt.legend(legend)
    plt.show()

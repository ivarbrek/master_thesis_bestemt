import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_alns_history(solution_costs: List[Tuple[int, int]]) -> None:
    x, y = zip(*solution_costs)
    plt.scatter(x, y, s=7, alpha=0.4, c='black')
    plt.show()

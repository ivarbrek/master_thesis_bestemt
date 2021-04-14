import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd

def plot_alns_history(solution_costs: List[Tuple[int, int]],
                      feasible_solutions: List[Tuple[int, bool]] = None) -> None:
    color_map = {True: 'black', False: 'red'}
    if feasible_solutions:
        colors = [color_map[is_feasible] for iter, is_feasible in feasible_solutions]
    else:
        colors = 'black'
    x, y = zip(*solution_costs)
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    plt.scatter(x, y, s=7, alpha=0.4, c=colors)
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


def plot_alns_history_with_production_feasibility(solution_costs: List[Tuple[int, int]],
                                                  production_feasibility: List[bool]) -> None:

    df = pd.DataFrame(dict(iter=[elem[0] for elem in solution_costs],
                           cost=[elem[1] for elem in solution_costs],
                           feasible=production_feasibility))

    fig, ax = plt.subplots()
    colors = {False: 'red', True: 'green'}

    ax.scatter(df['iter'], df['cost'], c=df['feasible'].apply(lambda x: colors[x]))
    plt.show()

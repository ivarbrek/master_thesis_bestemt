import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
from src.alns.solution import ProblemDataExtended
from src.util.plot import plot_operator_weights


def save_weights_stats(prbl: ProblemDataExtended, stat_destroy_weights: Dict[str, List[float]],
                       stat_repair_weights: Dict[str, List[float]], stat_noise_weights: Dict[str, List[float]]) -> None:

    stat_weights = {'destroy': stat_destroy_weights, 'repair': stat_repair_weights, 'noise': stat_noise_weights}

    dir_path = "data/output_stats/"

    # Find unique file name
    i = 0
    filename = f"weight_stats_{len(prbl.vessels)}-{len(prbl.factory_nodes)}-{len(prbl.order_nodes)}-{i}.json"
    while filename in os.listdir(dir_path):
        i += 1
        filename = f"weight_stats_{len(prbl.vessels)}-{len(prbl.factory_nodes)}-{len(prbl.order_nodes)}-{i}.json"
        print(i)

    with open(dir_path + filename, "w") as f:
        json.dump(stat_weights, f)


def plot_avg_weights(op_type: str, match_string: str = "", update_interval: int = 40):
    list_len = 0
    data = defaultdict(list)
    dir_path = "data/output_stats/"
    for file_name in os.listdir(dir_path):
        if file_name.startswith("weight_stats") and match_string in file_name:
            with open(dir_path + file_name) as f:
                json_obj = json.load(f)[op_type]
                for operator, stats_list in json_obj.items():
                    assert list_len == len(stats_list) or not list_len, "Input files have a varying number of iterations"
                    list_len = len(stats_list)
                    shortened_stats_list = _shorten_weight_stats(stats_list, update_interval)
                    data[operator].append(shortened_stats_list)


    assert data, "No raw data was found to base the mean computation on"
    mean_scores_for_operators = {}
    for operator, matrix in data.items():
        mean_scores_for_operators[operator] = np.array(matrix).mean(axis=0)
    x_values = [i * update_interval for i in range(len(next(iter(mean_scores_for_operators.values()))))]
    plot_operator_weights(mean_scores_for_operators, x_values)


def _shorten_weight_stats(stats_list: List[float], interval: int):
    return [stats_list[i] for i in range(0, len(stats_list), interval)]


if __name__ == '__main__':
    plot_avg_weights("repair")
    plot_avg_weights("destroy")
    plot_avg_weights("noise")




import pandas as pd
from typing import List, Dict, Tuple
import json

from numpy.lib import math


def get_results_raw(instance_ids: List[str], dir_path: str) -> Tuple[
    Dict[Tuple[str, str], pd.DataFrame], Dict[str, list]]:
    sheet_names_d = {}
    results = {}
    for inst_id in instance_ids:  # e.g. alns-test_10o_40t
        file_path = dir_path + inst_id + ".xlsx"
        print(f"Handling file with path {file_path}")
        file = pd.ExcelFile(file_path)
        sheet_names = file.sheet_names
        sheet_names_d[inst_id] = list(sheet_names)
        for sheet_name in sheet_names:
            results[(inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])
    return results, sheet_names_d


def get_results(instance_ids: List[str], dir_path: str) -> pd.DataFrame:
    results, sheet_names_d = get_results_raw(instance_ids=instance_ids, dir_path=dir_path)

    df: pd.DataFrame = pd.DataFrame(index=instance_ids)  # .set_index("instance_id", drop=True)

    # Number of calls to the production problem solver
    d = {}
    d_total = {}
    for inst_id in instance_ids:
        filepath = '../../data/input_data/demands/' + inst_id + '.txt'
        with open(filepath) as file:
            d_file = json.load(file)
        d_total[inst_id] = len(d_file.keys())
        d[inst_id] = len(sheet_names_d[inst_id])

    # d_fn: Number of false negatives (heuristic says False, but exact method found solution)
    # d_rth: Average runtime for heuristic
    # d_rte: Average runtime for exact 120 sec
    d_fn = {}
    # d_acc = {}
    d_rth = {}
    d_rte = {}
    d_rte30 = {}
    for inst_id in instance_ids:
        d_fn[inst_id] = 0
        # d_acc[inst_id] = 0
        d_rth[inst_id] = 0
        d_rte[inst_id] = 0
        d_rte30[inst_id] = 0
        sheet_names = sheet_names_d[inst_id]
        count_e_solution_found = 0
        for s in sheet_names:
            data = results[(inst_id, s)]
            d_fn[inst_id] += 1 if (data.loc['feasible', 'exact_120'] and not (data.loc['feasible', 'heuristic'])) else 0
            # d_acc += 1 if ((not data.loc['feasible', 'exact120'] and not data.loc['feasible', 'heuristic'])
            #                or (data.loc['feasible', 'heuristic'])) else 0
            d_rth[inst_id] += data.loc['time', 'heuristic']
            d_rte[inst_id] += data.loc['time', 'exact_120']
            d_rte30[inst_id] += data.loc['time', 'exact_30']

        d_fn[inst_id] /= len(sheet_names)
        d_rth[inst_id] /= len(sheet_names)
        d_rte[inst_id] /= len(sheet_names)
        d_rte30[inst_id] /= len(sheet_names)

    # Percentage number of times the heuristic finds a feasible solution and the exact methods do not
    d_30 = {}
    d_120 = {}
    for inst_id in instance_ids:
        d_30[inst_id] = 0
        d_120[inst_id] = 0
        sheet_names = sheet_names_d[inst_id]
        for s in sheet_names:
            data = results[(inst_id, s)]
            # Note: if heuristic finds a feasible solution, then a feasible solution _does_ exist
            d_30[inst_id] += 1 if (data.loc['feasible', 'heuristic'] and not (data.loc['feasible', 'exact_30'])) else 0
            d_120[inst_id] += 1 if (
                        data.loc['feasible', 'heuristic'] and not (data.loc['feasible', 'exact_120'])) else 0
        d_30[inst_id] /= len(sheet_names)
        d_120[inst_id] /= len(sheet_names)

    # Comparing e120 to the heuristic, the percentage difference in best solution: (heur-ex)/heur
    d_diff_e120 = {}
    d_diff_e30 = {}
    for inst_id in instance_ids:
        d_diff_e120[inst_id] = 0
        d_diff_e30[inst_id] = 0
        count_e120 = 0
        count_e30 = 0
        sheet_names = sheet_names_d[inst_id]
        for s in sheet_names:
            data = results[(inst_id, s)]
            # e120
            if (data.loc['feasible', 'heuristic'] == data.loc['feasible', 'exact_120']) and data.loc[
                'feasible', 'exact_120']:
                d_diff_e120[inst_id] += ((data.loc['cost', 'heuristic'] - data.loc['cost', 'exact_120'])
                                    / data.loc['cost', 'heuristic'])
            if data.loc['feasible', 'heuristic']:
                count_e120 += 1  # implies gap of 0 if only heuristic finds feasible solution
            # e30
            if (data.loc['feasible', 'heuristic'] == data.loc['feasible', 'exact_30']) and data.loc[
                'feasible', 'exact_30']:
                d_diff_e30[inst_id] += ((data.loc['cost', 'heuristic'] - data.loc['cost', 'exact_30'])
                                    / data.loc['cost', 'heuristic'])
            if data.loc['feasible', 'heuristic']:
                count_e30 += 1  # implies gap of 0 if only heuristic finds feasible solution
        d_diff_e120[inst_id] /= count_e120
        d_diff_e30[inst_id] /= count_e30

    d_best_bound_heur_e120 = {}
    for inst_id in instance_ids:
        d_best_bound_heur_e120[inst_id] = 0
        count_feasible_e120 = 0
        sheet_names = sheet_names_d[inst_id]
        for s in sheet_names:
            data = results[(inst_id, s)]
            if data.loc['feasible', 'exact_120']:
                if data.loc['cost', 'heuristic'] != 0:
                    d_best_bound_heur_e120[inst_id] += ((data.loc['cost', 'heuristic'] - data.loc['lower_bound', 'exact_120'])
                                                        / data.loc['cost', 'heuristic'])
                    count_feasible_e120 += 1
        d_best_bound_heur_e120[inst_id] /= count_feasible_e120

    d_mipgap_e30 = {}
    d_mipgap_e120 = {}
    for inst_id in instance_ids:
        d_mipgap_e30[inst_id] = 0
        d_mipgap_e120[inst_id] = 0
        count_feasible_e30 = 0
        count_feasible_e120 = 0
        sheet_names = sheet_names_d[inst_id]
        for s in sheet_names:
            data = results[(inst_id, s)]
            if data.loc['feasible', 'exact_30']:
                d_mipgap_e30[inst_id] += data.loc['mip_gap', 'exact_30'] / 100
                count_feasible_e30 += 1
            if data.loc['feasible', 'exact_120']:
                d_mipgap_e120[inst_id] += data.loc['mip_gap', 'exact_120'] / 100
                count_feasible_e120 += 1
        d_mipgap_e30[inst_id] /= count_feasible_e30
        d_mipgap_e120[inst_id] /= count_feasible_e120

    d_gap_heur = {}
    d_gap_e30 = {}
    d_gap_e120 = {}
    for inst_id in instance_ids:
        d_gap_heur[inst_id] = 0
        d_gap_e30[inst_id] = 0
        d_gap_e120[inst_id] = 0
        count_feasible_heur = 0
        count_feasible_e30 = 0
        count_feasible_e120 = 0
        sheet_names = sheet_names_d[inst_id]
        for s in sheet_names:
            data = results[(inst_id, s)]

            # Find best value
            minimum = math.inf
            if data.loc['feasible', 'heuristic']:
                minimum = min(minimum, data.loc['cost', 'heuristic'])
            if data.loc['feasible', 'exact_30']:
                minimum = min(minimum, data.loc['cost', 'exact_30'])
            if data.loc['feasible', 'exact_120']:
                minimum = min(minimum, data.loc['cost', 'exact_120'])

            # Find gaps
            if data.loc['feasible', 'heuristic']:
                d_gap_heur[inst_id] += (data.loc['cost', 'heuristic'] - minimum) / minimum
                count_feasible_heur += 1
            if data.loc['feasible', 'exact_30']:
                d_gap_e30[inst_id] += (data.loc['cost', 'exact_30'] - minimum) / minimum
                count_feasible_e30 += 1
            if data.loc['feasible', 'exact_120']:
                d_gap_e120[inst_id] += (data.loc['cost', 'exact_120'] - minimum) / minimum
                count_feasible_e120 += 1

        d_gap_heur[inst_id] /= count_feasible_heur
        d_gap_e30[inst_id] /= count_feasible_e30
        d_gap_e120[inst_id] /= count_feasible_e120


    # Columns to USE
    df['average_runtime_heuristic'] = df.index.to_series().map(d_rth)
    df['average_runtime_exact120'] = df.index.to_series().map(d_rte)
    df['average_runtime_exact30'] = df.index.to_series().map(d_rte30)
    df['false_negative_percentage'] = df.index.to_series().map(d_fn)
    df['heuristic_finds_solution_exact30_doesnt'] = df.index.to_series().map(d_30)
    df['heuristic_finds_solution_exact120_doesnt'] = df.index.to_series().map(d_120)
    df['gap_heuristic'] = df.index.to_series().map(d_gap_heur)
    df['gap_exact30'] = df.index.to_series().map(d_gap_e30)
    df['gap_exact120'] = df.index.to_series().map(d_gap_e120)

    # Additional columns, not to use in report
    df['calls_to_production_problem'] = df.index.to_series().map(d_total)
    df['sample_size'] = df.index.to_series().map(d)
    df['percentage_diff_best_solution_e30'] = df.index.to_series().map(d_diff_e30)
    df['percentage_diff_best_solution_e120'] = df.index.to_series().map(d_diff_e120)
    df['best_bound_to_e120_sol'] = df.index.to_series().map(d_best_bound_heur_e120)
    df['mip_gap_exact30'] = df.index.to_series().map(d_mipgap_e30)
    df['mip_gap_exact120'] = df.index.to_series().map(d_mipgap_e120)

    return df


def write_result_to_file(instance_ids: List[str], dir_path: str) -> None:
    df = get_results(instance_ids=instance_ids, dir_path=dir_path)
    out_filepath = dir_path + "summary.xlsx"
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    df.to_excel(excel_writer, sheet_name="_")
    excel_writer.close()


if __name__ == '__main__':
    dir_path = "../../data/output_data/production_performance/"
    instance_ids = ['performance-3-1-10-l-0', 'performance-3-1-10-l-1', 'performance-3-1-10-l-2',
                    'performance-3-1-10-h-0', 'performance-3-1-10-h-1', 'performance-3-1-10-h-2',
                    'performance-3-1-15-l-0', 'performance-3-1-15-l-1', 'performance-3-1-15-l-2',
                    'performance-3-1-15-h-0', 'performance-3-1-15-h-1', 'performance-3-1-15-h-2',
                    'performance-3-1-20-l-0', 'performance-3-1-20-l-1', 'performance-3-1-20-l-2',
                    'performance-3-1-20-h-0', 'performance-3-1-20-h-1', 'performance-3-1-20-h-2',
                    'performance-3-1-40-l-0', 'performance-3-1-40-l-1', 'performance-3-1-40-l-2',
                    'performance-3-1-40-h-0', 'performance-3-1-40-h-1', 'performance-3-1-40-h-2',
                    'performance-3-1-60-l-0', 'performance-3-1-60-l-1', 'performance-3-1-60-l-2',
                    'performance-3-1-60-h-0', 'performance-3-1-60-h-1', 'performance-3-1-60-h-2',
                    'performance-5-2-10-l-0', 'performance-5-2-10-l-1', 'performance-5-2-10-l-2',
                    'performance-5-2-10-h-0', 'performance-5-2-10-h-1', 'performance-5-2-10-h-2',
                    'performance-5-2-15-l-0', 'performance-5-2-15-l-1', 'performance-5-2-15-l-2',
                    'performance-5-2-15-h-0', 'performance-5-2-15-h-1', 'performance-5-2-15-h-2',
                    'performance-5-2-20-l-0', 'performance-5-2-20-l-1', 'performance-5-2-20-l-2',
                    'performance-5-2-20-h-0', 'performance-5-2-20-h-1', 'performance-5-2-20-h-2',
                    'performance-5-2-40-l-0', 'performance-5-2-40-l-1', 'performance-5-2-40-l-2',
                    'performance-5-2-40-h-0', 'performance-5-2-40-h-1', 'performance-5-2-40-h-2',
                    'performance-5-2-60-l-0', 'performance-5-2-60-l-1', 'performance-5-2-60-l-2',
                    'performance-5-2-60-h-0', 'performance-5-2-60-h-1', 'performance-5-2-60-h-2']

    # instance_ids = ["tuning-3-1-40-l"] #"tuning-3-1-20-h", "performance-5-2-15-l-0"]

    write_result_to_file(instance_ids=instance_ids, dir_path=dir_path)

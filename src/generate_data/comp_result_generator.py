from typing import List, Dict, Tuple

import sys
import os

from numpy.lib import math

sys.path.append(os.getcwd())

import locale

locale.setlocale(locale.LC_ALL, '')

import pandas as pd


def get_intermediate(tune_param: str) -> List[str]:
    if tune_param == "remove_percentage_interval":
        return ["(0.05, 0.2)", "(0.1, 0.3)", "(0.2, 0.4)", "(0.3, 0.5)", "(0.1, 0.4)"]
    elif tune_param == "relatedness_location_time":
        return ["[0.02, 0.25]", "[0.01, 0.5]", "[0.005, 1]"]
    elif tune_param == "relatedness_location_precedence":
        return ["[0.02, 0.05]", "[0.01, 0.1]", "[0.005, 0.2]"]
    elif tune_param == "score_params":
        return ["[33, 9, 1]", "[9, 9, 9]", "[33, 9, 13]"]
    elif tune_param == "reaction_param":
        return ["0.1", "0.2", "0.5", "1"]
    elif tune_param == "noise_param":
        return ["0", "0.1", "0.2"]
    elif tune_param == "determinism_param":
        return ["3", "5", "7"]
    else:
        return []


def get_best_obj_vals(instance_ids: List[str], dir_path: str, tune_param: str) -> Dict[str, float]:
    d = {inst_id: math.inf for inst_id in instance_ids}
    tune_combs = get_intermediate(tune_param)
    results, sheet_names_d = get_results_raw(instance_ids=instance_ids, dir_path=dir_path, tune_combs=tune_combs)

    for tune_comb in tune_combs:
        for inst_id in instance_ids:  # e.g. alns-test_10o_40t
            for sheet_name in sheet_names_d[inst_id]:
                obj_val = float(results[(tune_comb, inst_id, sheet_name)].loc['obj_val'])
                if obj_val < d[inst_id]:
                    d[inst_id] = obj_val
    return d


def get_results_raw(instance_ids: List[str], dir_path: str, tune_combs: List[str]) -> Tuple[Dict[Tuple[str, str, str], pd.DataFrame], Dict[str, str]]:
    sheet_names_d = {}
    results = {}
    for tune_comb in tune_combs:
        for inst_id in instance_ids:  # e.g. alns-test_10o_40t
            file_path = dir_path + inst_id + "-" + tune_param + ":" + tune_comb + ".xlsx"
            file = pd.ExcelFile(file_path)
            sheet_names = file.sheet_names
            sheet_names_d[inst_id] = list(sheet_names)
            for sheet_name in sheet_names:  # e.g. run_0 and run_1
                results[(tune_comb, inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])
    return results, sheet_names_d


def get_tuning_results(instance_ids: List[str], dir_path: str, tune_param: str) -> pd.DataFrame:
    tune_combs = get_intermediate(tune_param)
    header = pd.MultiIndex.from_product([tune_combs, ['obj_val', 'time [sec]'] + ['gap']],
                                        names=['param_value', 'value_type'])
    df: pd.DataFrame = pd.DataFrame(index=instance_ids, columns=header)  # .set_index("instance_id", drop=True)

    results, sheet_names_d = get_results_raw(instance_ids, dir_path, tune_combs)

    # Add
    for tune_comb in tune_combs:  # e.g. score_params:[9,9,9]
        for field in ['obj_val', 'time [sec]']:
            d = {}
            for inst_id in instance_ids:
                d[inst_id] = float(
                    sum(results[(tune_comb, inst_id, sheet_name)].loc[field] for sheet_name in
                        sheet_names_d[inst_id])) / len(sheet_names_d[inst_id])
            df[tune_comb, field] = df.index.to_series().map(d)

    # Add average row
    avg_d = {}
    for tune_comb in tune_combs:
        for field in ['obj_val', 'time [sec]']:
            avg_d[(tune_comb, field)] = round(sum(df.loc[inst][tune_comb, field] for inst in instance_ids) / len(instance_ids), 0)
    new_row = pd.Series(data=avg_d, name='average')
    df = df.append(new_row, ignore_index=False)

    # Add gap column
    best_obj_vals_d = get_best_obj_vals(instance_ids=instance_ids, dir_path=dir_path, tune_param=tune_param)
    for tune_comb in tune_combs:
        d = {}
        for inst_id in instance_ids:
            avg_obj = df.loc[inst_id][tune_comb, 'obj_val']
            best_obj = best_obj_vals_d[inst_id]
            d[inst_id] = abs(avg_obj - best_obj) / best_obj
        d['average'] = round(100 * sum(gap for gap in d.values()) / len(instance_ids), 1)
        for inst_id in instance_ids:
            d[inst_id] = round(100 * d[inst_id], 1)

        df[tune_comb, 'gap'] = df.index.to_series().map(d)

    for tune_comb in tune_combs:
        for inst_id in instance_ids:
            df.loc[inst_id][tune_comb, 'obj_val'] = round(df.loc[inst_id][tune_comb, 'obj_val'], 0)
            df.loc[inst_id][tune_comb, 'time [sec]'] = round(df.loc[inst_id][tune_comb, 'time [sec]'], 0)

    return df


def write_result_to_file(dir_path: str, tune_param: str, instance_ids: List[str]) -> None:
    df = get_tuning_results(instance_ids=instance_ids, dir_path=dir_path, tune_param=tune_param)
    out_filepath = dir_path + tune_param + "-summary.xlsx"
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    df.to_excel(excel_writer, sheet_name="_")
    excel_writer.close()


if __name__ == '__main__':
    instance_ids = ["alns-tuning-3-1-20-l", "alns-tuning-3-1-20-h",
                    "alns-tuning-3-1-40-l", "alns-tuning-3-1-40-h",
                    "alns-tuning-3-1-60-l", "alns-tuning-3-1-60-h",
                    "alns-tuning-5-2-20-l", "alns-tuning-5-2-20-h",
                    "alns-tuning-5-2-40-l", "alns-tuning-5-2-40-h",
                    "alns-tuning-5-2-60-l", "alns-tuning-5-2-60-h"]
    dir_path = "../../data/output_data/determinism_param_tuning/"
    tune_param = "determinism_param"

    score_param_results_dict = get_tuning_results(instance_ids, dir_path, tune_param)
    write_result_to_file(dir_path=dir_path, tune_param=tune_param, instance_ids=instance_ids)

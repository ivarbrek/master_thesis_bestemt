from typing import List, Dict, Tuple
from statistics import mean
import sys
import os

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


def load_tuning_result_file(instance_ids: List[str], dir_path: str, tune_param: str) -> pd.DataFrame:
    relevant_avg_fields = ['obj_val', 'time [sec]']
    tune_combs = get_intermediate(tune_param)
    header = pd.MultiIndex.from_product([tune_combs, relevant_avg_fields + ['obj_val/time']],
                                        names=['param_value', 'value_type'])
    df: pd.DataFrame = pd.DataFrame(index=instance_ids, columns=header)  # .set_index("instance_id", drop=True)

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

    for tune_comb in tune_combs:  # e.g. score_params:[9,9,9]
        for field in relevant_avg_fields:
            d = {}
            for inst_id in instance_ids:
                d[inst_id] = float(
                    sum(results[(tune_comb, inst_id, sheet_name)].loc[field] for sheet_name in
                        sheet_names_d[inst_id])) / len(sheet_names_d[inst_id])
            df[tune_comb, field] = df.index.to_series().map(d)

    avg_d = {}
    for tune_comb in tune_combs:
        for field in relevant_avg_fields:
            avg_d[(tune_comb, field)] = sum(df.loc[inst][tune_comb, field] for inst in instance_ids) / len(instance_ids)
    new_row = pd.Series(data=avg_d, name='average')
    df = df.append(new_row, ignore_index=False)

    for tune_comb in tune_combs:
        d = {}
        for inst_id in instance_ids + ['average']:
            d[inst_id] = float(df.loc[inst_id][tune_comb, 'obj_val']) / float(df.loc[inst_id][tune_comb, 'time [sec]'])
        df[tune_comb, 'obj_val/time'] = df.index.to_series().map(d)

    return df


def load_performance_results(result_filenames: List[str], dir_path: str) -> pd.DataFrame:

    # TODO: Include production problem gap between Gurobi and ALNS
    gurobi_attrs = ['obj_val', 'time_limit [sec]', 'number_orders_not_served', 'lower_bound', 'production_start_cost', 'inventory_cost']
    alns_attrs = ['obj_val', 'time [sec]', 'num_orders_not_served', 'production_cost']  #, 'num_iterations']
    gurobi_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'gurobi']
    alns_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'alns']
    if len(gurobi_file_names) == len(alns_file_names):
        print(f"Different number of Gurobi ({len(gurobi_file_names)}) and ALNS ({alns_file_names}) files:")

    gurobi_data = []
    alns_data = []
    for filename in gurobi_file_names:
        df = pd.read_excel(f'{dir_path}/{filename}', skiprows=[0], index_col=0)
        # print(df)
        gurobi_data.append((filename.replace("gurobi-", ""),) + tuple(df.loc[attr, 0] for attr in gurobi_attrs))

    for filename in alns_file_names:
        file = pd.ExcelFile(f'{dir_path}/{filename}')
        sheet_names = file.sheet_names
        sheet_dfs = [file.parse(sheet_name, index_col=[0], skiprows=[1]) for sheet_name in sheet_names]
        for df in sheet_dfs:
            print(df)
        alns_data.append((filename.replace("alns-", ""),) +
                         tuple(mean(df.loc[attr, df.columns[0]] for df in sheet_dfs) for attr in alns_attrs))

    gurobi_df = pd.DataFrame(gurobi_data).set_index(0)
    gurobi_df.columns = gurobi_attrs
    # print(gurobi_df)
    alns_df = pd.DataFrame(alns_data).set_index(0)
    alns_df.columns = alns_attrs
    # print(alns_df)

    new_df = gurobi_df.join(alns_df, lsuffix='_gurobi', rsuffix='_alns')
    writer = pd.ExcelWriter('../../data/output_aggregates/performance_aggregates2.xlsx', engine='openpyxl', mode='w')
    new_df.to_excel(writer)
    writer.close()


def write_gurobi_alns_comparison_to_file():
    dir_path = '../../data/output_data/'
    # print([(i, len(i.split('-'))) for i in os.listdir(dir_path)])
    # for name in os.listdir(dir_path):
    #     print(name)
    #     print(name.split('-')[1])
    instance_ids = [inst_id for inst_id in os.listdir(dir_path)
                    if inst_id[0] != '.' and inst_id.split('-')[1] == 'performance']
    load_performance_results(instance_ids, dir_path)


def write_result_to_file(dir_path: str, tune_param: str, instance_ids: List[str]) -> None:
    df = load_tuning_result_file(instance_ids=instance_ids, dir_path=dir_path, tune_param=tune_param)
    out_filepath = dir_path + tune_param + "-summary.xlsx"
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    df.to_excel(excel_writer, sheet_name="_")
    excel_writer.close()


if __name__ == '__main__':
    write_gurobi_alns_comparison_to_file()
    # instance_ids = ["alns-tuning-3-1-20-l", "alns-tuning-3-1-20-h",
    #                 "alns-tuning-3-1-40-l", "alns-tuning-3-1-40-h",
    #                 "alns-tuning-3-1-60-l",  "alns-tuning-3-1-60-h",
    #                 "alns-tuning-5-2-20-l", "alns-tuning-5-2-20-h",
    #                 "alns-tuning-5-2-40-l", "alns-tuning-5-2-40-h",
    #                 "alns-tuning-5-2-60-l", "alns-tuning-5-2-60-h"]
    # dir_path = "../../data/output_data/removal_param_tuning/"
    # tune_param = "remove_percentage_interval"
    #
    # score_param_results_dict = load_tuning_result_file(instance_ids, dir_path, tune_param)
    # write_result_to_file(dir_path=dir_path, tune_param=tune_param, instance_ids=instance_ids)

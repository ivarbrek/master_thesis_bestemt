from typing import List, Dict, Tuple
from statistics import mean
import sys
import os

from numpy.lib import math

sys.path.append(os.getcwd())

import locale

locale.setlocale(locale.LC_ALL, '')

import src.util.plot as alnsplot
import pandas as pd


def get_intermediate(experiment: str) -> List[str]:
    if experiment == "remove_percentage_interval":
        return ["(0.05, 0.2)", "(0.1, 0.3)", "(0.2, 0.4)", "(0.3, 0.5)", "(0.1, 0.4)"]  # , "(0.1, 0.3)-2"]
    elif experiment == "relatedness_location_time":
        return ["[0.02, 0.25]", "[0.01, 0.5]", "[0.005, 1]"]
    elif experiment == "relatedness_location_precedence":
        return ["[0.02, 0.05]", "[0.01, 0.1]", "[0.005, 0.2]"]
    elif experiment == "score_params":
        return ["[33, 9, 1]", "[9, 9, 9]", "[33, 9, 13]"]
    elif experiment == "reaction_param":
        return ["0.1", "0.2", "0.5", "1"]
    elif experiment == "noise_param":
        return ["0", "0.1", "0.2"]
    elif experiment == "determinism_param":
        return ["3", "5", "7"]
    elif experiment == "noise_destination_param":
        return ["0.1", "0.5", "1"]
    elif experiment == "permute_chance":
        return ["0", "0.05", "0.1", "0.3"]
    elif experiment == "percentage_best_solution_production_solved":
        return ["0", "0.05", "0.1"]
    elif experiment == "score_params_second":
        return ["[33, 9, 1]", "[9, 9, 9]", "[33, 9, 13]"]
    elif experiment == "convergence":
        return ["construction_heuristic_solution", "improvement_heuristic_solution"]
    elif experiment == "lns_config":
        return ["all_adaptive", "one_pair", "all_random"]
    elif experiment == "subproblem_integration":
        return ["default", "all_ppfc", "no_ppfc"]
    else:
        return []


def get_best_obj_vals_in_experiment(instance_ids: List[str], dir_path: str, experiment: str) -> Dict[str, float]:
    d = {inst_id: math.inf for inst_id in instance_ids}
    configs = get_intermediate(experiment)
    results, sheet_names_d = get_results_raw(instance_ids=instance_ids, dir_path=dir_path, configs=configs,
                                             experiment=experiment)

    for config in configs:
        if config != "construction_heuristic_solution":  # construction heuristic will at best be as good as other solutions (deterministic)
            for inst_id in instance_ids:  # e.g. alns-test_10o_40t
                for sheet_name in sheet_names_d[inst_id]:
                    obj_val = float(results[(config, inst_id, sheet_name)].loc['obj_val'])
                    if obj_val < d[inst_id]:
                        d[inst_id] = obj_val
    return d


def get_results_raw(instance_ids: List[str], dir_path: str, configs: List[str], experiment: str,
                    recover_penalty: bool = False) -> Tuple[
    Dict[Tuple[str, str, str], pd.DataFrame], Dict[str, list]]:
    sheet_names_d = {}
    results = {}
    for config in configs:
        for inst_id in instance_ids:  # e.g. alns-test_10o_40t
            file_path = dir_path + inst_id + "-" + experiment + ":" + config + ".xlsx"
            file = pd.ExcelFile(file_path)
            sheet_names = file.sheet_names
            sheet_names_d[inst_id] = list(sheet_names)
            for sheet_name in sheet_names:  # e.g. run_0 and run_1
                results[(config, inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])

    if recover_penalty:
        results = recover_penalty_value(instance_ids, results, configs, sheet_names_d)

    return results, sheet_names_d


def recover_penalty_value(instance_ids: List[str], results: Dict[Tuple[str, str, str], pd.DataFrame],
                          configs: List[str], sheet_names_d: Dict[str, list]) -> Dict[
    Tuple[str, str, str], pd.DataFrame]:
    adjust_values_dict = {"alns-tuning-3-1-20-l": 117351.1, "alns-tuning-3-1-20-h": 75623.2,
                          "alns-tuning-3-1-40-l": 120691.9, "alns-tuning-3-1-40-h": 94741.1,
                          "alns-tuning-3-1-60-l": 127851.1, "alns-tuning-3-1-60-h": 100569.8,
                          "alns-tuning-5-2-20-l": 112781.2, "alns-tuning-5-2-20-h": 115090.6,
                          "alns-tuning-5-2-40-l": 120339.8, "alns-tuning-5-2-40-h": 113498.3,
                          "alns-tuning-5-2-60-l": 124880.3, "alns-tuning-5-2-60-h": 128764.9}

    for config in configs:
        for inst_id in instance_ids:
            for sheet_name in sheet_names_d[inst_id]:
                if "_penalty_adjusted" not in list(results[(config, inst_id, sheet_name)].index):
                    num_orders_not_served = int(results[(config, inst_id, sheet_name)].loc['num_orders_not_served'])
                    if num_orders_not_served > 0:
                        print(f"Adjusting for {num_orders_not_served} at test instance {inst_id} {sheet_name} "
                              f"configuration {config}: adding {num_orders_not_served} of {adjust_values_dict[inst_id]} = {num_orders_not_served * adjust_values_dict[inst_id]}")
                    # prev = int(results[(config, inst_id, sheet_name)].loc['obj_val'])
                    results[(config, inst_id, sheet_name)].loc['routing_cost'] = \
                    results[(config, inst_id, sheet_name)].loc['routing_cost'] + num_orders_not_served * \
                    adjust_values_dict[inst_id]
                    results[(config, inst_id, sheet_name)].loc['obj_val'] = results[(config, inst_id, sheet_name)].loc[
                                                                                'obj_val'] + num_orders_not_served * \
                                                                            adjust_values_dict[inst_id]
                    new_row = pd.Series(data={'Unnamed: 1': True}, name='_penalty_adjusted')
                    results[(config, inst_id, sheet_name)] = results[(config, inst_id, sheet_name)].append(new_row,
                                                                                                           ignore_index=False)
                    # new = int(results[(config, inst_id, sheet_name)].loc['obj_val'])
                    # if prev != new:
                    #     print("Changed objective!")

    return results


def get_results(instance_ids: List[str], dir_path: str, experiment: str, recover_penalty: bool = False) -> pd.DataFrame:
    configs = get_intermediate(experiment)
    header = pd.MultiIndex.from_product([configs, ['obj_val', 'prod_obj_%', 'time [sec]'] + ['gap']],
                                        names=['param_value', 'value_type'])
    df: pd.DataFrame = pd.DataFrame(index=instance_ids, columns=header)  # .set_index("instance_id", drop=True)

    results, sheet_names_d = get_results_raw(instance_ids, dir_path, configs=configs, experiment=experiment,
                                             recover_penalty=recover_penalty)

    # Add obj_val column
    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(
                sum(results[(config, inst_id, sheet_name)].loc['obj_val'] for sheet_name in
                    sheet_names)) / len(sheet_names)
        df[config, 'obj_val'] = df.index.to_series().map(d)

    # Add prod_obj_% column
    for config in configs:
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = 100 * (
                    float(sum(results[(config, inst_id, sheet_name)].loc['production_cost'] for sheet_name in
                              sheet_names)) / len(sheet_names)) / df.loc[inst_id][config, 'obj_val']
            df[config, 'prod_obj_%'] = df.index.to_series().map(d)

    # Add time [sec] column
    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(
                sum(results[(config, inst_id, sheet_name)].loc['time [sec]'] for sheet_name in
                    sheet_names)) / len(sheet_names)
        df[config, 'time [sec]'] = df.index.to_series().map(d)

    # Add average row
    avg_d = {}
    for config in configs:
        # Zero decimal points
        for field in ['obj_val', 'time [sec]']:
            avg_d[(config, field)] = sum(df.loc[inst][config, field] for inst in instance_ids) / len(instance_ids)
        # Two decimal points
        for field in ['prod_obj_%']:
            avg_d[(config, field)] = sum(df.loc[inst][config, field] for inst in instance_ids) / len(instance_ids)
    new_row = pd.Series(data=avg_d, name='average')
    df = df.append(new_row, ignore_index=False)

    # Add gap column
    best_obj_vals_d = get_best_obj_vals_in_experiment(instance_ids=instance_ids, dir_path=dir_path,
                                                      experiment=experiment)
    for config in configs:
        d = {}
        for inst_id in instance_ids:
            avg_obj = df.loc[inst_id][config, 'obj_val']
            best_obj = best_obj_vals_d[inst_id]
            d[inst_id] = abs(avg_obj - best_obj) / best_obj
        d['average'] = 100 * sum(gap for gap in d.values()) / len(instance_ids)
        for inst_id in instance_ids:
            d[inst_id] = 100 * d[inst_id]
        df[config, 'gap'] = df.index.to_series().map(d)

    # Round off
    # for config in configs:
    #     for inst_id in instance_ids:
    #         df.loc[inst_id][config, 'obj_val'] = round(df.loc[inst_id][config, 'obj_val'], 0)
    #         df.loc[inst_id][config, 'time [sec]'] = round(df.loc[inst_id][config, 'time [sec]'], 0)
    #         df.loc[inst_id][config, 'prod_obj_%'] = round(df.loc[inst_id][config, 'prod_obj_%'], 1)

    return df


def load_performance_results(result_filenames: List[str], dir_path: str, production: bool = False) -> pd.DataFrame:
    # TODO: Include production problem gap between Gurobi and ALNS
    """
    :param result_filenames:
    :param dir_path:
    :param production: True if comparing Gurobi's production solution to greedy production heuristic, else False
    :return:
    """
    extra_attr = [] if production else ['number_orders_not_served']
    gurobi_attrs = ['obj_val', 'time_limit [sec]'] + extra_attr + ['lower_bound', 'production_start_cost',
                                                                   'inventory_cost']
    extra_attr = [] if production else ['number_orders_not_served']
    alns_attrs = ['obj_val', 'time [sec]'] + extra_attr + ['production_cost']  # , 'num_iterations']

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


# def load_production_performance_results(result_filenames, dir_path) -> pd.DataFrame:
#     gurobi_attrs = ['obj_val', 'time_limit [sec]', 'lower_bound', 'production_start_cost',
#                     'inventory_cost']
#     alns_attrs = ['obj_val', 'time [sec]', 'num_orders_not_served', 'production_cost']  # , 'num_iterations']
#     gurobi_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'gurobi']
#     alns_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'alns']
#     if len(gurobi_file_names) == len(alns_file_names):
#         print(f"Different number of Gurobi ({len(gurobi_file_names)}) and ALNS ({alns_file_names}) files:")
#
#
# return pd.DataFrame


def write_gurobi_alns_comparison_to_file():
    dir_path = '../../data/output_data/production_performance'  # '../../data/output_data/'
    # print([(i, len(i.split('-'))) for i in os.listdir(dir_path)])
    # for name in os.listdir(dir_path):
    #     print(name)
    #     print(name.split('-')[1])
    instance_ids = [inst_id for inst_id in os.listdir(dir_path)
                    if inst_id[0] != '.' and inst_id.split('-')[1] == 'performance']
    load_performance_results(instance_ids, dir_path)


def write_result_to_file(dir_path: str, experiment: str, instance_ids: List[str],
                         recover_penalty: bool = False) -> None:
    df = get_results(instance_ids=instance_ids, dir_path=dir_path, experiment=experiment,
                     recover_penalty=recover_penalty)
    out_filepath = dir_path + experiment + "-summary.xlsx"
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    df.to_excel(excel_writer, sheet_name="_")
    excel_writer.close()


def get_solution_development_df(inst_id: str, dir_path: str) -> pd.DataFrame:
    file_path = dir_path + inst_id + "-stats.xlsx"
    file = pd.ExcelFile(file_path)
    sheet_names = list(file.sheet_names)
    header = pd.Index(sheet_names)
    iters = [i for i in range(len(file.parse(sheet_names[0], index_col=[0], skiprows=[0])))]
    df: pd.DataFrame = pd.DataFrame(index=iters, columns=header)

    for sheet_name in sheet_names:  # e.g. run_0 and run_1
        results = file.parse(sheet_name, index_col=[0])
        df[sheet_name] = results

    df["average_obj"] = df.mean(numeric_only=True, axis=1)
    return df


def plot_solution_development(solution_dev_df: pd.DataFrame, inst_id: str, minimum: bool, save: bool):
    if not minimum:  # using average value
        solution_dev_df = solution_dev_df[['average_obj']]
    else:
        solution_dev_df = pd.DataFrame(data=solution_dev_df.min(axis=1))
    solution_tup = list(solution_dev_df.to_records())
    y_label = "Lowest Objective Value" if minimum else "Average Objective Value"
    alnsplot.plot_alns_history(solution_costs=solution_tup, lined=True, legend=inst_id, save=save, x_label="Iterations",
                               y_label=y_label)


if __name__ == '__main__':
    dir_path = "../../data/output_data/"  # noise_destination_param_tuning/"
    experiment = "convergence"  # supported: tuning / convergence / lns_config / subproblem_integration

    assert (experiment in ["tuning", "convergence", "lns_config", "subproblem_integration"])

    if experiment == "tuning":
        instance_ids = ["alns-tuning-3-1-20-l", "alns-tuning-3-1-20-h",
                        "alns-tuning-3-1-40-l", "alns-tuning-3-1-40-h",
                        "alns-tuning-3-1-60-l", "alns-tuning-3-1-60-h",
                        "alns-tuning-5-2-20-l", "alns-tuning-5-2-20-h",
                        "alns-tuning-5-2-40-l", "alns-tuning-5-2-40-h",
                        "alns-tuning-5-2-60-l", "alns-tuning-5-2-60-h"]
        tune_param = "score_params"
        dir_path += tune_param + "_tuning_4/"

        penalty_recover_params = ["remove_percentage_interval", "relatedness_location_time",
                                  "relatedness_location_precedence", "reaction_param", "noise_param",
                                  "determinism_param", "noise_destination_param"] #"score_params",
        recover_penalty = True if tune_param in penalty_recover_params else False

        # score_param_results_dict = get_results(instance_ids, dir_path, experiment=tune_param,
        #                                        recover_penalty=recover_penalty)
        write_result_to_file(dir_path=dir_path, experiment=tune_param, instance_ids=instance_ids,
                             recover_penalty=recover_penalty)

    elif experiment == "production_performance":
        write_gurobi_alns_comparison_to_file()

    else:
        instance_ids = ["alns-tuning-3-1-20-l", "alns-tuning-3-1-20-h",
                        "alns-tuning-3-1-40-l", "alns-tuning-3-1-40-h",
                        "alns-tuning-3-1-60-l", "alns-tuning-3-1-60-h",
                        "alns-tuning-5-2-20-l", "alns-tuning-5-2-20-h",
                        "alns-tuning-5-2-40-l", "alns-tuning-5-2-40-h",
                        "alns-tuning-5-2-60-l", "alns-tuning-5-2-60-h"]
        dir_path += experiment + "/"

        # results_dict = get_results(instance_ids, dir_path, experiment=experiment)
        # write_result_to_file(dir_path=dir_path, experiment=experiment, instance_ids=instance_ids)

        if experiment == "convergence":
            # Make plots
            save = True
            instance_ids_plot = instance_ids
            for inst_id in instance_ids_plot:
                sol_costs = get_solution_development_df(inst_id=inst_id, dir_path=dir_path)
                plot_solution_development(sol_costs, inst_id=inst_id, minimum=False, save=save)

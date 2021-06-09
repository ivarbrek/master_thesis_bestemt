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
    elif experiment == "convergence":
        return ["construction_heuristic_solution", "improvement_heuristic_solution"]
    elif experiment == "lns_config":
        return ["all_adaptive", "one_pair", "all_random"]
    elif experiment == "subproblem_integration":
        return ["default", "reinsert_with_ppfc"]
    elif experiment == "time_periods":
        return ["tp_length=1", "tp_length=2", "tp_length=3"]
    elif experiment == "time_windows":
        return ["tw_length=3d", "tw_length=4d", "tw_length=5d"]
    else:
        return [""]


def get_best_obj_vals_in_experiment(instance_ids: List[str], dir_path: str, experiment: str) -> Dict[str, float]:
    d = {inst_id: math.inf for inst_id in instance_ids}
    configs = get_intermediate(experiment)
    results, sheet_names_d = get_results_raw(instance_ids=instance_ids, dir_path=dir_path, configs=configs,
                                             experiment=experiment)

    if experiment in ["time_periods", "time_windows"]:
        instance_ids_for_config = {config: [inst_id for inst_id in instance_ids if config in inst_id]
                                   for config in configs}
    else:
        instance_ids_for_config = {config: instance_ids for config in configs}

    for config in configs:
        if config != "construction_heuristic_solution":  # construction heuristic will at best be as good as other solutions (deterministic)
            for inst_id in instance_ids_for_config[config]:  # e.g. alns-test_10o_40t
            # for inst_id in instance_ids:
                for sheet_name in sheet_names_d[inst_id]:
                    obj_val = float(results[(config, inst_id, sheet_name)].loc['obj_val'])
                    if obj_val < d[inst_id]:
                        d[inst_id] = obj_val
    return d


def get_results_raw(instance_ids: List[str], dir_path: str, configs: List[str], experiment: str) -> Tuple[
    Dict[Tuple[str, str, str], pd.DataFrame], Dict[str, list]]:
    sheet_names_d = {}
    results = {}
    if experiment in ["time_periods", "time_windows"]:
        for config in configs:
            for inst_id in instance_ids:  # e.g. alns-test_10o_40t
                i = -2 if experiment == "time_periods" else -1
                if inst_id.split("-")[i] == config:
                    file_path = dir_path + inst_id + ".xlsx"
                    file = pd.ExcelFile(file_path)
                    sheet_names = file.sheet_names
                    sheet_names_d[inst_id] = list(sheet_names)
                    for sheet_name in sheet_names:  # e.g. run_0 and run_1
                        results[(config, inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])
    elif experiment == "None":
        for inst_id in instance_ids:  # e.g. alns-test_10o_40t
                file_path = dir_path + inst_id
                file = pd.ExcelFile(file_path)
                sheet_names = file.sheet_names
                sheet_names_d[inst_id] = list(sheet_names)
                for sheet_name in sheet_names:  # e.g. run_0 and run_1
                    results[("", inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])
    else:
        for config in configs:
            for inst_id in instance_ids:  # e.g. alns-test_10o_40t
                file_path = dir_path + inst_id + "-" + experiment + ":" + config + ".xlsx"
                file = pd.ExcelFile(file_path)
                sheet_names = file.sheet_names
                sheet_names_d[inst_id] = list(sheet_names)
                for sheet_name in sheet_names:  # e.g. run_0 and run_1
                    results[(config, inst_id, sheet_name)] = file.parse(sheet_name, index_col=[0])
    return results, sheet_names_d


def get_results(instance_ids: List[str], dir_path: str, experiment: str) -> pd.DataFrame:
    configs = get_intermediate(experiment)
    header = pd.MultiIndex.from_product([configs, ['obj_val', 'prod_obj_%', 'time [sec]'] + ['gap']],
                                        names=['param_value', 'value_type'])

    results, sheet_names_d = get_results_raw(instance_ids, dir_path, configs=configs, experiment=experiment)

    if experiment == "time_periods":
        instance_ids_old = instance_ids[:]
        a, b = -14, -2
        instance_ids = list(set(inst_id[:a] + inst_id[b:] for inst_id in instance_ids))
        results = {(config, inst_id[:a] + inst_id[b:], sheet): datafr
                   for (config, inst_id, sheet), datafr in results.items()}
        sheet_names_d = {inst_id[:a] + inst_id[b:]: sheets for inst_id, sheets in sheet_names_d.items()}
    elif experiment == "time_windows":
        instance_ids_old = instance_ids[:]
        a = -13
        instance_ids = list(set(inst_id[:a] for inst_id in instance_ids))
        results = {(config, inst_id[:a], sheet): datafr for (config, inst_id, sheet), datafr in results.items()}
        sheet_names_d = {inst_id[:a]: sheets for inst_id, sheets in sheet_names_d.items()}
    df: pd.DataFrame = pd.DataFrame(index=instance_ids, columns=header)  # .set_index("instance_id", drop=True)

    # Add obj_val column
    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(sum(results[(config, inst_id, sheet_name)].loc['obj_val']
                                   for sheet_name in sheet_names)) / len(sheet_names)
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

    # Add unserved orders row
    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(
                sum(results[(config, inst_id, sheet_name)].loc['num_orders_not_served'] for sheet_name in
                    sheet_names)) / len(sheet_names)
        df[config, 'num_orders_not_served'] = df.index.to_series().map(d)

    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(
                sum(results[(config, inst_id, sheet_name)].loc['transport_cost'] for sheet_name in
                    sheet_names)) / len(sheet_names)
        df[config, 'transport_cost'] = df.index.to_series().map(d)

    # Add best obj row
    for config in configs:  # e.g. score_params:[9,9,9]
        d = {}
        for inst_id in instance_ids:
            sheet_names = sheet_names_d[inst_id] if config != "construction_heuristic_solution" else ["run_initial"]
            d[inst_id] = float(
                min(results[(config, inst_id, sheet_name)].loc[
                        'obj_val', results[(config, inst_id, sheet_name)].columns[0]] for sheet_name in sheet_names))
        df[config, 'min_obj_val'] = df.index.to_series().map(d)

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
    if experiment == "time_periods":
        best_obj_vals_d = get_best_obj_vals_in_experiment(instance_ids=instance_ids_old, dir_path=dir_path,
                                                          experiment=experiment)
        best_obj_vals_d = {inst_id: min(best_obj for old_inst_id, best_obj in best_obj_vals_d.items()
                                        if old_inst_id[:-14] == inst_id[:-2] and old_inst_id[-2:] == inst_id[-2:])
                           for inst_id in instance_ids}
    elif experiment == "time_windows":
        best_obj_vals_d = get_best_obj_vals_in_experiment(instance_ids=instance_ids_old, dir_path=dir_path,
                                                          experiment=experiment)
        best_obj_vals_d = {inst_id: min(best_obj for old_inst_id, best_obj in best_obj_vals_d.items()
                                        if old_inst_id[:-13] == inst_id)
                           for inst_id in instance_ids}
    else:
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


def load_performance_results(result_filenames: List[str], dir_path: str) -> pd.DataFrame:

    # TODO: Include production problem gap between Gurobi and ALNS
    gurobi_attrs = ['obj_val', 'number_orders_not_served', 'lower_bound', 'production_start_cost', 'inventory_cost']
    alns_attrs = ['obj_val', 'time [sec]', 'num_orders_not_served', 'production_cost']  #, 'num_iterations']
    # gurobi_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'gurobi']
    gurobi3600_file_names = [name for name in os.listdir(dir_path + '/gurobi-3600')]
    gurobi600_file_names = [name for name in os.listdir(dir_path + '/gurobi-600')]
    alns_file_names = [name for name in os.listdir(dir_path + '/alns_performance')]
    # alns_file_names = [instance_id for instance_id in result_filenames if instance_id.split('-')[0] == 'alns']
    if len(gurobi3600_file_names) != len(alns_file_names) or len(gurobi3600_file_names) != len(gurobi600_file_names):
        print(f"Different number of Gurobi ({len(gurobi3600_file_names), len(gurobi600_file_names)}) "
              f"and ALNS ({len(alns_file_names)}) files:")

    gurobi_data_600 = []
    gurobi_data_3600 = []
    best_solution = {}
    for filename in gurobi3600_file_names:
        try:
            df_600 = pd.read_excel(f'{dir_path}/gurobi-600/{filename}', skiprows=[0], index_col=0)
            solve_time = df_600.loc['solve_time', 0] if 'solve_time' in df_600.index else df_600.loc['time_limit [sec]', 0]
            gurobi_data_600.append((filename.replace("gurobi-", ""), solve_time) + tuple(df_600.loc[attr, 0] for attr in gurobi_attrs))
        except ValueError:
            gurobi_data_600.append((filename.replace("gurobi-", ""), 600) + (0, 0, 0, 0, 0))

        df_3600 = pd.read_excel(f'{dir_path}/gurobi-3600/{filename}', skiprows=[0], index_col=0)
        solve_time = df_3600.loc['solve_time', 0] if 'solve_time' in df_3600.index else df_3600.loc['time_limit [sec]', 0]
        gurobi_data_3600.append((filename.replace("gurobi-", ""), solve_time) + tuple(df_3600.loc[attr, 0] for attr in gurobi_attrs))
        best_solution[filename[7:]] = df_3600.loc['obj_val', 0]
        # print(df)

    gurobi_df_600 = pd.DataFrame(gurobi_data_600).set_index(0)
    gurobi_df_3600 = pd.DataFrame(gurobi_data_3600).set_index(0)
    gurobi_df_600.columns = gurobi_df_3600.columns = ['time'] + gurobi_attrs

    gurobi_df = gurobi_df_3600.join(gurobi_df_600, lsuffix='_gurobi_3600', rsuffix='_gurobi_600')

    alns_data = []
    for filename in alns_file_names:
        file = pd.ExcelFile(f'{dir_path}/alns_performance/{filename}')
        sheet_names = file.sheet_names
        sheet_dfs = [file.parse(sheet_name, index_col=[0], skiprows=[1]) for sheet_name in sheet_names]
        min_obj = min(*(df.loc['obj_val', df.columns[0]] for df in sheet_dfs), best_solution[filename[5:]])
        best_solution[filename[5:]] = min_obj
        avg_obj = mean(df.loc['obj_val', df.columns[0]] for df in sheet_dfs)
        avg_gap = mean(df.loc['obj_val', df.columns[0]] - min_obj for df in sheet_dfs) / min_obj
        avg_prod_share = mean(df.loc['production_cost', df.columns[0]] for df in sheet_dfs) / avg_obj
        alns_data.append((filename.replace("alns-", ""),) +
                         tuple(mean(df.loc[attr, df.columns[0]] for df in sheet_dfs) for attr in alns_attrs) +
                         (avg_gap, avg_prod_share))
    alns_df = pd.DataFrame(alns_data).set_index(0)
    alns_df.columns = alns_attrs + ['gap', 'prod_obj_%']

    # Add gurobi gap
    gurobi_df['gap_gurobi_3600'] = [(gurobi_df.loc[fname[7:], 'obj_val_gurobi_3600'] - best_solution[fname[7:]]) / best_solution[fname[7:]]
                                    for fname in gurobi3600_file_names]
    gurobi_df['gap_gurobi_600'] = [(gurobi_df.loc[fname[7:], 'obj_val_gurobi_600'] - best_solution[fname[7:]]) / best_solution[fname[7:]]
                                   for fname in gurobi600_file_names]
    new_df = gurobi_df.join(alns_df, rsuffix='_alns')
    writer = pd.ExcelWriter('../../data/output_aggregates/performance_aggregates2.xlsx', engine='openpyxl', mode='w')
    new_df.sort_index(inplace=True)
    new_df.to_excel(writer)
    writer.close()


def write_gurobi_alns_comparison_to_file():
    dir_path = '../../data/output_data/'
    # print([(i, len(i.split('-'))) for i in os.listdir(dir_path)])
    # for name in os.listdir(dir_path):
    #     print(name)
    #     print(name.split('-')[1])
    instance_ids = [inst_id for inst_id in os.listdir(dir_path)
                    if len(inst_id.split('-')) > 1 and inst_id.split('-')[1] == 'performance']
    load_performance_results(instance_ids, dir_path)


def write_result_to_file(dir_path: str, experiment: str, instance_ids: List[str]) -> None:
    df = get_results(instance_ids=instance_ids, dir_path=dir_path, experiment=experiment)
    out_filepath = dir_path + experiment + "-summary.xlsx"
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    df.to_excel(excel_writer, sheet_name="_")
    excel_writer.close()
    print("Results written to", out_filepath)


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


def plot_solution_development(solution_dev_df: pd.DataFrame, inst_id: str):
    solution_dev_df = solution_dev_df[['average_obj']]
    solution_tup = list(solution_dev_df.to_records())
    alnsplot.plot_alns_history(solution_costs=solution_tup, lined=True, legend=inst_id)


if __name__ == '__main__':
    dir_path = "../../data/output_data/"  # noise_destination_param_tuning/"
    experiment = "None"  # supported: tuning / convergence / lns_config / subproblem_integration / time_periods

    assert (experiment in ["tuning", "convergence", "lns_config", "subproblem_integration", "time_periods",
                           "time_windows", "performance", "None"])

    if experiment == "tuning":
        instance_ids = ["alns-tuning-3-1-20-l", "alns-tuning-3-1-20-h",
                        "alns-tuning-3-1-40-l", "alns-tuning-3-1-40-h",
                        "alns-tuning-3-1-60-l", "alns-tuning-3-1-60-h",
                        "alns-tuning-5-2-20-l", "alns-tuning-5-2-20-h",
                        "alns-tuning-5-2-40-l", "alns-tuning-5-2-40-h",
                        "alns-tuning-5-2-60-l", "alns-tuning-5-2-60-h"]
        tune_param = "noise_destination_param"
        dir_path += tune_param + "_tuning/"
        score_param_results_dict = get_results(instance_ids, dir_path, experiment=tune_param)
        write_result_to_file(dir_path=dir_path, experiment=tune_param, instance_ids=instance_ids)
    elif experiment in ["time_periods", "time_windows"]:
        dir_path += experiment + "/"
        instance_ids = [filename[:-5] for filename in os.listdir(dir_path)
                        if filename.startswith('alns')]
        # instance_ids = ['alns-tp_testing-3-1-30-tp_length=1-0',
        #                 'alns-tp_testing-3-1-30-tp_length=2-0',
        #                 'alns-tp_testing-3-1-30-tp_length=3-0']
        results_dict = get_results(instance_ids, dir_path, experiment=experiment)
        write_result_to_file(dir_path=dir_path, experiment=experiment, instance_ids=instance_ids)
    elif experiment == "time_windows":
        instance_ids = [filename[:-5] for filename in os.listdir("../../data/output_data/time_windows")
                        if filename.startswith('alns')]
    elif experiment == "performance":
        write_gurobi_alns_comparison_to_file()
    else:
        file_prefix = 'alns-'
        dir_path = '../../data/output_data/external_depot_60/'
        instance_ids = [inst_id for inst_id in os.listdir(dir_path) if inst_id.startswith(file_prefix)]

        results_dict = get_results(instance_ids, dir_path, experiment=experiment)
        write_result_to_file(dir_path=dir_path, experiment=experiment, instance_ids=instance_ids)

        if experiment == "convergence":
            # Make plots
            instance_ids_plot = instance_ids  # ["alns-3-1-20-l", "alns-3-1-20-h"]
            for inst_id in instance_ids_plot:
                sol_costs = get_solution_development_df(inst_id=inst_id, dir_path=dir_path)
                plot_solution_development(sol_costs, inst_id=inst_id)

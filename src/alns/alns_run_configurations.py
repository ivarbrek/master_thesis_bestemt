import sys
import os
from typing import Dict, List

sys.path.append(os.getcwd())
import src.alns.alns as a
import pandas as pd
import json


def create_excel_writer(args, config: str):
    output_filepath_orig = "data/output_data/alns-" + str(args.input_filepath.split("/")[-1])
    # Just for Ivar's instances
    # output_filepath_orig = "data/output_data/ivars_instances/" + str(args.input_filepath.split("/")[-1])
    output_filepath = output_filepath_orig[:-5] + "-" + str(args.experiment) + ":" + config + ".xlsx"
    return pd.ExcelWriter(output_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})


def run_alns_for_tuning(prbl, experiment_values, args, num_alns_iterations):
    for idx in range(len(experiment_values)):
        excel_writer = create_excel_writer(args, str(experiment_values[idx]))
        val = experiment_values[idx]

        for i in range(args.num_runs):
            alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl,
                                                                                   iterations=num_alns_iterations,
                                                                                   parameter_tune=args.experiment,
                                                                                   parameter_tune_value=val)
            alns.write_to_file(excel_writer, i, stop_criterion, alns_iter, run_time,
                               parameter_tune=args.experiment, parameter_tune_value=val)

        excel_writer.close()


def run_alns_for_convergence(prbl, args, num_alns_iterations):
    output_filepath_stats = ("data/output_data/performance-" + str(args.input_filepath.split("/")[-1]))[
                            :-5] + "-stats.xlsx"

    # Just for Ivar's instances
    # output_filepath_stats = ("data/output_data/ivars_instances/" + str(args.input_filepath.split("/")[-1]))[
    #                         :-5] + "-stats.xlsx"

    excel_writer_init = create_excel_writer(args=args, config="construction_heuristic_solution")
    excel_writer_alns = create_excel_writer(args=args, config="improvement_heuristic_solution")
    excel_writer_stats = pd.ExcelWriter(output_filepath_stats, engine='openpyxl', mode='w',
                                        options={'strings_to_formulas': False})

    # First write initial solution data to file (deterministic)
    alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=0,
                                                                           parameter_tune="None",
                                                                           parameter_tune_value="None")
    alns.write_to_file(excel_writer_init, -1, stop_criterion, alns_iter, run_time)

    # Then write solution data for each run
    for i in range(args.num_runs):
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None")
        alns.write_to_file(excel_writer_alns, i, stop_criterion, alns_iter, run_time)

        sheet_name = "run_" + str(i)
        df = pd.DataFrame(solution_stats, index=[0]).transpose()
        df.to_excel(excel_writer_stats, sheet_name=sheet_name, startrow=1)

    excel_writer_init.close()
    excel_writer_alns.close()
    excel_writer_stats.close()


def run_alns_for_lns_config(prbl, args, num_alns_iterations):
    excel_writer_adaptive = create_excel_writer(args=args, config="all_adaptive")
    excel_writer_onepair = create_excel_writer(args=args, config="one_pair")
    excel_writer_random = create_excel_writer(args=args, config="all_random")

    for i in range(args.num_runs):
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               adaptive_weights=True)
        alns.write_to_file(excel_writer_adaptive, i, stop_criterion, alns_iter, run_time)
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               all_operators=False
                                                                               )
        alns.write_to_file(excel_writer_onepair, i, stop_criterion, alns_iter, run_time)
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               adaptive_weights=False)
        alns.write_to_file(excel_writer_random, i, stop_criterion, alns_iter, run_time)

    excel_writer_adaptive.close()
    excel_writer_onepair.close()
    excel_writer_random.close()


def run_alns_for_subproblem_integration(prbl, args, num_alns_iterations):
    excel_writer_default = create_excel_writer(args=args, config="default")
    excel_writer_all_ppfc = create_excel_writer(args=args, config="all_ppfc")
    excel_writer_no_ppfc = create_excel_writer(args=args, config="no_ppfc")

    for i in range(args.num_runs):
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               ppfc="default")
        alns.write_to_file(excel_writer_default, i, stop_criterion, alns_iter, run_time)
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               ppfc="all_ppfc")
        alns.write_to_file(excel_writer_all_ppfc, i, stop_criterion, alns_iter, run_time)
        alns, stop_criterion, alns_iter, run_time, solution_stats = a.run_alns(prbl, iterations=num_alns_iterations,
                                                                               parameter_tune="None",
                                                                               parameter_tune_value="None",
                                                                               ppfc="no_ppfc")
        alns.write_to_file(excel_writer_no_ppfc, i, stop_criterion, alns_iter, run_time)

    excel_writer_default.close()
    excel_writer_all_ppfc.close()
    excel_writer_no_ppfc.close()


def run_alns_for_production_heuristic_performance(prbl, args, num_alns_iterations):
    demands: List[Dict] = a.run_alns(prbl, iterations=num_alns_iterations,
                                     parameter_tune="None",
                                     parameter_tune_value="None",
                                     record_demand=True)
    d = {i: str(demands[i]) for i in range(len(demands))}  # value is also a dict: (factory, product, time): amount
    filename = args.input_filepath.split("/")[-1][:-5]
    with open('data/input_data/demands/' + filename + '.txt', 'w') as file:
        file.write(json.dumps(d))

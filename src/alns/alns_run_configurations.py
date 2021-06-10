import sys
import os

sys.path.append(os.getcwd())
import src.alns.alns as a
from src.alns.run_alns import run_alns
import pandas as pd
from src.alns.solution import ProblemDataExtended
from typing import List
import math
from src.util.plot import plot_clustered_locations


def create_excel_writer(args, config: str):
    output_filepath_orig = "data/output_data/alns-" + str(args.input_filepath.split("/")[-1])
    # Just for Ivar's instances
    # output_filepath_orig = "data/output_data/ivars_instances/" + str(args.input_filepath.split("/")[-1])
    if args.experiment in ["None"]:
        output_filepath = output_filepath_orig
    elif args.experiment == "factory-decompose":
        output_filepath = output_filepath_orig[:-5] + "-factory-decompose.xlsx"
    else:
        output_filepath = output_filepath_orig[:-5] + "-" + str(args.experiment) + ":" + config + ".xlsx"
    return pd.ExcelWriter(output_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})


def run_alns_for_tuning(prbl, experiment_values, args, num_alns_iterations):
    for idx in range(len(experiment_values)):
        excel_writer = create_excel_writer(args, str(experiment_values[idx]))
        val = experiment_values[idx]

        for i in range(args.num_runs):
            alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl,
                                                                                iterations=num_alns_iterations,
                                                                                parameter_tune=args.experiment,
                                                                                parameter_tune_value=val)
            alns.write_to_file(excel_writer, i, stop_crierion, alns_iter, run_time,
                               parameter_tune=args.experiment, parameter_tune_value=val)

        excel_writer.close()


def run_alns_for_convergence(prbl, args, num_alns_iterations):
    output_filepath_stats = ("data/output_data/performance-" + str(args.input_filepath.split("/")[-1]))[:-5] + "-stats.xlsx"

    # Just for Ivar's instances
    # output_filepath_stats = ("data/output_data/ivars_instances/" + str(args.input_filepath.split("/")[-1]))[
    #                         :-5] + "-stats.xlsx"

    excel_writer_init = create_excel_writer(args=args, config="construction_heuristic_solution")
    excel_writer_alns = create_excel_writer(args=args, config="improvement_heuristic_solution")
    excel_writer_stats = pd.ExcelWriter(output_filepath_stats, engine='openpyxl', mode='w',
                                        options={'strings_to_formulas': False})

    # First write initial solution data to file (deterministic)
    alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=0,
                                                                        parameter_tune="None",
                                                                        parameter_tune_value="None")
    alns.write_to_file(excel_writer_init, -1, stop_crierion, alns_iter, run_time)

    # Then write solution data for each run
    for i in range(args.num_runs):
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None")
        alns.write_to_file(excel_writer_alns, i, stop_crierion, alns_iter, run_time)

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
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None",
                                                                            adaptive_weights=True)
        alns.write_to_file(excel_writer_adaptive, i, stop_crierion, alns_iter, run_time)
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None",
                                                                            all_operators=False)
        alns.write_to_file(excel_writer_onepair, i, stop_crierion, alns_iter, run_time)
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None",
                                                                            adaptive_weights=False)
        alns.write_to_file(excel_writer_random, i, stop_crierion, alns_iter, run_time)

    excel_writer_adaptive.close()
    excel_writer_onepair.close()
    excel_writer_random.close()


def run_alns_for_subproblem_integration(prbl, args, num_alns_iterations):
    excel_writer_default = create_excel_writer(args=args, config="default")
    excel_writer_reinsertppfc = create_excel_writer(args=args, config="reinsert_with_ppfc")

    for i in range(args.num_runs):
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None",
                                                                            force_reinsert_ppfc_true=False)
        alns.write_to_file(excel_writer_default, i, stop_crierion, alns_iter, run_time)
        alns, stop_crierion, alns_iter, run_time, solution_stats = run_alns(prbl, iterations=num_alns_iterations,
                                                                            parameter_tune="None",
                                                                            parameter_tune_value="None",
                                                                            force_reinsert_ppfc_true=True)
        alns.write_to_file(excel_writer_reinsertppfc, i, stop_crierion, alns_iter, run_time)

    excel_writer_default.close()
    excel_writer_reinsertppfc.close()


def run_alns_basic(prbl, args, num_alns_iterations):
    excel_writer = create_excel_writer(args, config="")

    for i in range(args.num_runs):
        alns, stop_criterion, alns_iter, run_time, solution_stats = run_alns(prbl,
                                                                             iterations=num_alns_iterations,
                                                                             parameter_tune="None",
                                                                             parameter_tune_value="None")
        alns.write_to_file(excel_writer, i, stop_criterion, alns_iter, run_time)

    excel_writer.close()


def run_alns_decompose_per_factory(prbl: ProblemDataExtended, args, num_alns_iterations):
    # Decompose problem
    order_assignments = {f: [] for f in prbl.factory_nodes}
    v = prbl.vessels[0]  # dummy vessel
    for order in prbl.order_nodes:
        closest_factory = min(((f, prbl.transport_times_exact[(v, f, order)]) for f in prbl.factory_nodes),
                              key=lambda item: item[1])[0]
        order_assignments[closest_factory].append(order)

    vessel_assignments = {f: [] for f in prbl.factory_nodes}
    for vessel in prbl.vessels:
        start_factory = prbl.vessel_initial_locations[vessel]
        vessel_assignments[start_factory].append(vessel)

    # Run each sub-problem
    excel_writer = create_excel_writer(args, config="")
    for i in range(args.num_runs):
        total_time = 0
        alns_list = []
        stop_criterion = None

        for factory in prbl.factory_nodes:
            print(factory, vessel_assignments[factory], order_assignments[factory])
            sub_prbl = prbl.get_factory_sub_problem(factory, order_assignments[factory], vessel_assignments[factory])
            alns, stop_criterion, _, run_time, solution_stats = run_alns(sub_prbl,
                                                                         iterations=num_alns_iterations,
                                                                         parameter_tune="None",
                                                                         parameter_tune_value="None")
            alns_list.append(alns)
            total_time += run_time

        write_factory_decompose_subs_to_file(excel_writer, alns_list, i, stop_criterion, num_alns_iterations, total_time)
    excel_writer.close()


def write_factory_decompose_subs_to_file(excel_writer: pd.ExcelWriter, alns_list: List['Alns'], id: int, stop_criterion: str,
                                         alns_iter: int, alns_time: int) -> None:
    cooling_rate = str(0.999) if alns_iter == 0 else str(round(math.pow(0.002, (1 / alns_iter)), 4))
    solution_dict = {'obj_val': round(sum(alns.best_sol_total_cost for alns in alns_list), 2),
                     'production_cost': round(sum(alns.best_sol_production_cost for alns in alns_list), 2),
                     'routing_cost': round(sum(alns.best_sol_routing_cost for alns in alns_list), 2),
                     'num_orders_not_served': sum(len(alns.best_sol.get_orders_not_served()) for alns in alns_list),
                     'stop_crierion': stop_criterion,
                     'num_iterations': alns_iter,
                     'time [sec]': alns_time,
                     'score_params': str(a.alnsparam.score_params),
                     'reaction_param': str(a.alnsparam.reaction_param),
                     'start_temperature_controlparam': str(a.alnsparam.start_temperature_controlparam),
                     'cooling_rate': cooling_rate,
                     'remove_percentage_interval': str(a.alnsparam.remove_percentage_interval),
                     'noise_param': str(a.alnsparam.noise_param),
                     'determinism_param': str(a.alnsparam.determinism_param),
                     'related_removal_weight_param': str(a.alnsparam.related_removal_weight_param),
                     'noise_destination_param': str(a.alnsparam.noise_destination_param)}
    if id is None:
        id = "initial"
    sheet_name = "run_" + str(id)
    df = pd.DataFrame(solution_dict, index=[0]).transpose()
    df.to_excel(excel_writer, sheet_name=sheet_name, startrow=1)





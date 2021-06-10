import sys
import os

sys.path.append(os.getcwd())
from typing import List, Union, Tuple, Dict
from src.alns.alns import ProblemDataExtended, Alns
from src.util import stats
import src.alns.alns_parameters as alnsparam
import math
from collections import defaultdict
from time import time
import alns_run_configurations
import argparse


def parse_experiment_values(experiment: str) -> Tuple[
    bool, List[Union[Tuple[float, float], int, float, str, List[float]]]]:
    if experiment == "None":
        return False, []
    if experiment == "remove_percentage_interval":
        return True, [(0.05, 0.2), (0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.1, 0.4)]
    if experiment == "relatedness_location_time":
        return True, [[0.02, 0.25], [0.01, 0.5], [0.005, 1]]
    if experiment == "relatedness_location_precedence":
        return True, [[0.02, 0.05], [0.01, 0.1], [0.005, 0.2]]
    if experiment == "score_params":
        return True, [[33, 9, 1], [9, 9, 9], [33, 9, 13]]
    if experiment == "reaction_param":
        return True, [0.1, 0.2, 0.5, 1]
    if experiment == "noise_param":
        return True, [0, 0.1, 0.2]
    if experiment == "determinism_param":
        return True, [3, 5, 7]
    if experiment == "noise_destination_param":
        return True, [0.1, 0.5, 1]
    if experiment in ["convergence", "lns_config", "subproblem_integration"]:
        return False, []
    print(f"Values could not be parsed")
    return False, []


def run_alns(prbl: ProblemDataExtended, parameter_tune: str, parameter_tune_value, iterations: int = 0,
             max_time: int = 0, skip_production_problem_postprocess: bool = False, adaptive_weights: bool = True,
             all_operators: bool = True, force_reinsert_ppfc_true: bool = False, save_weights: bool = False,
             verbose: bool = True) -> Union[Dict[Tuple[str, str, int], int], None,
                                            Tuple[Alns, str, int, int, Dict[int, int]]]:
    if iterations * max_time > 0:
        print(f"Multiple stopping criteria given. Choosing iterations criteria, where iterations={iterations}.")
        max_time = 0

    if iterations == 0:
        cooling_rate = 0.999

    else:
        cooling_rate = math.pow(0.002, (1 / iterations))
    print(f"Using cooling rate of", cooling_rate)

    parameter_values = {
        'weight_min_threshold': alnsparam.weight_min_threshold,  # 0.2
        'reaction_param': alnsparam.reaction_param,  # 0.1,
        'score_params': alnsparam.score_params,
        'start_temperature_controlparam': alnsparam.start_temperature_controlparam,
        'cooling_rate': cooling_rate,  # alnsparam.cooling_rate,  # 0.995,
        'max_iter_same_solution': alnsparam.max_iter_same_solution,  # 50,
        'max_iter_seg': alnsparam.max_iter_seg,  # 40,
        'percentage_best_solution_production_solved': alnsparam.percentage_best_solution_production_solved,
        'remove_percentage_interval': alnsparam.remove_percentage_interval,  # (0.1, 0.3),
        'remove_num_percentage_adjust': alnsparam.remove_num_percentage_adjust,  # 0.05,
        'determinism_param': alnsparam.determinism_param,  # 5,
        'noise_param': alnsparam.noise_param,  # 0.25,
        'noise_destination_param': alnsparam.noise_destination_param,
        'relatedness_precedence': alnsparam.relatedness_precedence,
        'related_removal_weight_param': alnsparam.related_removal_weight_param,
        'production_infeasibility_strike_max': alnsparam.production_infeasibility_strike_max,  # 0,
        'ppfc_slack_increment': alnsparam.ppfc_slack_increment,  # 0.05,
        'inventory_reward': alnsparam.inventory_reward,  # False,
        'reinsert_with_ppfc': alnsparam.reinsert_with_ppfc}

    if parameter_tune != "None":
        if parameter_tune == 'relatedness_location_time':
            parameter_values['related_removal_weight_param']['relatedness_location_time'] = parameter_tune_value
        elif parameter_tune == 'relatedness_location_precedence':
            parameter_values['related_removal_weight_param']['relatedness_location_precedence'] = parameter_tune_value
        if parameter_tune in parameter_values.keys():
            parameter_values[parameter_tune] = parameter_tune_value
        else:
            print(f"Parameter {parameter_tune} does not exist. Parameters left unchanged.")

    if not adaptive_weights:
        parameter_values['reaction_param'] = 0  # weights remain the same throughout the whole search (not adaptive)
    if force_reinsert_ppfc_true:
        parameter_values['reinsert_with_ppfc'] = True

    destroy_op = ['d_random',
                  'd_worst',
                  'd_voyage_random',
                  'd_voyage_worst',
                  'd_route_random',
                  'd_route_worst',
                  'd_related_location_time',
                  'd_related_location_precedence']
    repair_op = ['r_greedy', 'r_2regret', 'r_3regret']

    if not all_operators:
        destroy_op = ['d_random']
        repair_op = ['r_greedy']

    print("ALNS running...")
    alns = Alns(problem_data=prbl,
                destroy_op=destroy_op,
                repair_op=repair_op,
                weight_min_threshold=parameter_values['weight_min_threshold'],  # 0.2
                reaction_param=parameter_values['reaction_param'],  # 0.1,
                score_params=parameter_values['score_params'],
                start_temperature_controlparam=parameter_values['start_temperature_controlparam'],
                cooling_rate=parameter_values['cooling_rate'],  # 0.995,
                max_iter_same_solution=parameter_values['max_iter_same_solution'],  # 50,
                max_iter_seg=parameter_values['max_iter_seg'],  # 40,
                percentage_best_solutions_production_solved=parameter_values['percentage_best_solution_production_solved'],
                remove_percentage_interval=parameter_values['remove_percentage_interval'],  # (0.1, 0.3),
                remove_num_percentage_adjust=parameter_values['remove_num_percentage_adjust'],  # 0.05,
                determinism_param=parameter_values['determinism_param'],  # 5,
                noise_param=parameter_values['noise_param'],  # 0.25,
                noise_destination_param=parameter_values['noise_destination_param'],
                relatedness_precedence=parameter_values['relatedness_precedence'],
                related_removal_weight_param=parameter_values['related_removal_weight_param'],
                production_infeasibility_strike_max=parameter_values['production_infeasibility_strike_max'],  # 0,
                ppfc_slack_increment=parameter_values['ppfc_slack_increment'],  # 0.05,
                inventory_reward=parameter_values['inventory_reward'],  # False,
                reinsert_with_ppfc=parameter_values['reinsert_with_ppfc'],  # False,
                verbose=False
                )

    if verbose:
        print("Route after initialization")
        alns.current_sol.print_routes()
        print(f"Obj: {alns.current_sol_cost:n}   Not served: {alns.current_sol.get_orders_not_served()}")
        print("\nRemove num:", alns.remove_num_interval, "\n")

    _stat_solution_cost = []
    _stat_repair_weights = defaultdict(list)
    _stat_destroy_weights = defaultdict(list)
    _stat_noise_weights = defaultdict(list)
    _stat_best_routing_solution_dict = {0: alns.best_sol_routing_cost}  # initializing with construction heuristic solution
    _stat_best_total_solution_dict = {0: alns.best_sol_total_cost}
    t0 = time()
    # print(t0)
    i = 0

    while i < iterations or (time() - t0) < max_time:
        alns.run_alns_iteration()
        i += 1

        if verbose:
            print("Iteration", i)
            alns.current_sol.print_routes()
            for f, visits in alns.current_sol.factory_visits.items():
                print(f"{f}: {visits}")
            print(f"Obj: {alns.current_sol_cost:,}   Not served: {alns.current_sol.get_orders_not_served()}")
            print("Slack factor:", round(alns.current_sol.ppfc_slack_factor, 2),
                  "  Infeasible strike:", alns.production_infeasibility_strike)
            print()

        _stat_best_routing_solution_dict[i] = alns.best_sol_routing_cost
        _stat_best_total_solution_dict[i] = alns.best_sol_total_cost
        _stat_solution_cost.append((i, alns.current_sol_cost))
        for op, score in alns.destroy_op_weight.items():
            _stat_destroy_weights[op].append(score)
        for op, score in alns.repair_op_weight.items():
            _stat_repair_weights[op].append(score)
        for op, score in alns.noise_op_weight.items():
            _stat_noise_weights[op].append(score)

    if skip_production_problem_postprocess:  # do not need to solve the production problem
        alns.best_sol.print_routes()
        return alns.best_sol.get_y_dict()

    exact_method_prod_cost = alns.production_model.get_production_cost(alns.best_sol, verbose=True, time_limit=30)
    heuristic_method_prod_cost = alns.production_heuristic.get_cost(alns.best_sol)
    if exact_method_prod_cost < heuristic_method_prod_cost:
        print("Exact method improved heuristic solution. "
              "Diff:",  round(heuristic_method_prod_cost - exact_method_prod_cost, 1))
        alns.best_sol_production_cost = exact_method_prod_cost
        exact_prod_sol_is_best = True
    else:
        print(f"Heuristic solution was NOT improved by exact method. "
              f"Diff:",  round(heuristic_method_prod_cost - exact_method_prod_cost, 1))
        alns.best_sol_production_cost = heuristic_method_prod_cost
        exact_prod_sol_is_best = False
    alns.best_sol_total_cost = alns.best_sol_routing_cost + alns.best_sol_production_cost

    alns_time = time() - t0
    if verbose:
        print()
        print(f"...ALNS terminating  ({round(time() - t0)}s)")
        alns.best_sol.print_routes()
        print("Not served:", alns.best_sol.get_orders_not_served())
        if exact_prod_sol_is_best:
            alns.production_model.print_solution_simple()
        else:
            alns.production_heuristic.print_sol(first_n_time_periods=40)

        print("Routing obj:", alns.best_sol_routing_cost, "Prod obj:", round(alns.best_sol_production_cost, 1),
              "Total:", alns.best_sol_total_cost)

        print(f"Best solution updated {alns.new_best_solution_feasible_production_count} times")
        print(f"Candidate to become best solution rejected {alns.new_best_solution_infeasible_production_count} times, "
              f"because of production infeasibility")
        print(f"{len(alns.previous_solutions)} different solutions accepted")
        print(f"Repaired solution rejected {alns.ppfc_infeasible_count} times, because of PPFC infeasibility")

        # plot.plot_alns_history(_stat_solution_cost)
        # plot.plot_operator_weights(_stat_destroy_weights)
        # plot.plot_operator_weights(_stat_repair_weights)
        # plot.plot_operator_weights(_stat_noise_weights)
        # plot.plot_alns_history(_stat_best_routing_solution)

    if save_weights:
        stats.save_weights_stats(prbl, _stat_destroy_weights, _stat_repair_weights, _stat_noise_weights)

    print("Orders not_served:", len(alns.best_sol.get_orders_not_served()))
    stop_criterion = str(iterations) + " iterations" if max_time == 0 else str(max_time) + " sec"
    return alns, stop_criterion, i, int(alns_time), _stat_best_total_solution_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process ALNS input parameters')
    parser.add_argument('input_filepath', type=str, help='path of input data file')
    parser.add_argument('num_runs', type=int, help='number of runs using same input data file')
    parser.add_argument('experiment', type=str,
                        help="parameter to tune, or convergence/lns_config/subproblem_integration")
    parser.add_argument('num_iterations', type=int,
                        help="number of ALNS iterations")
    args = parser.parse_args()
    # Execution line format: python3 src/alns/alns.py data/input_data/f1-v3-o20-t50.xlsx 5

    tuning, tuning_values = parse_experiment_values(args.experiment)

    num_alns_iterations = args.num_iterations
    write_solution_details = False

    # prbl = ProblemDataExtended('../../data/input_data/large_testcase.xlsx')
    # prbl = ProblemDataExtended('../../data/input_data/gurobi_testing/f1-v3-o20-t72-i0.05-tw4.xlsx')
    print("File:", args.input_filepath.split('/')[-1])
    prbl = ProblemDataExtended(args.input_filepath)

    # WRITE SUMMARY OF RESULT DATA TO FILE
    if tuning:
        alns_run_configurations.run_alns_for_tuning(prbl=prbl, experiment_values=tuning_values, args=args,
                                    num_alns_iterations=num_alns_iterations)
    elif args.experiment == "convergence":
        alns_run_configurations.run_alns_for_convergence(prbl=prbl, args=args, num_alns_iterations=num_alns_iterations)
    elif args.experiment == "lns_config":
        alns_run_configurations.run_alns_for_lns_config(prbl=prbl, args=args, num_alns_iterations=num_alns_iterations)
    elif args.experiment == "subproblem_integration":
        alns_run_configurations.run_alns_for_subproblem_integration(prbl=prbl, args=args, num_alns_iterations=num_alns_iterations)
    elif args.experiment == "factory-decompose":
        alns_run_configurations.run_alns_decompose_per_factory(prbl=prbl, args=args, num_alns_iterations=num_alns_iterations)
    else:
        alns_run_configurations.run_alns_basic(prbl=prbl, args=args, num_alns_iterations=num_alns_iterations)


if __name__ == '__main__':
    print("HEI!")
    parser = argparse.ArgumentParser(description='process ALNS input parameters')
    parser.add_argument('input_filepath', type=str, help='path of input data file')
    parser.add_argument('num_iterations', type=int,
                        help="number of ALNS iterations")
    args = parser.parse_args()

    num_alns_iterations = args.num_iterations
    print("File:", args.input_filepath.split('/')[-1])
    prbl = ProblemDataExtended(args.input_filepath)

    run_alns(prbl, parameter_tune="None", iterations=num_alns_iterations)

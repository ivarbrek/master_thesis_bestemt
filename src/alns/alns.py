import sys
import os

sys.path.append(os.getcwd())

import math
import random
import numpy as np
from time import time
from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict
import function
import argparse
import pandas as pd

from src.alns.solution import Solution, ProblemDataExtended
from src.models.production_model import ProductionModel
import src.alns.alns_parameters as alnsparam
from src.util import plot
from src.util import stats
from src.alns.production_problem_heuristic import ProductionProblemHeuristic, ProductionProblem
import locale

locale.setlocale(locale.LC_ALL, '')

int_inf = 999999


class Alns:
    it_seg_count: int
    best_sol: Solution
    best_sol_routing_cost: int
    best_sol_production_cost: int
    best_sol_total_cost: int
    current_sol: Solution
    current_sol_cost: int
    production_model: ProductionModel
    production_heuristic: ProductionProblemHeuristic
    inventory_reward: bool
    update_type: int
    destroy_op_score: Dict[str, float]
    repair_op_score: Dict[str, float]
    destroy_op_weight: Dict[str, float]
    repair_op_weight: Dict[str, float]
    destroy_op_segment_usage: Dict[str, int]
    repair_op_segment_usage: Dict[str, int]
    reaction_param: float
    score_params: Dict[int, int]
    weight_min_threshold: float
    temperature: float
    cooling_rate: float
    max_iter_same_solution: int
    iter_same_solution: int
    determinism_param: int
    relatedness_precedence: Dict[Tuple[str, str], int]
    related_removal_weight_param: Dict[str, List[float]]
    new_best_solution_feasible_production_count = 0
    new_best_solution_infeasible_production_count = 0
    ppfc_infeasible_count = 0
    production_infeasibility_strike = 0
    production_infeasibility_strike_max: int
    previous_solutions: Set[str] = set()

    def __init__(self, problem_data: ProblemDataExtended,
                 destroy_op: List[str],
                 repair_op: List[str],
                 weight_min_threshold: float,
                 reaction_param: float,
                 score_params: List[int],
                 start_temperature_controlparam: float,
                 cooling_rate: float,
                 max_iter_seg: int,
                 percentage_best_solutions_production_solved: float,
                 max_iter_same_solution: int,
                 remove_percentage_interval: Tuple[float, float],
                 remove_num_percentage_adjust: float,
                 determinism_param: int,
                 noise_param: float,
                 noise_destination_param: float,
                 relatedness_precedence: Dict[Tuple[str, str], int],
                 related_removal_weight_param: Dict[str, List[float]],
                 inventory_reward: bool,
                 production_infeasibility_strike_max: int,
                 ppfc_slack_increment: float,
                 reinsert_with_ppfc: bool,
                 verbose: bool = False) -> None:

        self.verbose = verbose

        # ALNS  parameters
        self.max_iter_seg = max_iter_seg
        self.remove_num_interval = [round(remove_percentage_interval[0] * len(problem_data.order_nodes)),
                                    round(remove_percentage_interval[1] * len(problem_data.order_nodes))]
        self.remove_num_adjust = math.ceil(remove_num_percentage_adjust * len(problem_data.order_nodes))
        self.determinism_param = determinism_param
        self.noise_param = noise_param
        self.noise_destination_param = noise_destination_param
        self.max_iter_same_solution = max_iter_same_solution
        self.iter_same_solution = 0
        self.reinsert_with_ppfc = reinsert_with_ppfc

        # Solutions
        self.current_sol = self.repair_kregret(2, Solution(problem_data))
        self.production_model = ProductionModel(prbl=problem_data,
                                                demands=self.current_sol.get_demand_dict(),
                                                inventory_reward_extension=inventory_reward)
        self.production_heuristic = ProductionProblemHeuristic(ProductionProblem(problem_data))

        # Ensure prod feasibility
        self.current_sol = self.adjust_sol_exact(self.current_sol, remove_num=self.remove_num_adjust)
        self.current_sol_cost = self.current_sol.get_solution_routing_cost()
        self.best_sol = self.current_sol
        self.best_sol_routing_cost = round(self.current_sol_cost)
        self.best_sol_total_cost = self.best_sol_routing_cost + self.production_heuristic.get_cost(routing_sol=self.best_sol)
        self.record_solution(self.current_sol)
        self.percentage_best_solutions_production_solved = percentage_best_solutions_production_solved

        # Operator weights, scores and usage
        noise_op = [True, False]
        self.weight_min_threshold = weight_min_threshold
        self.destroy_op_weight = {op: self.weight_min_threshold for op in destroy_op}
        self.repair_op_weight = {op: self.weight_min_threshold for op in repair_op}
        self.noise_op_weight = {op: self.weight_min_threshold for op in noise_op}
        self.destroy_op_score = {op: 0 for op in destroy_op}
        self.repair_op_score = {op: 0 for op in repair_op}
        self.noise_op_score = {op: 0 for op in noise_op}
        self.destroy_op_segment_usage = {op: 0 for op in destroy_op}
        self.repair_op_segment_usage = {op: 0 for op in repair_op}
        self.noise_op_segment_usage = {op: 0 for op in noise_op}
        self.reaction_param = reaction_param
        self.score_params = {i: score_params[i] for i in range(len(score_params))}

        self.cooling_rate = cooling_rate
        self.temperature = -(self.best_sol_routing_cost * start_temperature_controlparam) / math.log(0.5)

        self.it_seg_count = 0  # Iterations done in one segment - can maybe do this in run_alns_iteration?
        self.num_best_total_sol_updates = 0

        self.production_infeasibility_strike_max = production_infeasibility_strike_max
        self.ppfc_slack_increment = ppfc_slack_increment

        self.unique_initial_factory_visits: Set = set()
        self.performed_initial_factory_visits_permutations = 0

        # Relatedness operator parameters
        self.related_removal_weight_param = related_removal_weight_param
        self.relatedness_precedence = self.set_relatedness_precedence_dict(relatedness_precedence)

    def __repr__(self):
        destroy_op = [(k, round(v, 2)) for k, v in sorted(self.destroy_op_weight.items(), key=lambda item: item[0])]
        repair_op = [(k, round(v, 2)) for k, v in sorted(self.repair_op_weight.items(), key=lambda item: item[0])]
        return (
            f"Best solution with routing cost {self.best_sol_routing_cost}: \n"
            f"{self.best_sol} \n"
            # f"Current solution with routing cost {self.current_sol_cost}: \n"
            # f"{self.current_sol} \n"
            f"Insertion candidates (orders not served): {self.best_sol.get_orders_not_served()} \n"
            f"Destroy operators {destroy_op} \n"
            f"and repair operators {repair_op} \n")

    def set_relatedness_precedence_dict(self, d: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], int]:
        # Returns a dict with key all combinations of zones (z1, z2) and value defined relatedness of two zones
        zones = set()
        original_keys: List[Tuple[str, str]] = list(d.keys())
        for (z1, z2) in original_keys:
            d[(z2, z1)] = d[(z1, z2)]  # symmetry
            zones.add(z1)
            zones.add(z2)
        for z in zones:
            d[(z, z)] = 0  # a zone is perfectly related to itself
        if len(zones) < len(self.current_sol.prbl.zones):
            print("Relatedness of zones is not properly defined.")
        return d

    def adjust_sol_exact(self, sol: Solution = None, remove_num: int = None) -> Solution:
        """
        :param sol: solution to be adjusted, default is self.current_sol
        :param remove_num: number of order nodes to be removed before checking for prod feasibility,
                default is self.remove_num
        :return: exact production feasible solution
        """
        sol = sol if sol else self.current_sol
        prod_feasible, infeasible_factory = self.production_heuristic.is_feasible(sol)
        while not prod_feasible:
            sol = self.remove_from_factory(sol, infeasible_factory, remove_num)
            prod_feasible, infeasible_factory = self.production_heuristic.is_feasible(sol)
        return sol

    def adjust_sol_ppfc(self, sol: Solution = None, remove_num: int = None, repair_op: str = None) -> Solution:
        """
        :param sol: solution to be adjusted, default is self.current_sol
        :param remove_num: number of order nodes to be removed before checking for prod feasibility,
                default is self.remove_num
        :param repair_op: repair operator
        :return: production feasible solution according to ppfc
        """
        sol = sol if sol else self.current_sol
        prod_feasible, infeasible_factory = sol.check_production_feasibility()
        self.ppfc_infeasible_count += int(not prod_feasible)
        infeasible = [infeasible_factory]
        while not prod_feasible:
            sol = self.remove_from_factory(sol, infeasible_factory, remove_num)
            prod_feasible, infeasible_factory = sol.check_production_feasibility()
            infeasible.append(infeasible_factory)

        # print("Infeasible:", infeasible)
        orders_not_served = len(sol.get_orders_not_served())
        if repair_op and self.reinsert_with_ppfc:  # Reinsert nodes using the ppfc check
            sol = self.repair(repair_op, sol, apply_noise=True, use_ppfc=True)
            # print(orders_not_served - len(sol.get_orders_not_served()), "orders reinserted")
        return sol

    def update_scores(self, destroy_op: str, repair_op: str, noise_op: bool, update_type: int) -> None:
        if update_type == -1:
            return
        elif update_type == -2:  # New global best routing, but not production feasible
            return
        elif update_type == -3:  # Solution was updated because max_iter_same_solution was reached
            return
        elif not 0 <= update_type < len(self.score_params):
            print("Update type does not exist. Scores not updated.")
            return
        self.destroy_op_score[destroy_op] += self.score_params[update_type]
        self.repair_op_score[repair_op] += self.score_params[update_type]
        self.noise_op_score[noise_op] += self.score_params[update_type]

    def update_weights(self) -> None:
        # Update weights based on scores, "blank" scores and operator segment usage
        # w = (1-r)w + r*(pi/theta)

        # Destroy
        for op in self.destroy_op_weight.keys():
            if self.destroy_op_segment_usage[op] > 0:
                self.destroy_op_weight[op] = max(((1 - self.reaction_param) * self.destroy_op_weight[op] +
                                                  self.reaction_param * self.destroy_op_score[op] /
                                                  self.destroy_op_segment_usage[op]),
                                                 self.weight_min_threshold)
            self.destroy_op_score[op] = 0
            self.destroy_op_segment_usage[op] = 0

        # Repair
        for op in self.repair_op_weight.keys():
            if self.repair_op_segment_usage[op] > 0:
                self.repair_op_weight[op] = max(((1 - self.reaction_param) * self.repair_op_weight[op] +
                                                 self.reaction_param * self.repair_op_score[op] /
                                                 self.repair_op_segment_usage[op]),
                                                self.weight_min_threshold)
            self.repair_op_score[op] = 0
            self.repair_op_segment_usage[op] = 0

        # Noise
        for op in self.noise_op_weight:
            if self.noise_op_segment_usage:
                self.noise_op_weight[op] = max(((1 - self.reaction_param) * self.noise_op_weight[op] +
                                                self.reaction_param * self.noise_op_score[op] /
                                                self.noise_op_segment_usage[op]),
                                               self.weight_min_threshold)

        self.it_seg_count = 0

    def choose_operators(self) -> Tuple[str, str, bool]:
        # Choose operators, probability of choosing an operator is based on current weights
        destroy_weights_normalized = (np.fromiter(self.destroy_op_weight.values(), float) /
                                      sum(self.destroy_op_weight.values()))
        destroy_op = np.random.choice(list(self.destroy_op_weight.keys()), p=destroy_weights_normalized)
        self.destroy_op_segment_usage[destroy_op] += 1

        noise_weights_normalized = (np.fromiter(self.noise_op_weight.values(), float) /
                                    sum(self.noise_op_weight.values()))
        noise_op = np.random.choice(list(self.noise_op_weight.keys()), p=noise_weights_normalized)
        self.noise_op_segment_usage[noise_op] += 1

        repair_weights_normalized = (np.fromiter(self.repair_op_weight.values(), float) /
                                     sum(self.repair_op_weight.values()))
        repair_op = np.random.choice(list(self.repair_op_weight.keys()), p=repair_weights_normalized)
        self.repair_op_segment_usage[repair_op] += 1

        # print(destroy_op, repair_op, 'noise' if noise_op else 'no noise')
        return destroy_op, repair_op, noise_op

    def generate_new_solution(self, destroy_op: str, repair_op: str, apply_noise: bool) -> Solution:
        # Generate new feasible solution x' based on given operators
        # Return x'
        candidate_sol = self.current_sol.copy()
        candidate_sol = self.destroy(destroy_op, candidate_sol)
        if random.random() < alnsparam.permute_chance:
            candidate_sol = self.permute_initial_factory_visits(candidate_sol)
        candidate_sol = self.repair(repair_op, candidate_sol, apply_noise)
        return candidate_sol

    def permute_initial_factory_visits(self, candidate_sol: Solution) -> Solution:
        permutation_found = candidate_sol.permute_factory_visits()
        self.performed_initial_factory_visits_permutations += int(permutation_found)
        for factory in candidate_sol.prbl.factory_nodes:
            self.unique_initial_factory_visits.add((factory, candidate_sol.get_initial_factory_visits(factory)))
        return candidate_sol

    def destroy(self, destroy_op: str, sol: Solution) -> Union[Solution, None]:
        remove_num = random.randint(*self.remove_num_interval)
        if destroy_op == "d_random":
            return self.destroy_random(sol, remove_num)
        elif destroy_op == "d_worst":
            return self.destroy_worst(sol, remove_num)
        elif destroy_op == "d_related_location_time":
            return self.destroy_related(sol, remove_num, rel_measure=self._relatedness_location_time)
        elif destroy_op == "d_related_location_precedence":
            return self.destroy_related(sol, remove_num, rel_measure=self._relatedness_location_precedence)
        elif destroy_op == "d_voyage_random":
            return self.destroy_voyage_random(sol, remove_num)
        elif destroy_op == "d_voyage_worst":
            return self.destroy_voyage_worst(sol, remove_num)
        elif destroy_op == "d_route_random":
            return self.destroy_route_random(sol, remove_num)
        elif destroy_op == "d_route_worst":
            return self.destroy_route_worst(sol, remove_num)
        else:
            print("Destroy operator does not exist")
            return None

    def remove_from_factory(self, sol: Solution, factory_node_id: str, remove_num: int = 1):
        # Compute removal candidates
        candidates = [(vessel, idx, sol.get_removal_utility(vessel, idx))
                      for vessel, idx in sol.get_order_vessel_idx_for_factory(factory_node_id)]
        candidates.sort(key=lambda tup: tup[2], reverse=True)

        removals = 0
        while removals < remove_num and candidates:
            chosen_idx = int(pow(random.random(), self.determinism_param) * len(candidates))
            vessel, idx, _ = candidates.pop(chosen_idx)
            sol.remove_node(vessel, idx)
            removals += 1

            # Recompute removal candidates
            candidates = [(vessel, idx, sol.get_removal_utility(vessel, idx))
                          for vessel, idx in sol.get_order_vessel_idx_for_factory(factory_node_id)]
            candidates.sort(key=lambda tup: tup[2], reverse=True)

        sol.recompute_solution_variables()
        return sol

    def destroy_random(self, sol: Solution, remove_num: int) -> Solution:
        served_orders = [(vessel, order)
                         for vessel in sol.prbl.vessels
                         for order in sol.routes[vessel]
                         if not sol.prbl.nodes[order].is_factory]
        random.shuffle(served_orders)
        removals = 0
        while removals < remove_num and served_orders:
            vessel, order = served_orders.pop()
            remove_index = sol.routes[vessel].index(order)
            sol.remove_node(vessel, remove_index)
            removals += 1
        sol.recompute_solution_variables()
        return sol

    def destroy_voyage_random(self, sol: Solution, remove_num: int) -> Solution:
        # voyage: factory visit + orders until next factory visit
        illegal_removals = sol.get_illegal_voyage_removals()
        voyage_start_indexes = [(vessel, idx)
                                for vessel in sol.prbl.vessels
                                for idx in range(len(sol.routes[vessel]) - 1)
                                if sol.prbl.nodes[sol.routes[vessel][idx]].is_factory
                                and len(sol.routes[vessel]) > 1
                                and (vessel, idx) not in illegal_removals]
        random.shuffle(voyage_start_indexes)
        destroy_voyage_vessels = []
        orders_removed = 0
        while orders_removed < remove_num and voyage_start_indexes:
            vessel, voyage_start_idx = voyage_start_indexes.pop()  # pick one voyage to destroy
            if vessel in destroy_voyage_vessels:
                # avoid choosing several voyages from the same vessel, as this is messy with the voyage_start_indexes
                continue
            remove_indexes = [i for i in range(voyage_start_idx, sol.get_temp_voyage_end_idx(vessel, voyage_start_idx))
                              if i > 0]  # don't remove initial factory
            for idx in reversed(remove_indexes):
                sol.remove_node(vessel, idx)
                orders_removed += 1
            orders_removed -= 1  # to adjust for one factory being removed
            destroy_voyage_vessels.append(vessel)

        sol.recompute_solution_variables()
        return sol

    def destroy_voyage_worst(self, sol: Solution, remove_num: int) -> Solution:
        # voyage: factory visit + orders until next factory visit
        illegal_removals = sol.get_illegal_voyage_removals()
        voyage_start_indexes = [(vessel, idx, sol.get_voyage_profit(vessel, idx))
                                for vessel in sol.prbl.vessels
                                for idx in range(len(sol.routes[vessel]) - 1)
                                if sol.prbl.nodes[sol.routes[vessel][idx]].is_factory
                                and len(sol.routes[vessel]) > 1
                                and (vessel, idx) not in illegal_removals]
        voyage_start_indexes.sort(key=lambda item: item[2], reverse=True)
        destroy_voyage_vessels = []
        orders_removed = 0
        while orders_removed < remove_num and voyage_start_indexes:
            chosen_idx = int(pow(random.random(), self.determinism_param) * len(voyage_start_indexes))
            vessel, voyage_start_idx, _ = voyage_start_indexes.pop(chosen_idx)  # pick one voyage to destroy
            if vessel in destroy_voyage_vessels:
                # avoid choosing several voyages from the same vessel, as this is messy with the voyage_start_indexes
                continue
            remove_indexes = [i for i in range(voyage_start_idx, sol.get_temp_voyage_end_idx(vessel, voyage_start_idx))
                              if i > 0]  # don't remove initial factory
            for idx in reversed(remove_indexes):
                sol.remove_node(vessel, idx)
                orders_removed += 1
            orders_removed -= 1  # to adjust for one factory being removed
            destroy_voyage_vessels.append(vessel)

        sol.recompute_solution_variables()
        return sol

    def destroy_route_random(self, sol: Solution, remove_num: int) -> Solution:
        # choose a vessel that has a route
        illegal_removals = sol.get_illegal_route_removals()
        vessel = random.choice([vessel for vessel in sol.prbl.vessels
                                if len(sol.routes[vessel]) > 1
                                and vessel not in illegal_removals])
        route = sol.routes[vessel]
        for idx in range(len(route) - 1, 0, -1):
            sol.remove_node(vessel, idx)
        sol.recompute_solution_variables()
        return sol

    def destroy_route_worst(self, sol: Solution, remove_num: int) -> Solution:
        illegal_removals = sol.get_illegal_route_removals()
        route_profits = [(vessel, sol.get_route_profit(vessel))
                         for vessel in sol.prbl.vessels
                         if len(sol.routes[vessel]) > 1
                         and vessel not in illegal_removals]
        route_profits.sort(key=lambda item: item[1], reverse=True)  # descending order

        chosen_idx = int(pow(random.random(), self.determinism_param) * len(route_profits))
        vessel_for_worst_route = route_profits.pop(chosen_idx)[0]
        route = sol.routes[vessel_for_worst_route]
        for idx in range(len(route) - 1, 0, -1):
            sol.remove_node(vessel_for_worst_route, idx)
        sol.recompute_solution_variables()
        return sol

    def destroy_worst(self, sol: Solution, remove_num: int) -> Solution:
        candidates = True  # dummy initialization
        removals = 0
        while removals < remove_num and candidates:
            candidates = [(vessel, idx, sol.get_removal_utility(vessel, idx))
                          for vessel in sol.prbl.vessels
                          for idx, order in enumerate(sol.routes[vessel])
                          if not sol.prbl.nodes[order].is_factory]
            candidates.sort(key=lambda tup: tup[2], reverse=True)
            chosen_idx = int(pow(random.random(), self.determinism_param) * len(candidates))
            vessel, idx, _ = candidates.pop(chosen_idx)
            sol.remove_node(vessel, idx)
            removals += 1
        sol.recompute_solution_variables()
        return sol

    def _relatedness_location_time(self, order1: str, order2: str, vessel: str) -> float:
        w_0 = self.related_removal_weight_param['relatedness_location_time'][0]
        w_1 = self.related_removal_weight_param['relatedness_location_time'][1]

        time_window_difference = (abs(self.current_sol.prbl.nodes[order1].tw_start
                                      - self.current_sol.prbl.nodes[order2].tw_start) +
                                  abs(self.current_sol.prbl.nodes[order1].tw_end
                                      - self.current_sol.prbl.nodes[order2].tw_end))

        # Normalize difference with respect to n.o. time periods in problem
        time_window_difference /= self.current_sol.prbl.no_time_periods

        return w_0 * self.current_sol.prbl.transport_times_exact[vessel, order1, order2] + w_1 * time_window_difference

    def _relatedness_location_precedence(self, order1: str, order2: str, vessel: str) -> float:
        w_0 = self.related_removal_weight_param['relatedness_location_precedence'][0]
        w_1 = self.related_removal_weight_param['relatedness_location_precedence'][1]

        return (w_0 * self.current_sol.prbl.transport_times_exact[vessel, order1, order2] +
                w_1 * self.relatedness_precedence[(self.current_sol.prbl.nodes[order1].zone,
                                                   self.current_sol.prbl.nodes[order2].zone)])

    def destroy_related(self, sol: Solution, remove_num: int, rel_measure: function) -> Solution:
        # Related removal, based on inputted relatedness measure
        similar_orders: List[Tuple[str, int, str]] = []  # (order, index, vessel)
        served_orders = [(sol.routes[v][idx], idx, v)  # (order, index, vessel)
                         for v in sol.prbl.vessels
                         for idx in range(len(sol.routes[v]))
                         if not sol.prbl.nodes[sol.routes[v][idx]].is_factory]

        # Select a random first order
        random_first_idx = random.randint(0, len(served_orders) - 1)
        similar_orders.append(served_orders.pop(random_first_idx))

        while len(similar_orders) < remove_num and served_orders:
            base_order = similar_orders[random.randint(0, len(similar_orders) - 1)][0]
            candidates = [(order, idx, vessel, rel_measure(base_order, order, vessel))
                          for vessel in sol.prbl.vessels
                          for idx, order in enumerate(sol.routes[vessel])
                          if (order, idx, vessel) in served_orders]
            candidates.sort(key=lambda tup: tup[3])  # sort according to relatedness asc (most related orders first)

            chosen_related = candidates[int(pow(random.random(), self.determinism_param) * len(candidates))][:3]
            similar_orders.append(chosen_related)
            served_orders.remove(chosen_related)

        similar_orders.sort(key=lambda tup: tup[1], reverse=True)  # sort according to idx desc

        for (order, idx, vessel) in similar_orders:
            sol.remove_node(vessel=vessel, idx=idx)

        sol.recompute_solution_variables()
        return sol

    def repair(self, repair_op: str, sol: Solution, apply_noise: bool, use_ppfc: bool = False) -> Union[None, Solution]:
        if repair_op == "r_greedy":
            repaired_sol = self.repair_greedy(sol, apply_noise, use_ppfc)
        elif repair_op == "r_2regret":
            repaired_sol = self.repair_kregret(2, sol, apply_noise, use_ppfc)
        elif repair_op == 'r_3regret':
            repaired_sol = self.repair_kregret(3, sol, apply_noise, use_ppfc)
        else:
            print("Repair operator does not exist")
            return None

        if not use_ppfc:  # If PPFC was not used during repair, we remove orders until PPFC is satisfied
            repaired_sol = self.adjust_sol_ppfc(repaired_sol, self.remove_num_adjust, repair_op)

        return repaired_sol

    def repair_greedy(self, sol: Solution, apply_noise: bool = False, ppfc: bool = False) -> Solution:
        insertion_cand = sol.get_orders_not_served()
        insertions = [(node_id, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[node_id], vessel, idx,
                                                                       apply_noise * self.noise_param))
                      for node_id in insertion_cand
                      for vessel in sol.prbl.vessels
                      for idx in range(1, len(sol.routes[vessel]) + 1)]
        insertions.sort(key=lambda tup: tup[3])  # sort by gain

        while len(insertion_cand) > 0:  # try all possible insertions
            insert_node_id, vessel, idx, _ = insertions[-1]
            if sol.check_insertion_feasibility(insert_node_id, vessel, idx,
                                               noise_factor=apply_noise * self.noise_destination_param, ppfc=ppfc):
                sol.insert_last_checked()
                insertion_cand.remove(insert_node_id)
                # recalculate profit gain and omit other insertions of node_id
                insertions = [(node_id, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[node_id], vessel, idx,
                                                                               apply_noise * self.noise_param))
                              for node_id, vessel, idx, utility in insertions if node_id != insert_node_id]
                insertions.sort(key=lambda item: item[3])  # sort by gain

                if self.verbose:
                    sol.print_routes(highlight=[(vessel, idx)])
                    print()
            else:
                sol.clear_last_checked()
                insertions.pop()

                # Remove node_id from insertion candidate list if no insertions left for this node_id
                if len([insert for insert in insertions if insert[0] == insert_node_id]) == 0:
                    insertion_cand.remove(insert_node_id)
        return sol

    def repair_kregret(self, k: int, sol: Solution, apply_noise: bool = False, ppfc: bool = False) -> Solution:
        unrouted_orders = sol.get_orders_not_served()
        while unrouted_orders:
            # find the largest regret for each order and insert the largest regret.
            # repeat until no unrouted orders, or no feasible insertion
            repair_candidates: List[Tuple[str, int, str, float]] = []  # (order, idx, vessel, regret)

            for order in unrouted_orders:
                # Get all insertion utilities
                insertions = [(idx, v, sol.get_insertion_utility(sol.prbl.nodes[order], v, idx,
                                                                 apply_noise * self.noise_param))
                              for v in sol.prbl.vessels for idx in range(1, len(sol.routes[v]) + 1)]
                insertions.sort(key=lambda item: item[2])  # sort by gain

                insertion_regret = 0
                num_feasible = 0
                best_insertion: Tuple[str, int, str] = tuple()

                while num_feasible < k and len(insertions) > 0:
                    idx, vessel, util = insertions.pop()
                    feasible = sol.check_insertion_feasibility(order, vessel, idx,
                                                               noise_factor=apply_noise * self.noise_destination_param,
                                                               ppfc=ppfc)
                    sol.clear_last_checked()

                    if feasible and num_feasible == 0:
                        insertion_regret += (k - 1) * util  # inserting for k-1 terms in sum
                        best_insertion = (order, idx, vessel)
                        num_feasible += 1
                    elif feasible and num_feasible > 0:
                        insertion_regret -= util
                        num_feasible += 1

                if num_feasible < k:  # less than k feasible solutions found -> num_feasible * infinite regret
                    insertion_regret -= - (k - num_feasible) * int_inf

                if num_feasible >= 1 and best_insertion:  # at least one feasible solution found
                    node_id, idx, vessel = best_insertion
                    repair_candidates.append((node_id, idx, vessel, insertion_regret))

            if not repair_candidates:  # no orders to insert feasibly
                return sol

            # insert the greatest regret from repair_candidates
            node_id, idx, vessel, _ = max(repair_candidates, key=lambda item: item[3])
            sol.check_insertion_feasibility(node_id, vessel, idx,
                                            noise_factor=apply_noise * self.noise_destination_param)
            sol.insert_last_checked()
            unrouted_orders.remove(node_id)

            if self.verbose:
                sol.print_routes(highlight=[(vessel, idx)])
                print()

        return sol

    def accept_solution(self, sol: Solution) -> Tuple[bool, int, int]:  # accept, update_type, cost
        """
        :param sol: solution to be compared with self.current_sol
        :return: True if sol should be accepted to self.current_sol, else False, update_type and cost of new solution
        """
        sol_cost = sol.get_solution_routing_cost()

        # If routing solution within 5% of best routing solution, check production cost/feasibility
        if sol_cost < self.best_sol_routing_cost * (1 + self.percentage_best_solutions_production_solved):
            prod_cost = self.production_heuristic.get_cost(routing_sol=sol)
            accept = prod_cost < math.inf
            if not accept:
                return False, -2, sol_cost
            elif (accept and prod_cost + sol_cost < self.best_sol_total_cost and
                sol.get_solution_hash() not in self.previous_solutions):
                self.best_sol_total_cost = prod_cost + sol_cost
                self.num_best_total_sol_updates += 1
                return True, 0, sol_cost

        # if sol_cost < self.best_sol_cost:
        #     accept, _ = self.production_heuristic.is_feasible(sol)
        #     # exact_accept = self.production_model.is_feasible(sol)
        #     # if accept != exact_accept:
        #     #     print(f"Heuristic result ({accept})  deviates from exact ({exact_accept}). Sol cost: {sol_cost}")
        #     update_type = 0 if accept else -2  # -2 means new global best was not production feasible
        #     return accept, update_type, sol_cost
        if sol_cost < self.current_sol_cost:
            update_type = -1 if sol.get_solution_hash() in self.previous_solutions else 1
            return True, update_type, sol_cost
        elif self.iter_same_solution < self.max_iter_same_solution:
            # Simulated annealing criterion
            prob = pow(math.e, -((sol_cost - self.current_sol_cost) / self.temperature))
            accept = np.random.choice(np.array([True, False]), p=(np.array([prob, (1 - prob)])))
            update_type = -1 if sol.get_solution_hash() in self.previous_solutions else -1 + 3 * accept
            return accept, update_type, sol_cost
        else:  # maximum number of iterations from same current solution is reached, solution is accepted
            print("Solution is accepted because maximum was reached,", self.iter_same_solution)
            if sol.get_solution_hash() not in self.previous_solutions:
                self.record_solution(sol)  # operators not rewarded (negative accept type), even if solution is new
            return True, -3, sol_cost

    def record_solution(self, sol: Solution) -> None:
        if self.verbose:
            print(f"> Recording solution hash...")
        self.previous_solutions.add(sol.get_solution_hash())

    def run_alns_iteration(self) -> None:
        # Choose a destroy heuristic and a repair heuristic based on adaptive weights wdm
        # for method d in the current segment of iterations m
        d_op, r_op, noise_op = self.choose_operators()

        # Generate a candidate solution x′ from the current solution x using the chosen destroy and repair heuristics
        candidate_sol: Solution
        candidate_sol = self.generate_new_solution(destroy_op=d_op, repair_op=r_op, apply_noise=noise_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        accept_solution, self.update_type, cost = self.accept_solution(candidate_sol)

        if accept_solution:
            self.iter_same_solution = 0
            self.current_sol = candidate_sol
            self.current_sol_cost = cost
            if self.update_type > -1:  # if we except a solution that has not been accepted before
                self.record_solution(candidate_sol)
            if self.verbose:
                print(f'> Solution is accepted as current solution')
        else:
            self.iter_same_solution += 1

        # Update scores πd of the destroy and repair heuristics - dependent on how good the solution is
        self.update_scores(destroy_op=d_op, repair_op=r_op, noise_op=noise_op, update_type=self.update_type)

        # if IS iterations has passed since last weight update then
        # Update the weight wdm+1 for method d for the next segment m + 1 based on
        # the scores πd obtained for each method in segment m
        if self.it_seg_count == self.max_iter_seg:
            self.update_weights()

        # if f(x) > f(x∗) then
        # set x∗ =x
        if self.update_type == 0:  # type 0 means global best solution is found
            self.best_sol = self.current_sol
            self.best_sol_routing_cost = round(self.current_sol_cost)
            self.new_best_solution_feasible_production_count += 1
            if self.verbose:
                print(f'> Solution is accepted as best solution')
            # print("New best solutions' routing obj:", self.best_sol_routing_cost)

        if self.update_type == -2:  # type -2 means solution gave global best routing, but was not production feasible
            self.new_best_solution_infeasible_production_count += 1
            self.production_infeasibility_strike += 1
            if self.production_infeasibility_strike > self.production_infeasibility_strike_max:
                self.current_sol.ppfc_slack_factor += self.ppfc_slack_increment
        else:
            self.production_infeasibility_strike = 0
            self.current_sol.ppfc_slack_factor = 1.0

        # update iteration parameters
        self.temperature = self.temperature * self.cooling_rate
        self.it_seg_count += 1

    def write_to_file(self, excel_writer: pd.ExcelWriter, id: int, stop_criterion: str, alns_iter: int,
                      alns_time: int, parameter_tune: str = None, parameter_tune_value=None) -> None:
        cooling_rate = str(0.999) if alns_iter == 0 else str(round(math.pow(0.002, (1 / alns_iter)), 4))
        solution_dict = {'obj_val': round(self.best_sol_total_cost, 2),
                         'production_cost': round(self.best_sol_production_cost, 2),
                         'routing_cost': round(self.best_sol_routing_cost, 2),
                         'transport_cost': self.best_sol.get_solution_transport_cost(),
                         'num_orders_not_served': len(self.best_sol.get_orders_not_served()),
                         'stop_crierion': stop_criterion,
                         'num_iterations': alns_iter,
                         'time [sec]': alns_time,
                         'score_params': str(alnsparam.score_params),
                         'reaction_param': str(alnsparam.reaction_param),
                         'start_temperature_controlparam': str(alnsparam.start_temperature_controlparam),
                         'cooling_rate': cooling_rate,
                         'remove_percentage_interval': str(alnsparam.remove_percentage_interval),
                         'noise_param': str(alnsparam.noise_param),
                         'determinism_param': str(alnsparam.determinism_param),
                         'related_removal_weight_param': str(alnsparam.related_removal_weight_param),
                         'noise_destination_param': str(alnsparam.noise_destination_param)}

        if parameter_tune is not None:
            if parameter_tune == 'relatedness_location_time':
                solution_dict['related_removal_weight_param'] = str(parameter_tune_value) + str(
                    alnsparam.related_removal_weight_param['relatedness_location_precedence'])
            elif parameter_tune == 'relatedness_location_precedence':
                solution_dict['related_removal_weight_param'] = str(
                    alnsparam.related_removal_weight_param['relatedness_location_time']) + str(parameter_tune_value)
            else:
                solution_dict[parameter_tune] = str(parameter_tune_value)

        # TODO: Add relevant data
        if id == -1:
            id = "initial"
        sheet_name = "run_" + str(id)
        df = pd.DataFrame(solution_dict, index=[0]).transpose()
        df.to_excel(excel_writer, sheet_name=sheet_name, startrow=1)

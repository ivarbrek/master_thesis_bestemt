import math
import random
import numpy as np
from time import time
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import function
import joblib

from src.alns.solution import Solution, ProblemDataExtended
from src.models.production_model import ProductionModel
import src.util.plot as util

int_inf = 999999


class Alns:
    it_seg_count: int
    best_sol: Solution
    best_sol_cost: int
    current_sol: Solution
    current_sol_cost: int
    production_model: ProductionModel
    inventory_reward: bool
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
    determinism_param: int
    relatedness_precedence: Dict[Tuple[str, str], int]
    related_removal_weight_param: Dict[str, List[float]]
    new_best_solution_feasible_production_count = 0
    new_best_solution_infeasible_production_count = 0
    new_best_solution_infeasible_revisited_count = 0
    production_problems_infeasible_cache = defaultdict(lambda: False)
    production_infeasibility_strike = 0
    production_infeasibility_strike_max: int

    def __init__(self, problem_data: ProblemDataExtended,
                 destroy_op: List[str],
                 repair_op: List[str],
                 weight_min_threshold: float,
                 reaction_param: float,
                 score_params: List[int],
                 start_temperature_controlparam: float,
                 cooling_rate: float,
                 max_iter_seg: int,
                 remove_percentage: float,
                 determinism_param: int,
                 noise_param: float,
                 relatedness_precedence: Dict[Tuple[str, str], int],
                 related_removal_weight_param: Dict[str, List[float]],
                 inventory_reward: bool,
                 production_infeasibility_strike_max: int,
                 ppfc_slack_increment: float,
                 verbose: bool = False) -> None:

        self.verbose = verbose

        # ALNS  parameters
        self.max_iter_seg = max_iter_seg
        self.remove_num = round(remove_percentage * len(problem_data.order_nodes))
        self.determinism_param = determinism_param
        self.noise_param = noise_param

        # Solutions
        self.current_sol = self.repair_kregret(2, Solution(problem_data))
        self.production_model = ProductionModel(prbl=problem_data,
                                                demands=self.current_sol.get_demand_dict(),
                                                inventory_reward_extension=inventory_reward)
        self.best_sol_production_cost = self.adjust_curr_sol_to_prod_feasibility()  # Make sure initial solution is production-feasible
        self.current_sol_cost = self.current_sol.get_solution_routing_cost()
        self.best_sol = self.current_sol
        self.best_sol_cost = self.current_sol_cost

        # Operator weights, scores and usage
        self.weight_min_threshold = weight_min_threshold
        self.destroy_op_weight = {op: self.weight_min_threshold for op in destroy_op}
        self.repair_op_weight = {op: self.weight_min_threshold for op in repair_op}
        self.destroy_op_score = {op: 0 for op in destroy_op}
        self.repair_op_score = {op: 0 for op in repair_op}
        self.destroy_op_segment_usage = {op: 0 for op in destroy_op}
        self.repair_op_segment_usage = {op: 0 for op in repair_op}
        self.reaction_param = reaction_param
        self.score_params = {i: score_params[i] for i in range(len(score_params))}

        self.cooling_rate = cooling_rate
        self.temperature = -(self.best_sol_cost * start_temperature_controlparam) / math.log(0.5)

        self.it_seg_count = 0  # Iterations done in one segment - can maybe do this in run_alns_iteration?

        self.production_infeasibility_strike_max = production_infeasibility_strike_max
        self.ppfc_slack_increment = ppfc_slack_increment

        # Relatedness operator parameters
        self.related_removal_weight_param = related_removal_weight_param
        if problem_data.precedence:
            self.relatedness_precedence = self.set_relatedness_precedence_dict(relatedness_precedence)

        # TODO: Delete these when done with testing
        # self.rel_components = {'loc_time_amount__transport_time': 0, 'loc_time_amount__time_window': 0,
        #                        'loc_time_usage': 0,
        #                        'loc_prec_amount__transport_time': 0, 'loc_prec_amount__prec': 0,
        #                        'loc_prec_usage': 0}

    def __repr__(self):
        destroy_op = [(k, round(v, 2)) for k, v in sorted(self.destroy_op_weight.items(), key=lambda item: item[0])]
        repair_op = [(k, round(v, 2)) for k, v in sorted(self.repair_op_weight.items(), key=lambda item: item[0])]
        return (
            f"Best solution with routing cost {self.best_sol_cost}: \n"
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
        if len(zones) < len(self.current_sol.prbl.get_zones()):
            print("Relatedness of zones is not properly defined.")
        return d


    def adjust_curr_sol_to_prod_feasibility(self) -> float:
        production_cost = self.current_sol.get_production_cost(self.production_model)
        while production_cost == math.inf:
            self.current_sol = self.destroy_worst(self.current_sol)
            production_cost = self.current_sol.get_production_cost(self.production_model)
        return production_cost

    def update_scores(self, destroy_op: str, repair_op: str, update_type: int) -> None:
        if update_type == -1:
            return
        if not 0 <= update_type < len(self.score_params):
            print("Update type does not exist. Scores not updated.")
            return
        self.destroy_op_score[destroy_op] += self.score_params[update_type]
        self.repair_op_score[repair_op] += self.score_params[update_type]

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

        self.it_seg_count = 0

    def choose_operators(self) -> Tuple[str, str]:
        # Choose operators, probability of choosing an operator is based on current weights
        destroy_operators = list(zip(*self.destroy_op_weight.items()))
        destroy_op: str = np.random.choice(
            np.array([op for op in destroy_operators[0]]),
            p=(np.array([w for w in destroy_operators[1]]) / np.array(sum(destroy_operators[1]))))
        self.destroy_op_segment_usage[destroy_op] += 1

        repair_operators = list(zip(*self.repair_op_weight.items()))
        repair_op: str = np.random.choice(
            np.array([op for op in repair_operators[0]]),
            p=(np.array([w for w in repair_operators[1]]) / np.array(sum(repair_operators[1]))))
        self.repair_op_segment_usage[repair_op] += 1
        print(destroy_op, repair_op)
        return destroy_op, repair_op

    def generate_new_solution(self, destroy_op: str, repair_op: str) -> Solution:
        # Generate new feasible solution x' based on given operators
        # Return x'
        candidate_sol = self.current_sol.copy()
        candidate_sol = self.destroy(destroy_op, candidate_sol)
        candidate_sol = self.repair(repair_op, candidate_sol)
        return candidate_sol

    def destroy(self, destroy_op: str, sol: Solution) -> Union[Solution, None]:
        # Run function based on operator "ID"
        if destroy_op == "d_random":
            return self.destroy_random(sol)
        elif destroy_op == "d_worst":
            return self.destroy_worst(sol)
        elif destroy_op == "d_related_location_time":
            return self.destroy_related(sol, rel_measure=self._relatedness_location_time)
        elif destroy_op == "d_related_location_precedence":
            return self.destroy_related(sol, rel_measure=self._relatedness_location_precedence)
        elif destroy_op == "d_voyage_random":
            return self.destroy_voyage_random(sol)
        elif destroy_op == "d_voyage_worst":
            return self.destroy_voyage_worst(sol)
        elif destroy_op == "d_route_random":
            return self.destroy_route_random(sol)
        elif destroy_op == "d_route_worst":
            return self.destroy_route_worst(sol)
        else:
            print("Destroy operator does not exist")
            return None

    def destroy_random(self, sol: Solution) -> Solution:
        served_orders = [(vessel, order)
                         for vessel in sol.prbl.vessels
                         for order in sol.routes[vessel]
                         if not sol.prbl.nodes[order].is_factory]
        random.shuffle(served_orders)
        removals = 0
        while removals < self.remove_num and served_orders:
            vessel, order = served_orders.pop()
            remove_index = sol.routes[vessel].index(order)
            sol.remove_node(vessel, remove_index)
            removals += 1
        sol.recompute_solution_variables()
        return sol

    def destroy_voyage_random(self, sol: Solution) -> Solution:
        # voyage: factory visit + orders until next factory visit
        voyage_start_indexes = [(vessel, idx)
                                for vessel in sol.prbl.vessels
                                for idx in range(len(sol.routes[vessel]) - 1)
                                if sol.prbl.nodes[sol.routes[vessel][idx]].is_factory
                                and len(sol.routes[vessel]) > 1]
        random.shuffle(voyage_start_indexes)
        destroy_voyage_vessels = []
        orders_removed = 0
        while orders_removed < self.remove_num and voyage_start_indexes:
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

    def destroy_voyage_worst(self, sol: Solution) -> Solution:
        # voyage: factory visit + orders until next factory visit
        voyage_start_indexes = [(vessel, idx, sol.get_voyage_profit(vessel, idx))
                                for vessel in sol.prbl.vessels
                                for idx in range(len(sol.routes[vessel]) - 1)
                                if sol.prbl.nodes[sol.routes[vessel][idx]].is_factory
                                and len(sol.routes[vessel]) > 1]
        voyage_start_indexes.sort(key=lambda item: item[2], reverse=True)
        destroy_voyage_vessels = []
        orders_removed = 0
        while orders_removed < self.remove_num and voyage_start_indexes:
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

    def destroy_route_random(self, sol: Solution) -> Solution:
        # choose a vessel that has a route
        vessel = random.choice([vessel for vessel in sol.prbl.vessels if len(sol.routes[vessel]) > 1])
        route = sol.routes[vessel]
        for idx in range(len(route) - 1, 0, -1):
            sol.remove_node(vessel, idx)
        sol.recompute_solution_variables()
        return sol

    def destroy_route_worst(self, sol: Solution) -> Solution:
        route_profits = [(vessel, sol.get_route_profit(vessel))
                         for vessel in sol.prbl.vessels
                         if len(sol.routes[vessel]) > 1]
        route_profits.sort(key=lambda item: item[1], reverse=True)  # descending order

        chosen_idx = int(pow(random.random(), self.determinism_param) * len(route_profits))
        vessel_for_worst_route = route_profits.pop(chosen_idx)[0]
        route = sol.routes[vessel_for_worst_route]
        for idx in range(len(route) - 1, 0, -1):
            sol.remove_node(vessel_for_worst_route, idx)
        sol.recompute_solution_variables()
        return sol

    def destroy_worst(self, sol: Solution) -> Solution:
        candidates = True  # dummy initialization
        removals = 0
        while removals < self.remove_num and candidates:
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

    def _relatedness_location_time(self, order1: str, order2: str) -> float:
        w_0 = self.related_removal_weight_param['relatedness_location_time'][0]
        w_1 = self.related_removal_weight_param['relatedness_location_time'][1]

        time_window_difference = (abs(self.current_sol.prbl.nodes[order1].tw_start
                                      - self.current_sol.prbl.nodes[order2].tw_start) +
                                  abs(self.current_sol.prbl.nodes[order1].tw_end
                                      - self.current_sol.prbl.nodes[order2].tw_end))

        # # Used for weight tuning
        # self.rel_components['loc_time_amount__transport_time'] += \
        #     w_0 * self.current_sol.prbl.transport_times[order1, order2]
        # self.rel_components['loc_time_amount__time_window'] += \
        #     w_1 * time_window_difference
        # self.rel_components['loc_time_usage'] += 1

        return w_0 * self.current_sol.prbl.transport_times[order1, order2] + w_1 * time_window_difference

    def _relatedness_location_precedence(self, order1: str, order2: str) -> float:
        w_0 = self.related_removal_weight_param['relatedness_location_precedence'][0]
        w_1 = self.related_removal_weight_param['relatedness_location_precedence'][1]

        # # Used for weight tuning
        # self.rel_components['loc_prec_amount__transport_time'] += w_0 * self.current_sol.prbl.transport_times[
        #     order1, order2]
        # self.rel_components['loc_prec_amount__prec'] += \
        #     w_1 * self.relatedness_precedence[(self.current_sol.prbl.nodes[order1].zone,
        #                                        self.current_sol.prbl.nodes[order2].zone)]
        # self.rel_components['loc_prec_usage'] += 1

        return (w_0 * self.current_sol.prbl.transport_times[order1, order2] +
                w_1 * self.relatedness_precedence[(self.current_sol.prbl.nodes[order1].zone,
                                                   self.current_sol.prbl.nodes[order2].zone)])

    def destroy_related(self, sol: Solution, rel_measure: function) -> Solution:
        # Related removal, based on inputted relatedness measure
        similar_orders: List[Tuple[str, int, str]] = []  # (order, index, vessel)
        served_orders = [(sol.routes[v][idx], idx, v)  # (order, index, vessel)
                         for v in sol.prbl.vessels
                         for idx in range(len(sol.routes[v]))
                         if not sol.prbl.nodes[sol.routes[v][idx]].is_factory]

        # Select a random first order
        random_first_idx = random.randint(0, len(served_orders) - 1)
        similar_orders.append(served_orders.pop(random_first_idx))

        while len(similar_orders) < self.remove_num and served_orders:
            base_order = similar_orders[random.randint(0, len(similar_orders) - 1)][0]
            candidates = [(order, idx, vessel, rel_measure(base_order, order))
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

    def repair(self, repair_op: str, sol: Solution) -> Union[None, Solution]:
        if repair_op == "r_greedy":
            return self.repair_greedy(sol)
        elif repair_op == "r_2regret":
            return self.repair_kregret(2, sol)
        elif repair_op == 'r_3regret':
            return self.repair_kregret(3, sol)
        else:
            print("Repair operator does not exist")
            return None

    def repair_greedy(self, sol: Solution) -> Solution:
        insertion_cand = sol.get_orders_not_served()
        insertions = [(node_id, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[node_id], vessel, idx, 
                                                                       self.noise_param))
                      for node_id in insertion_cand
                      for vessel in sol.prbl.vessels
                      for idx in range(1, len(sol.routes[vessel]) + 1)]
        insertions.sort(key=lambda tup: tup[3])  # sort by gain

        while len(insertion_cand) > 0:  # try all possible insertions
            insert_node, vessel, idx, _ = insertions[-1]
            if sol.check_insertion_feasibility(insert_node, vessel, idx):
                sol.insert_last_checked()
                insertion_cand.remove(insert_node)
                # recalculate profit gain and omit other insertions of node_id
                insertions = [(node, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[insert_node], vessel, idx, 
                                                                            self.noise_param))
                              for node, vessel, idx, utility in insertions if node != insert_node]
                insertions.sort(key=lambda item: item[3])  # sort by gain

                if self.verbose:
                    sol.print_routes(highlight=[(vessel, idx)])
                    print()
            else:
                sol.clear_last_checked()
                insertions.pop()

                # Remove node_id from insertion candidate list if no insertions left for this node_id
                if len([insert for insert in insertions if insert[0] == insert_node]) == 0:
                    insertion_cand.remove(insert_node)
        return sol

    def repair_kregret(self, k: int, sol: Solution) -> Solution:
        unrouted_orders = sol.get_orders_not_served()
        while unrouted_orders:
            # find the largest regret for each order and insert the largest regret.
            # repeat until no unrouted orders, or no feasible insertion
            repair_candidates: List[Tuple[str, int, str, float]] = []  # (order, idx, vessel, regret)

            for order in unrouted_orders:
                # Get all insertion utilities
                insertions = [(idx, v, sol.get_insertion_utility(sol.prbl.nodes[order], v, idx, self.noise_param))
                              for v in sol.prbl.vessels for idx in range(1, len(sol.routes[v]) + 1)]
                insertions.sort(key=lambda item: item[2])  # sort by gain

                insertion_regret = 0
                num_feasible = 0
                best_insertion: Tuple[str, int, str] = tuple()

                while num_feasible < k and len(insertions) > 0:
                    idx, vessel, util = insertions.pop()
                    feasible = sol.check_insertion_feasibility(order, vessel, idx)
                    sol.clear_last_checked()

                    if feasible and num_feasible == 0:
                        insertion_regret += util
                        best_insertion = (order, idx, vessel)
                        num_feasible += 1
                    elif feasible and num_feasible == 1:
                        insertion_regret -= util
                        num_feasible += 1

                if num_feasible < k:  # less than k feasible solutions found -> num_feasible * infinite regret
                    insertion_regret -= - num_feasible * int_inf

                if num_feasible >= 1 and best_insertion:  # at least one feasible solution found
                    node_id, idx, vessel = best_insertion
                    repair_candidates.append((node_id, idx, vessel, insertion_regret))

            if not repair_candidates:  # no orders to insert feasibly
                return sol

            # insert the greatest regret from repair_candidates
            node_id, idx, vessel, _ = max(repair_candidates, key=lambda item: item[3])
            sol.check_insertion_feasibility(node_id, vessel, idx)
            sol.insert_last_checked()
            unrouted_orders.remove(node_id)

            if self.verbose:
                sol.print_routes(highlight=[(vessel, idx)])
                print()

        return sol

    def accept_solution(self, sol: Solution) -> Tuple[bool, int, int]:  # accept, accept_type, cost
        """
        :param sol: solution to be compared with self.current_sol
        :return: True if sol should be accepted to self.current_sol, else False
        """
        sol_cost = sol.get_solution_routing_cost()
        if sol_cost < self.best_sol_cost:
            return True, 0, sol_cost
        elif sol_cost < self.current_sol_cost:
            return True, 1, sol_cost
        else:
            # Simulated annealing criterion
            prob = pow(math.e, -((sol_cost - self.current_sol_cost) / self.temperature))
            accept = np.random.choice(np.array([True, False]), p=(np.array([prob, (1 - prob)])))
            return accept, -1 + 3 * accept, sol_cost  # cost only used if accept=True

    def run_alns_iteration(self) -> None:
        # Choose a destroy heuristic and a repair heuristic based on adaptive weights wdm
        # for method d in the current segment of iterations m
        d_op, r_op = self.choose_operators()

        # Generate a candidate solution x′ from the current solution x using the chosen destroy and repair heuristics
        candidate_sol: Solution
        candidate_sol = self.generate_new_solution(destroy_op=d_op, repair_op=r_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        accept_solution, update_type, cost = self.accept_solution(candidate_sol)

        production_cost = None
        if accept_solution:
            self.current_sol = candidate_sol
            self.current_sol_cost = cost
            if self.verbose:
                print(f'> Solution is accepted as current solution')

            # If in a production infeasibility strike, check if new current solution is production feasible
            if self.production_infeasibility_strike >= 1:
                production_cost = self.current_sol.get_production_cost(pp_model=self.production_model)
                if production_cost < math.inf:
                    self.production_infeasibility_strike = 0
                    self.current_sol.ppfc_slack_factor = 1
                    if self.verbose:
                        print(f"PPFC slack factor reset to {self.current_sol.ppfc_slack_factor}")
                else:
                    self.production_infeasibility_strike += 1

                    # Increase slack requirement in PPFC to avoid further infeasibility
                    if self.production_infeasibility_strike >= self.production_infeasibility_strike_max:
                        self.current_sol.ppfc_slack_factor *= (1 + self.ppfc_slack_increment)
                        if self.verbose:
                            print(f"Increasing PPFC slack factor by {self.ppfc_slack_increment * 100}%, "
                                  f"to {self.current_sol.ppfc_slack_factor}")

        # Update scores πd of the destroy and repair heuristics - dependent on how good the solution is
        self.update_scores(destroy_op=d_op, repair_op=r_op, update_type=update_type)

        # if IS iterations has passed since last weight update then
        # Update the weight wdm+1 for method d for the next segment m + 1 based on
        # the scores πd obtained for each method in segment m
        if self.it_seg_count == self.max_iter_seg:
            self.update_weights()

        # if f(x) > f(x∗) then
        # set x∗ =x
        if update_type == 0:  # type 0 means global best solution is found
            if self.production_problems_infeasible_cache[joblib.hash(self.current_sol.routes)]:
                production_cost = math.inf
            elif not production_cost:
                production_cost = self.current_sol.get_production_cost(self.production_model)

            if production_cost < math.inf:
                self.best_sol = self.current_sol
                self.best_sol_cost = self.current_sol_cost
                self.best_sol_production_cost = production_cost
                self.new_best_solution_feasible_production_count += 1
                if self.verbose:
                    print(f'> Solution is accepted as best solution')
                print("New best solutions' total obj:", round(production_cost) + self.best_sol_cost)

            else:
                self.production_problems_infeasible_cache[joblib.hash(self.current_sol.routes)] = True
                self.new_best_solution_infeasible_production_count += 1
                if self.production_infeasibility_strike == 0:  # initiate infeasibility strike
                    self.production_infeasibility_strike += 1
                    if self.production_infeasibility_strike >= self.production_infeasibility_strike_max:
                        self.current_sol.ppfc_slack_factor *= (1 + self.ppfc_slack_increment)
                        if self.verbose:
                            print(f"Increasing PPFC slack factor by {self.ppfc_slack_increment * 100}%, "
                                  f"to {self.current_sol.ppfc_slack_factor}")

        # update iteration parameters
        self.temperature = self.temperature * self.cooling_rate
        self.it_seg_count += 1


if __name__ == '__main__':
    precedence: bool = False

    prbl = ProblemDataExtended('../../data/input_data/larger_testcase2.xlsx', precedence=precedence)
    destroy_op = ['d_random',
                  'd_worst',
                  'd_voyage_random',
                  'd_voyage_worst',
                  'd_route_random',
                  'd_route_worst',
                  'd_related_location_time']
    destroy_op += ['d_related_location_precedence'] if precedence else []

    print()
    print("ALNS starting...")
    alns = Alns(problem_data=prbl,
                destroy_op=destroy_op,
                repair_op=['r_greedy', 'r_2regret', 'r_3regret'],
                weight_min_threshold=0.2,
                reaction_param=0.1,
                score_params=[5, 3, 1],  # corresponding to sigma_1, sigma_2, sigma_3 in R&P and L&N
                start_temperature_controlparam=0.4,  # solution 50% worse than best solution is accepted with 50% prob.
                cooling_rate=0.995,
                max_iter_seg=40,
                remove_percentage=0.4,
                determinism_param=5,
                noise_param=200,
                relatedness_precedence={('green', 'yellow'): 6, ('green', 'red'): 10, ('yellow', 'red'): 4},
                related_removal_weight_param={'relatedness_location_time': [1, 0.9],
                                              'relatedness_location_precedence': [0.25, 1]},
                production_infeasibility_strike_max=1,
                ppfc_slack_increment=0.1,  # increase slack factor by 10% after x subsequent production infeasibilities
                inventory_reward=False,
                verbose=False
                )

    print("Route after initialization")
    alns.current_sol.print_routes()
    print("\nRemove num:", alns.remove_num, "\n")

    iterations = 1000

    _stat_solution_cost = []
    _stat_operator_weights = defaultdict(list)
    t0 = time()
    for i in range(iterations):
        print("Iteration", i)
        alns.run_alns_iteration()

        alns.current_sol.print_routes()
        print(f"Obj: {alns.current_sol_cost}   Not served: {alns.current_sol.get_orders_not_served()}")

        _stat_solution_cost.append((i, alns.current_sol_cost))
        for op, score in alns.destroy_op_weight.items():
            _stat_operator_weights[op].append(score)
        for op, score in alns.repair_op_weight.items():
            _stat_operator_weights[op].append(score)
        print()

    print()
    print(f"...ALNS terminating  ({round(time() - t0)}s)")
    alns.best_sol.print_routes()
    print("Not served:", alns.best_sol.get_orders_not_served())

    print("Routing obj:", alns.best_sol_cost, "Prod obj:", round(alns.best_sol_production_cost, 1),
          "Total:", alns.best_sol_cost + round(alns.best_sol_production_cost, 1))

    print(f"Best solution updated {alns.new_best_solution_feasible_production_count} times")
    print(f"Candidate to become best solution rejected {alns.new_best_solution_infeasible_production_count} times, "
          f"because of production infeasibility, where cache was used "
          f"{alns.new_best_solution_infeasible_revisited_count} times")

    util.plot_alns_history(_stat_solution_cost)
    util.plot_operator_weights(_stat_operator_weights)

import math
import random
from typing import Dict, List, Tuple, Union

import function
import pyomo.environ as pyo

from src.alns.solution import Solution, ProblemDataExtended
from src.models.production_model import ProductionModel
import numpy as np
from src.util.plot import plot_alns_history, plot_alns_history_with_production_feasibility

int_inf = 999999


# What to do on ALNS:
#
# General
# [x] Construct initial solution
# [x] Update solution (e and l) after removal.
# [x] Calculate and update scores and weights
# DROP THIS: Check that solution is feasible after removal (vessel load and n.o. products, production constraints)
# [x] Calculate and update scores and weights
#
# Operators
# [/] Greedy insertion
# [/] Regret insertion
# [/] Random removal
# [/] Worst removal (objective, longest wait?)
# [/] Related removal (distance, time window, shaw)
# [ ] Voyage removal


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
    unrouted_orders: List[str] = []  # TODO: Remove field? Not currently in use.
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
                 relatedness_precedence: Dict[Tuple[str, str], int],
                 related_removal_weight_param: Dict[str, List[float]],
                 inventory_reward: bool,
                 verbose: bool = False) -> None:

        # ALNS  parameters
        self.max_iter_seg = max_iter_seg
        self.remove_num = round(remove_percentage * len(problem_data.order_nodes))

        # Solutions and production problem
        self.current_sol = self.construct_initial_solution(problem_data)
        self.production_model = ProductionModel(prbl=problem_data,
                                                demands=self.current_sol.get_demand_dict(),
                                                inventory_reward_extension=inventory_reward)
        self.adjust_initial_sol(verbose=verbose)  # Make sure initial solution is production-feasible
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
        self.temperature = -(self.best_sol_cost * start_temperature_controlparam) / math.log2(0.5)

        self.it_seg_count = 0  # Iterations done in one segment - can maybe do this in run_alns_iteration?
        self.insertion_candidates = self.current_sol.get_orders_not_served()

        # Relatedness operator parameters
        self.determinism_param = determinism_param
        self.relatedness_precedence = self.set_relatedness_precedence_dict(
            relatedness_precedence) if problem_data.precedence else None
        self.related_removal_weight_param = related_removal_weight_param

        self.verbose = verbose

        # TODO: Delete these when done with testing
        # self.rel_components = {'loc_time_amount__transport_time': 0, 'loc_time_amount__time_window': 0,
        #                        'loc_time_usage': 0,
        #                        'loc_prec_amount__transport_time': 0, 'loc_prec_amount__prec': 0,
        #                        'loc_prec_usage': 0}

    def __repr__(self):
        return (
            f"Best solution with routing cost {self.best_sol_cost}: \n"
            f"{self.best_sol} \n"
            # f"Current solution with routing cost {self.current_sol_cost}: \n"
            # f"{self.current_sol} \n"
            f"Insertion candidates (orders not served): {self.insertion_candidates} \n"
            f"Destroy operators {[(k, round(v, 2)) for k, v in self.destroy_op_weight.items()]} \n"
            f"and repair operators {[(k, round(v, 2)) for k, v in self.repair_op_weight.items()]} \n")

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

    @staticmethod
    def construct_initial_solution(problem_data: ProblemDataExtended) -> Solution:
        sol: Solution = Solution(problem_data)
        unrouted_orders = list(sol.prbl.order_nodes.keys())
        unrouted_order_cost = [sol.prbl.external_delivery_penalties[o] for o in unrouted_orders]
        unrouted_orders = [o for _, o in sorted(zip(unrouted_order_cost, unrouted_orders),
                                                key=lambda pair: pair[0], reverse=True)]  # desc according to cost

        for o in unrouted_orders:
            insertion_gain = []
            insertion: List[Tuple[int, str]] = []
            for v in sol.prbl.vessels:
                for idx in range(1, len(sol.routes[v]) + 1):
                    insertion_gain.append(sol.get_insertion_utility(node=sol.prbl.nodes[o],
                                                                    idx=idx, vessel=v))
                    insertion.append((idx, v))
            insertion = [ins for _, ins in sorted(zip(insertion_gain, insertion),
                                                  key=lambda pair: pair[0], reverse=True)]
            feasible = False
            while not feasible and len(insertion) > 0:
                idx, vessel = insertion[0]
                feasible = sol.check_insertion_feasibility(o, vessel, idx)
                if feasible:
                    sol.insert_last_checked()
                else:
                    sol.clear_last_checked()
                insertion.pop(0)
        return sol

    def adjust_initial_sol(self, verbose: bool = False) -> None:
        self.production_model.solve(verbose=verbose)
        while self.production_model.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            if verbose:
                print(f"\nDestroying order nodes to find production feasible initial solution \n")
            self.current_sol = self.destroy_random(self.current_sol)
            self.production_model.reconstruct_demand(new_demands=self.current_sol.get_demand_dict())
            self.production_model.solve(verbose=verbose)

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
                                                  self.destroy_op_segment_usage[op]), self.weight_min_threshold)
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

    def generate_new_solution(self, destroy_op: str, repair_op: str) -> Tuple[Solution, List[str]]:
        # Generate new feasible solution x' based on given operators
        # Return x'
        candidate_sol = self.current_sol.copy()
        candidate_sol = self.destroy(destroy_op, candidate_sol)
        candidate_sol, insertion_candidates = self.repair(repair_op, candidate_sol)
        return candidate_sol, insertion_candidates

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
        else:
            print("Destroy operator does not exist")
            return None

    def destroy_random(self, sol: Solution) -> Solution:
        served_orders = [(v, o) for v in sol.prbl.vessels
                         for o in sol.routes[v] if not sol.prbl.nodes[o].is_factory]
        random.shuffle(served_orders)
        removals = 0
        while removals < self.remove_num and served_orders:
            vessel, order = served_orders.pop()
            remove_index = sol.routes[vessel].index(order)
            sol.remove_node(vessel, remove_index)
            removals += 1

        sol.recompute_solution_variables()
        return sol

    def destroy_worst(self, sol: Solution) -> Solution:
        served_orders = [node for v in sol.prbl.vessels
                         for node in sol.routes[v]
                         if not sol.prbl.nodes[node].is_factory]
        removals = 0

        while removals < self.remove_num and served_orders:

            candidates: List[Tuple[str, int, str, float]] = []  # order, idx, vessel, utility
            for v in sol.prbl.vessels:
                route = sol.routes[v]
                for idx in range(len(route)):
                    node = sol.prbl.nodes[route[idx]]
                    if not node.is_factory:
                        candidates.append((node.id, idx, v, sol.get_removal_utility(vessel=v, idx=idx)))

            if not candidates:  # no orders to remove
                return sol

            candidates.sort(key=lambda tup: tup[3], reverse=True)  # sort according to utility desc
            chosen_worst = (candidates[int(pow(random.random(), self.determinism_param)
                                           * len(candidates))])[:3]  # (order, idx, vessel)

            served_orders.remove(chosen_worst[0])
            sol.remove_node(vessel=chosen_worst[2], idx=chosen_worst[1])
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

        return (w_0 * self.current_sol.prbl.transport_times[order1, order2] +
                w_1 * time_window_difference)

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
        served_orders = [(sol.routes[v][idx], idx, v) for v in sol.prbl.vessels  # (order, index, vessel)
                         for idx in range(len(sol.routes[v]))
                         if not sol.prbl.nodes[sol.routes[v][idx]].is_factory]

        # Select a random first order
        random_first_idx = random.randint(0, len(served_orders) - 1)
        similar_orders.append(served_orders[random_first_idx])
        served_orders.pop(random_first_idx)

        while len(similar_orders) < self.remove_num and served_orders:
            base_order = similar_orders[random.randint(0, len(similar_orders) - 1)][0]

            candidates: List[Tuple[str, int, str, float]] = []  # tuple: (order, idx, vessel, relatedness)
            for v in sol.prbl.vessels:
                route = sol.routes[v]
                for idx in range(len(route)):
                    order = route[idx]
                    if (order, idx, v) in served_orders:
                        candidates.append((order, idx, v, rel_measure(base_order, order)))
            candidates.sort(key=lambda tup: tup[3])  # sort according to relatedness asc (most related orders first)

            chosen_related = candidates[int(pow(random.random(), self.determinism_param) * len(candidates))][:3]
            similar_orders.append(chosen_related)
            served_orders.remove(chosen_related)

        similar_orders.sort(key=lambda tup: tup[1], reverse=True)  # sort according to idx desc

        for (order, idx, vessel) in similar_orders:
            sol.remove_node(vessel=vessel, idx=idx)

        sol.recompute_solution_variables()
        return sol

    def repair(self, repair_op: str, sol: Solution) -> Union[None, Tuple[Solution, List[str]]]:
        if repair_op == "r_greedy":
            return self.repair_greedy(sol)
        elif repair_op == "r_2regret":
            return self.repair_2regret(sol)
        else:
            print("Repair operator does not exist")
            return None

    def repair_greedy(self, sol: Solution) -> Tuple[Solution, List[str]]:
        insertion_cand = sol.get_orders_not_served()
        insertions = [(node_id, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[node_id], vessel, idx))
                      for node_id in insertion_cand
                      for vessel in sol.prbl.vessels
                      for idx in range(1, len(sol.routes[vessel]) + 1)]
        insertions.sort(key=lambda tup: tup[3])  # sort by gain

        unrouted_orders = []

        while len(insertion_cand) > 0:  # try all possible insertions
            insert_node, vessel, idx, _ = insertions[-1]
            if sol.check_insertion_feasibility(insert_node, vessel, idx):
                sol.insert_last_checked()
                insertion_cand.remove(insert_node)
                # recalculate profit gain and omit other insertions of node_id
                insertions = [(node, vessel, idx, sol.get_insertion_utility(sol.prbl.nodes[insert_node], vessel, idx))
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
                    unrouted_orders.append(insert_node)
        return sol, unrouted_orders

    def repair_2regret(self, sol: Solution) -> Tuple[Solution, List[str]]:
        for i in range(self.remove_num):  # TODO: Find out how to do varying q^{ALNS}
            insertion_cand = sol.get_orders_not_served()
            insertions: List[Tuple[int, str, float]] = []  # tuple (idx, vessel, gain)
            repair_candidates: List[Tuple[str, int, str, float]] = []  # (order, idx, vessel, regret)

            for o in insertion_cand:
                # Get all insertion utilities
                for v in sol.prbl.vessels:
                    for idx in range(1, len(sol.routes[v]) + 1):
                        insertions.append((idx, v,
                                           sol.get_insertion_utility(node=sol.prbl.nodes[o], vessel=v, idx=idx)))

                insertions.sort(key=lambda item: item[2], reverse=True)  # sort by gain, order desc

                insertion_regret = 0
                num_feasible = 0
                best_insertion: Tuple[str, int, str]

                while num_feasible < 2 and len(insertions) > 0:
                    idx, vessel, util = insertions[0]
                    feasible = sol.check_insertion_feasibility(o, vessel, idx)
                    if feasible:
                        if num_feasible == 0:
                            insertion_regret += util
                            best_insertion = (o, idx, vessel)
                        elif num_feasible == 1:
                            insertion_regret -= util
                        num_feasible += 1

                    sol.clear_last_checked()
                    insertions.pop(0)

                if num_feasible == 1:  # Only one feasible solution found
                    insertion_regret -= -int_inf

                if num_feasible >= 1 and best_insertion:  # at least one feasible solution found
                    repair_candidates.append((best_insertion[0],
                                              best_insertion[1],
                                              best_insertion[2],
                                              insertion_regret))

            if not repair_candidates:  # no orders to insert feasibly
                return sol, sol.get_orders_not_served()

            highest_regret_insertion = max(repair_candidates, key=lambda item: item[3])
            sol.check_insertion_feasibility(node_id=highest_regret_insertion[0],
                                            idx=highest_regret_insertion[1],
                                            vessel=highest_regret_insertion[2])
            sol.insert_last_checked()

        return sol, sol.get_orders_not_served()

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
        insertion_candidates: List[str]
        candidate_sol, insertion_candidates = self.generate_new_solution(destroy_op=d_op, repair_op=r_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        accept_solution, update_type, cost = self.accept_solution(candidate_sol)

        if accept_solution:
            self.current_sol = candidate_sol
            self.current_sol_cost = cost
            self.insertion_candidates = insertion_candidates

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
            if self.current_sol.get_production_cost(pp_model=self.production_model) < int_inf:
                # TODO: int_inf risky, must be less than int_inf in production_model.py
                self.best_sol = self.current_sol
                self.best_sol_cost = self.current_sol_cost
                self.new_best_solution_feasible_production_count += 1
            else:
                self.feasible_production = False
                self.new_best_solution_infeasible_production_count += 1

        # update iteration parameters
        self.temperature = self.temperature * self.cooling_rate
        self.it_seg_count += 1


if __name__ == '__main__':
    precedence: bool = True

    prbl = ProblemDataExtended('../../data/input_data/larger_testcase_4vessels.xlsx', precedence=precedence)
    destroy_op = ['d_random', 'd_worst', 'd_related_location_time']
    if precedence:
        destroy_op.append('d_related_location_precedence')

    print()
    print("ALNS starting...")
    alns = Alns(problem_data=prbl,
                destroy_op=destroy_op,
                repair_op=['r_greedy', 'r_2regret'],
                weight_min_threshold=0.2,
                reaction_param=0.3,
                score_params=[5, 3, 1],  # corresponding to sigma_1, sigma_2, sigma_3 in R&P and L&N
                start_temperature_controlparam=0.5,  # solution 50% worse than best solution is accepted with 50% prob.
                cooling_rate=0.985,
                max_iter_seg=10,
                remove_percentage=0.3,
                determinism_param=5,
                relatedness_precedence={('green', 'yellow'): 6, ('green', 'red'): 10, ('yellow', 'red'): 4},
                related_removal_weight_param={'relatedness_location_time': [1, 0.9],
                                              'relatedness_location_precedence': [0.25, 1]},
                inventory_reward=False,
                verbose=False
                )
    print(alns)

    iterations = 400

    solution_costs = []
    for i in range(iterations):
        print("Iteration", i)
        # if i % 50 == 0 and i > 0:
        #     print("> Iteration", i)
        alns.run_alns_iteration()
        solution_costs.append((i, alns.current_sol_cost))
        print()

    # print(alns.rel_components)
    print()
    print("...ALNS terminating")
    print(alns)

    print(f"Best solution updated:{alns.new_best_solution_feasible_production_count} times")
    print(f"Candidate to become best solution rejected {alns.new_best_solution_infeasible_production_count} times, "
          f"because of production infeasibility")

    # print("PP final cost:", alns.best_sol.get_production_cost(pp_model=alns.production_model), "\n")

    # plot_alns_history(solution_costs)

import math
import random
from typing import Dict, List, Tuple, Union, Callable
from src.alns.solution import Solution, ProblemDataExtended
import numpy as np

int_inf = 9999


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
# [ ] Related removal (distance, time window, shaw)


class Alns:
    it_seg_count: int
    best_sol: Solution
    best_sol_cost: int
    current_sol: Solution
    current_sol_cost: int
    destroy_op_score: Dict[str, float]  # key is destroy operator "ID"; value is a score, updated for each segment
    repair_op_score: Dict[str, float]  # key is repair operator "ID"; value is score, updated for each segment
    destroy_op_weight: Dict[str, float]
    repair_op_weight: Dict[str, float]
    destroy_op_segment_usage: Dict[str, int]
    repair_op_segment_usage: Dict[str, int]
    unrouted_orders: List[str] = []  # orders not inserted in current_sol
    reaction_param: float
    score_params: Dict[int, int]
    weight_min_threshold: float
    temperature: float
    cooling_rate: float

    def __init__(self, problem_data: ProblemDataExtended,
                 destroy_op: List[str],
                 repair_op: List[str],
                 weight_min_threshold: float,
                 reaction_param: float,
                 score_params: List[int],
                 start_temperature_controlparam: float,
                 cooling_rate: float) -> None:
        # ALNS  parameters
        self.max_iter_alns = 100
        self.max_iter_seg = 10
        self.remove_num = 3
        #self.weight_min_threshold = 0.2

        # Solutions
        self.current_sol = self.construct_initial_solution(problem_data)
        self.current_sol_cost = self.current_sol.get_solution_cost()
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

        self.verbose = True

    def __repr__(self):
        return (
            f"Best solution with routing cost {self.best_sol_cost}: \n"
            f"{self.best_sol} \n"
            f"Current solution with routing cost {self.current_sol_cost}: \n"
            f"{self.current_sol} \n"
            f"Insertion candidates (orders not served): {self.insertion_candidates} \n"
            f"Destroy operators {[(k, round(v, 2)) for k, v in self.destroy_op_weight.items()]} \n"
            f"and repair operators {[(k, round(v, 2)) for k, v in self.repair_op_weight.items()]} \n")

    def construct_initial_solution(self, problem_data: ProblemDataExtended) -> Solution:
        sol: Solution = Solution(problem_data)
        sol.verbose = False
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

    def update_scores(self, destroy_op: str, repair_op: str, update_type: int) -> None:
        self.it_seg_count += 1
        if update_type == -1:
            return
        if not 0 <= update_type < len(self.score_params):
            print("Update type does not exist. Scores not updated.")
            return
        self.destroy_op_score[destroy_op] += self.score_params[update_type]
        self.repair_op_score[repair_op] += self.score_params[update_type]
        return

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
                                                 self.repair_op_segment_usage[
                                                     op]), self.weight_min_threshold)
            self.repair_op_score[op] = 0
            self.repair_op_segment_usage[op] = 0

        self.it_seg_count = 0

        return

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
        else:
            print("Destroy operator does not exist")
            return None

    def destroy_random(self, sol: Solution) -> Solution:
        served_orders = [(v, o) for v in sol.prbl.vessels
                         for o in sol.routes[v] if not sol.prbl.nodes[o].is_factory]
        random.shuffle(served_orders)
        removals = 0
        while removals < remove_num and served_orders:
            vessel, order = served_orders.pop()
            remove_index = sol.routes[vessel].index(order)
            sol.remove_node(vessel, remove_index)
            removals += 1

        sol.recompute_solution_variables()
        return sol

    def destroy_worst(self, sol: Solution) -> Solution:
        # Operator only destroys costly order nodes, not factory nodes (should it?)
        served_orders = [o for v in sol.prbl.vessels
                         for o in sol.routes[v] if not sol.prbl.nodes[o].is_factory]

        removal_candidates: List[Tuple[str, int, str, float]] = []  # order, idx, vessel, utility

        for v in sol.prbl.vessels:
            route = sol.routes[v]
            for idx in range(len(route)):
                node = sol.prbl.nodes[route[idx]]
                if not node.is_factory:
                    removal_candidates.append((node.id, idx, v, sol.get_removal_utility(vessel=v, idx=idx)))

        if not removal_candidates:  # no orders to remove
            return sol

        removal_candidates.sort(key=lambda tup: tup[3], reverse=True)

        stop_idx = remove_num if remove_num <= len(removal_candidates) else len(removal_candidates)
        removals = removal_candidates[:stop_idx]  # nodes and positions to be removed
        removals.sort(key=lambda tup: tup[1], reverse=True)  # largest indices first

        for (node_id, idx, v, util) in removals:
            removals.pop(0)
            sol.remove_node(v, idx)

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
        for i in range(remove_num):  # TODO: Find out how to do this (q^{ALNS} stuff)
            insertion_cand = sol.get_orders_not_served()
            insertion_gain: List[float] = []
            insertion: List[Tuple[int, str]] = []  # tuple (idx, vessel)
            insertion_regrets: List[float] = []
            repair_candidates: List[Tuple[str, int, str]] = []

            for o in insertion_cand:
                # Get all insertion utilities
                for v in sol.prbl.vessels:
                    for idx in range(1, len(sol.routes[v]) + 1):
                        insertion_gain.append(sol.get_insertion_utility(node=sol.prbl.nodes[o], vessel=v, idx=idx))
                        insertion.append((idx, v))

                insertion = [ins for _, ins in sorted(zip(insertion_gain, insertion),
                                                      key=lambda pair: pair[0], reverse=True)]  # order desc
                insertion_gain.sort(reverse=True)

                insertion_regret = 0
                num_feasible = 0
                best_insertion: Tuple[str, int, str]
                while num_feasible < 2 and len(insertion) > 0:
                    idx, vessel = insertion[0]
                    feasible = sol.check_insertion_feasibility(o, vessel, idx)
                    if feasible:
                        if num_feasible == 0:
                            insertion_regret += insertion_gain[0]
                            best_insertion = (o, insertion[0][0], insertion[0][1])
                        elif num_feasible == 1:
                            insertion_regret -= insertion_gain[0]
                        num_feasible += 1

                    sol.clear_last_checked()
                    insertion.pop(0)
                    insertion_gain.pop(0)

                if num_feasible == 1:  # Only one feasible solution found
                    insertion_regret -= -int_inf

                if num_feasible >= 1 and best_insertion:  # at least one feasible solution found
                    repair_candidates.append(best_insertion)
                    insertion_regrets.append(insertion_regret)

            if len(repair_candidates) == 0:
                return sol, sol.get_orders_not_served()

            highest_regret_insertion = repair_candidates[insertion_regrets.index(max(insertion_regrets))]
            sol.check_insertion_feasibility(node_id=highest_regret_insertion[0],
                                            idx=highest_regret_insertion[1],
                                            vessel=highest_regret_insertion[2])
            sol.insert_last_checked()

        return sol, sol.get_orders_not_served()

    def accept_solution(self, sol: Solution) -> Tuple[bool, int]:
        """
        :param sol: solution to be compared with self.current_sol
        :return: True if sol should be accepted to self.current_sol, else False
        """
        sol_cost = sol.get_solution_cost()
        if sol_cost < self.best_sol_cost:
            self.temperature = self.temperature * self.cooling_rate
            return True, 0
        elif sol_cost < self.current_sol_cost:
            self.temperature = self.temperature * self.cooling_rate
            return True, 1
        else:
            # Simulated annealing criterion
            prob = pow(math.e, -((sol_cost - self.current_sol_cost) / self.temperature))
            accept = np.random.choice(
                np.array([True, False]), p=(np.array([prob, (1 - prob)]))
            )
            self.temperature = self.temperature * self.cooling_rate
            return accept, -1 + 3 * accept

    def run_alns_iteration(self) -> None:
        # Choose a destroy heuristic and a repair heuristic based on adaptive weights wdm
        # for method d in the current segment of iterations m
        d_op, r_op = self.choose_operators()

        # Generate a candidate solution x′ from the current solution x using the chosen destroy and repair heuristics
        candidate_sol: Solution
        insertion_candidates: List[str]
        candidate_sol, insertion_candidates = self.generate_new_solution(destroy_op=d_op, repair_op=r_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        accept_solution, update_type = self.accept_solution(candidate_sol)
        if accept_solution:
            self.current_sol = candidate_sol
            self.current_sol_cost = candidate_sol.get_solution_cost()
            self.insertion_candidates = insertion_candidates

        # Update scores πd of the destroy and repair heuristics - dependent on how good the solution is
        self.update_scores(destroy_op=d_op, repair_op=r_op, update_type=update_type)

        # if IS iterations has passed since last weight update then
        # Update the weight wdm+1 for method d for the next segment m + 1 based on
        # the scores πd obtained for each method in segment m
        if self.it_seg_count == max_iter_seg:
            self.update_weights()

        # if f(x) > f(x∗) then
        # set x∗ =x
        if update_type == 0:  # type 0 means global best solution is found
            self.best_sol = self.current_sol
            self.best_sol_cost = self.current_sol_cost
        return


if __name__ == '__main__':
    destroy_op = ['d_random', 'd_worst']  # 'd_related', 'd_worst', 'd_random']  # TBD
    repair_op = ['r_greedy', 'r_2regret']  # , 'r_regret']  # TBD
    max_iter_alns = 200
    max_iter_seg = 10
    remove_num = 4
    weight_min_threshold = 0.2
    reaction_param = 0.5
    score_params = [5, 3, 1]  # corresponding to sigma_1, sigma_2, sigma_3 in R&P and L&N
    start_temperature_controlparam = 0.05  # solution 5% worse than best solution is accepted with 50% probability
    cooling_rate = 0.98

    prbl = ProblemDataExtended('../../data/input_data/large_testcase.xlsx', precedence=True)

    print()
    print("ALNS starting...")
    alns = Alns(problem_data=prbl,
                destroy_op=destroy_op,
                repair_op=repair_op,
                weight_min_threshold=weight_min_threshold,
                reaction_param=reaction_param,
                score_params=score_params,
                start_temperature_controlparam=start_temperature_controlparam,
                cooling_rate=cooling_rate)
    print(alns)

    for i in range(max_iter_alns):
        print("Iteration", i)
        # if i % 50 == 0 and i > 0:
        #     print("> Iteration", i)
        alns.run_alns_iteration()
        print()

    print()
    print("...ALNS terminating")
    print(alns)

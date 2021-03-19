import copy
import random
from typing import Dict, List, Tuple, Union, Callable
from src.alns.solution import Solution, ProblemDataExtended
import numpy as np


# What to do on ALNS:
#
# General
# [ ] Construct initial solution
# [ ] Update solution (e and l) after removal.
# DROP THIS: Check that solution is feasible after removal (vessel load and n.o. products, production constraints)
# [ ] Calculate and update scores and weights
#
# Operators
# [/] Greedy insertion
# [ ] Regret insertion
# [/] Random removal
# [ ] Worst removal (objective, longest wait?)
# [ ] Related removal (distance, time window, shaw)


class Alns:
    it_seg_count: int
    best_sol: Solution
    current_sol: Solution
    destroy_op_score: Dict[str, float]  # key is destroy operator "ID"; value is a score, updated for each segment
    repair_op_score: Dict[str, float]  # key is repair operator "ID"; value is score, updated for each segment
    destroy_op_weight: Dict[str, float]
    repair_op_weight: Dict[str, float]
    unrouted_orders: List[str]  # orders not inserted in current_sol
    weight_min_threshold: float

    def __init__(self, init_sol: Solution, destroy_op: List[str], repair_op: List[str],
                 weight_min_threshold: float) -> None:
        # ALNS  parameters
        self.max_iter_alns = 100
        self.max_iter_seg = 10
        self.remove_num = 3
        self.weight_min_threshold = 0.2

        # Solutions
        self.current_sol = init_sol
        self.best_sol = self.current_sol

        # Operator weights, scores and usage
        self.weight_min_threshold = weight_min_threshold
        self.destroy_op_weight = {op: self.weight_min_threshold for op in destroy_op}
        self.repair_op_weight = {op: self.weight_min_threshold for op in repair_op}
        self.destroy_op_score = {op: 0 for op in destroy_op}
        self.repair_op_score = {op: 0 for op in repair_op}

        #
        self.it_seg_count = 0  # Iterations done in one segment - can maybe do this in run_alns_iteration?
        self.unrouted_orders = self.current_sol.get_orders_not_served()

        self.verbose = True

    def __repr__(self):
        return (
            f"Best solution: \n"
            f"{self.best_sol} \n"
            f"Current solution: \n"
            f"{self.current_sol} \n"
            f"Insertion candidates (orders not served): {self.unrouted_orders}\n"
            f"Destroy operators {[k for k in self.destroy_op_weight.keys()]}, "
            f"and repair operators {[k for k in self.repair_op_weight.keys()]} \n")

    def update_scores(self, destroy_op: str, repair_op: str) -> None:
        # TODO
        return

    def update_weights(self) -> None:
        # Update weights based on scores, "blank" scores
        # w = (1-r)w + r*(pi/theta)
        return

    def choose_operators(self) -> Tuple[str, str]:
        # Choose operators, probability of choosing an operator is based on current weights
        destroy_operators = list(zip(*self.destroy_op_weight.items()))
        destroy_op: str = np.random.choice(
            np.array([op for op in destroy_operators[0]]),
            p=(np.array([w for w in destroy_operators[1]]) / np.array(sum(destroy_operators[1]))))
        repair_operators = list(zip(*self.repair_op_weight.items()))
        repair_op: str = np.random.choice(
            np.array([op for op in repair_operators[0]]),
            p=(np.array([w for w in repair_operators[1]]) / np.array(sum(repair_operators[1]))))
        return destroy_op, repair_op

    def generate_new_solution(self, destroy_op: str, repair_op: str) -> Tuple[Solution, List[str]]:
        # Generate new feasible solution x' based on given operators
        # Return x'
        candidate_sol = self.current_sol.copy()
        candidate_sol = self.destroy(destroy_op, candidate_sol)
        candidate_sol, unrouted_orders = self.repair(repair_op, candidate_sol)
        return candidate_sol, unrouted_orders

    def destroy(self, destroy_op: str, sol: Solution) -> Union[Solution, None]:
        # Run function based on operator "ID"
        if destroy_op == "d_random":
            return self.destroy_random(sol)
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

    def repair(self, repair_op: str, sol: Solution) -> Union[None, Tuple[Solution, List[str]]]:
        if repair_op == "r_greedy":
            return self.repair_greedy(sol)
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
                if len([item for item in insertions if item[0] == insert_node]) == 0:
                    insertion_cand.remove(insert_node)
                    unrouted_orders.append(insert_node)
        return sol, unrouted_orders

    def accept_solution(self, sol: Solution) -> bool:
        """
        :param sol: solution to be compared with self.best_sol
        :return: True if self.best_sol should be accepted to sol, else False
        """
        # Simulated annealing criterion
        # TODO
        return True

    def run_alns_iteration(self) -> None:
        # Choose a destroy heuristic and a repair heuristic based on adaptive weights wdm
        # for method d in the current segment of iterations m
        d_op, r_op = self.choose_operators()

        # Generate a candidate solution x′ from the current solution x using the chosen destroy and repair heuristics
        candidate_sol: Solution
        unrouted_orders: List[str]
        candidate_sol, unrouted_orders = self.generate_new_solution(destroy_op=d_op, repair_op=r_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        if self.accept_solution(candidate_sol):
            self.current_sol = candidate_sol
            self.unrouted_orders = unrouted_orders

        # Update scores πd of the destroy and repair heuristics - dependent on how good the solution is
        self.update_scores(destroy_op=d_op, repair_op=r_op)

        # if IS iterations has passed since last weight update then
        # Update the weight wdm+1 for method d for the next segment m + 1 based on
        # the scores πd obtained for each method in segment m
        if self.it_seg_count == max_iter_seg:
            self.update_weights()
            self.it_seg_count = 0
        else:
            self.it_seg_count += 1

        # if f(x) > f(x∗) then
        # set x∗ =x
        if True:  # f(x) > f(x*) criterion  # TODO
            self.best_sol = self.current_sol
        return


if __name__ == '__main__':
    destroy_op = ['d_random']  # 'd_related', 'd_worst', 'd_random']  # TBD
    repair_op = ['r_greedy']  # , 'r_regret']  # TBD
    max_iter_alns = 100
    max_iter_seg = 10
    remove_num = 3
    weight_min_threshold = 0.2

    # Construct an initial solution  # TODO: do this in a better way!
    prbl = ProblemDataExtended('../../data/input_data/large_testcase.xlsx', precedence=True)
    init_sol = Solution(prbl)
    init_sol.verbose = False
    initial_insertions = [
            ('o_1', 'v_1', 1),
            ('f_1', 'v_1', 2),
            ('o_4', 'v_1', 2),
            ('o_2', 'v_1', 3),
            ('o_1', 'v_2', 2),
            ('f_2', 'v_2', 3),
            ('o_9', 'v_3', 1),
            ('f_2', 'v_3', 2),
            ('o_6', 'v_3', 2),
            ('o_7', 'v_3', 2),
            ('o_8', 'v_3', 2),
    ]

    for node, vessel, idx in initial_insertions:
        if init_sol.check_insertion_feasibility(node, vessel, idx):
            init_sol.insert_last_checked()
        else:
            init_sol.clear_last_checked()

    print()
    print("ALNS starting")
    alns = Alns(init_sol=init_sol, destroy_op=destroy_op, repair_op=repair_op,
                weight_min_threshold=weight_min_threshold)
    print(alns)
    print(alns.choose_operators())
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")

    # for iteration = 1 to I\^ALNS do
    for i in range(max_iter_alns):
        print("New iteration")
        alns.run_alns_iteration()
        print()

    print()
    print(alns)

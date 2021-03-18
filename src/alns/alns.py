import copy
import random
from typing import Dict, List, Tuple, Union
from src.alns.sketch import Solution, InternalProblemData
import numpy as np


def get_orders_not_served(sol: Solution) -> List[str]:
    served_orders = set(o for v in sol.prbl.vessels
                        for o in sol.routes[v])
    return list(set(sol.prbl.order_nodes) - served_orders)


class Alns:
    it_seg_count: int
    best_sol: Solution
    current_sol: Solution
    destroy_op_score: Dict[str, float]  # key is destroy operator "ID"; value is a score, updated for each segment
    repair_op_score: Dict[str, float]  # key is repair operator "ID"; value is score, updated for each segment
    destroy_op_weight: Dict[str, float]
    repair_op_weight: Dict[str, float]
    insertion_candidates: List[str]  # orders not inserted in current_sol
    weight_min_threshold: int

    def __init__(self, init_sol: Solution, destroy_op: List[str], repair_op: List[str], weight_min_threshold: int) -> None:
        self.current_sol = init_sol
        self.best_sol = self.current_sol

        # Operator weights, scores and usage
        self.weight_min_threshold = weight_min_threshold
        self.destroy_op_weight = {op: self.weight_min_threshold for op in destroy_op}
        self.repair_op_weight = {op: self.weight_min_threshold for op in repair_op}
        self.destroy_op_score = {op: 0 for op in destroy_op}
        self.repair_op_score = {op: 0 for op in repair_op}

        self.it_seg_count = 0  # Iterations done in one segment - can maybe do this in run_alns_iteration?
        self.insertion_candidates = get_orders_not_served(self.current_sol)

    def __repr__(self):
        return (
            f"Best solution: \n"
            f"{self.best_sol} \n"
            f"Current solution: \n"
            f"{self.current_sol} \n"
            f"Insertion candidates (orders not served): {self.insertion_candidates} \n"
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
        candidate_sol = copy.deepcopy(self.current_sol)
        candidate_sol = self.destroy(destroy_op, candidate_sol)
        candidate_sol, insertion_candidates = self.repair(repair_op, candidate_sol)
        return candidate_sol, insertion_candidates

    def destroy(self, destroy_op: str, sol: Solution) -> Union[Solution, None]:
        # Run function based on operator "ID"
        if destroy_op == "d_random":
            return self.destroy_random(sol)
        else:
            print("Destroy operator does not exist")
            return None

    def destroy_random(self, sol: Solution) -> Solution:
        served_orders = set(sol.prbl.order_nodes) - set(self.insertion_candidates)  # sol is equal to current_solution
        orders_to_remove = set()
        while len(orders_to_remove) < remove_num:
            orders_to_remove.add(random.randint(0, len(served_orders) - 1))

        for v in sol.prbl.vessels:
            for i in range(len(sol.routes[v])):
                node = sol.routes[v][i]
                if node in orders_to_remove:
                    sol.routes[v].remove(node)
        return sol

    def repair(self, repair_op: str, sol: Solution) -> Union[None, Tuple[Solution, List[str]]]:
        if repair_op == "r_greedy":
            return self.repair_greedy(sol)
        else:
            print("Repair operator does not exist")
            return None

    def repair_greedy(self, sol: Solution) -> Tuple[Solution, List[str]]:
        insertion_gain = []
        insertion: List[Tuple[str, int, str]] = []
        insertion_cand = get_orders_not_served(sol)
        for o in insertion_cand:
            for v in sol.prbl.vessels:
                for idx in range(1, len(sol.routes[v]) + 1):
                    insertion_gain.append(sol.get_insertion_utility(node=sol.prbl.nodes[o],
                                                                    idx=idx, vessel=v))
                    insertion.append((o, idx, v))

        # Sort candidates according to gain
        insertion = [ins for _, ins in sorted(zip(insertion_gain, insertion),
                                              key=lambda pair: pair[0], reverse=True)]  # desc order
        insertion_gain.sort(reverse=True)

        placed = 0
        while len(insertion) > 0 and placed < remove_num:
            if sol.check_insertion_feasibility(
                    insert_node=sol.prbl.nodes[insertion[0][0]],
                    idx=insertion[0][1], vessel=insertion[0][2]):
                sol.insert_node(node_id=insertion[0][0], idx=insertion[0][1], vessel=insertion[0][2])
                insertion_cand.remove(insertion[0][0])
                placed += 1
            insertion.pop(0)
            insertion_gain.pop(0)
        return sol, insertion_cand

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
        insertion_candidates: List[str]
        candidate_sol, insertion_candidates = self.generate_new_solution(destroy_op=d_op, repair_op=r_op)

        # if x′ is accepted by a simulated annealing–based acceptance criterion then set x = x′
        if self.accept_solution(candidate_sol):
            self.current_sol = candidate_sol
            self.insertion_candidates = insertion_candidates

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
    remove_num = 1
    weight_min_threshold = 0.2

    # Construct an initial solution  # TODO: do this in a better way!
    prbl = InternalProblemData('../../data/input_data/medium_testcase.xlsx', precedence=True)
    init_sol = Solution(prbl)
    init_sol.insert_node(node_id='o_5', vessel='v_1', idx=1)
    init_sol.insert_node(node_id='f_2', vessel='v_1', idx=len(init_sol.routes['v_1']))
    init_sol.insert_node(node_id='o_3', vessel='v_2', idx=1)
    init_sol.insert_node(node_id='f_2', vessel='v_2', idx=len(init_sol.routes['v_2']))

    alns = Alns(init_sol=init_sol, destroy_op=destroy_op, repair_op=repair_op, weight_min_threshold=weight_min_threshold)
    print(alns)
    print(alns.choose_operators())
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")
    # alns.repair("repair_greedy")

    # for iteration = 1 to I\^ALNS do
    # for i in range(max_iter_alns):
    #    alns.run_alns_iteration()

    print()
    print(alns)

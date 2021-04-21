from src.read_problem_data import ProblemData
from typing import Dict, Tuple, List
from collections import defaultdict
import itertools
from tabulate import tabulate
import numpy as np
from time import time
from src.alns.solution import Solution, ProblemDataExtended


class ProductionProblem:

    def __init__(self, base_problem_data: ProblemDataExtended, demand_dict: Dict[Tuple[str, str, int], int] = None) -> None:
        self.base_problem = base_problem_data
        self.products_index_map = {product: i for i, product in enumerate(self.base_problem.products)}
        self.index_product_map = {i: product for i, product in enumerate(self.base_problem.products)}
        self.product_permutations_same_group = {(self.products_index_map[p1], self.products_index_map[p2])
                                                for group_list in self.base_problem.product_groups.values()
                                                for p1, p2 in itertools.permutations(group_list, 2)}
        self.demands: Dict[str, Dict[int, List[int]]]
        self.pickup_times: Dict[str, List[int]]
        if demand_dict:
            self.set_demands_and_pickup_times(demand_dict)
        self.init_inventory = {factory: [inventory
                                         for (f, p), inventory in self.base_problem.factory_initial_inventories.items()
                                         if f == factory]
                               for factory in self.base_problem.factory_nodes}

    def set_demands_and_pickup_times(self, demand_dict: Dict[Tuple[str, str, int], int]) -> None:
        demands = {factory: defaultdict(lambda: [0 for _ in self.base_problem.products])
                   for factory in self.base_problem.factory_nodes}
        for (factory, product, t_demand), demand in demand_dict.items():
            if demand > 0:
                demands[factory][t_demand][self.products_index_map[product]] = demand
        self.demands = demands
        self.pickup_times = {factory: sorted(t_demand_keys) for factory, t_demand_keys in demands.items()}
        # demands example:
        # { 'f_1': {2: [330, 55, 50, 45, 50, 300],
        #           11: [150, 0, 50, 0, 0, 200]},
        #   'f_2': {8: [220, 40, 0, 0, 65, 0]}
        # }

    def is_same_product_group(self, p1: int, p2: int) -> bool:
        if p1 is None or p2 is None:
            return False
        else:
            return (p1, p2) in self.product_permutations_same_group or p1 == p2
    # Constraints for insertion: Inventory and unmet demand


class ProductionProblemSolution:

    def __init__(self, problem: ProductionProblem, factory: str) -> None:
        self.prbl: ProductionProblem = problem
        self.factory: str = factory
        self.inventory_capacity = self.prbl.base_problem.factory_inventory_capacities[self.factory]
        self.activities: Dict[str, List[int]] = {prod_line: [None for _ in self.prbl.base_problem.time_periods]  # list with product numbers
                                                 for f, prod_line in self.prbl.base_problem.production_lines_for_factories
                                                 if f == self.factory}
        self.inventory: Dict[int, List[int]] = self._init_inventory()

    def _init_inventory(self) -> Dict[int, List[int]]:
        init_inventory = np.array(self.prbl.init_inventory[self.factory])
        cumul_demand = np.zeros(len(self.prbl.base_problem.products))
        inventory = {}
        for t in self.prbl.pickup_times[self.factory]:
            inventory[t] = init_inventory - cumul_demand
            cumul_demand += np.array(self.prbl.demands[self.factory][t])
        return inventory

    def demand_is_satisfied(self, t_demand: int) -> bool:
        # Demand is satisfied if inventory is larger than the demand in t_demand
        return all(demand_p <= inventory_p for inventory_p, demand_p in
                   zip(self.inventory[t_demand], self.prbl.demands[self.factory][t_demand]))

    def get_insertion_candidates(self, t_demand: int) -> List[Tuple[str, int, int, int]]:
        unmet_demand_product_idxs = [i for i, (inv, dem) in
                                     enumerate(zip(self.inventory[t_demand], self.prbl.demands[self.factory][t_demand]))
                                     if inv < dem]
        candidates = [(prod_line, t_insert, product_idx,
                       self.get_insertion_cost(prod_line, t_insert, product_idx, t_demand))
                      for prod_line in self.activities.keys()
                      for product_idx in unmet_demand_product_idxs
                      for t_insert in range(0, t_demand)  # may insert until t_demand
                      if self.insertion_is_feasible(prod_line, t_insert, product_idx, t_demand)]
        return candidates

    def insertion_is_feasible(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> bool:
        if not self.activities[prod_line][t_insert] is None:
            return False

        if not self.insertion_is_product_group_change_feasible(prod_line, t_insert, product):
            return False

        if not self.insertion_is_inventory_feasible(prod_line, product, t_demand):
            return False

        return True

    def insertion_is_inventory_feasible(self, prod_line: str, product: int, t_demand: int) -> bool:
        product_name = self.prbl.index_product_map[product]
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]
        return sum(self.inventory[t_demand]) + extra_production <= self.inventory_capacity

    def insertion_is_product_group_change_feasible(self, prod_line: str, t_insert: int, product: int) -> bool:
        activity_before = self.activities[prod_line][t_insert - 1] if t_insert > 0 else None
        activity_after = self.activities[prod_line][t_insert + 1] if t_insert < len(self.activities[prod_line]) else None
        feasible_before = (self.prbl.is_same_product_group(activity_before, product)
                           or activity_before is None)
        feasible_after = (self.prbl.is_same_product_group(product, activity_after)
                          or activity_after is None)
        return feasible_before and feasible_after

    def insert_activity(self, prod_line: str, t_insert: int, product: int, t_demand: int):
        product_name = self.prbl.index_product_map[product]
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]

        # Insert activity
        self.activities[prod_line][t_insert] = product

        # Increase inventory for all remaining pickup times
        remaining_pickups = [t for t in self.inventory.keys() if t >= t_demand]
        for t in remaining_pickups:
            self.inventory[t][product] += extra_production

    def get_insertion_cost(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> int:
        product_name = self.prbl.index_product_map[product]
        t_before = max(0, t_insert - 1)
        t_after = min(len(self.activities[prod_line]), t_insert + 1)
        new_serie = self.activities[prod_line][t_before] != product and self.activities[prod_line][t_after] != product

        production_start_cost = (new_serie * self.prbl.base_problem.production_start_costs[self.factory, product_name])
        inventory_cost = ((t_demand - t_insert) * self.prbl.base_problem.inventory_unit_costs[self.factory]
                          * self.prbl.base_problem.production_max_capacities[prod_line, product_name])
        return production_start_cost + inventory_cost

    def get_cost(self) -> float:
        product_name = self.prbl.index_product_map
        prod_start_costs = self.prbl.base_problem.production_start_costs

        inventory_cost = 0
        temp_inventory = sum(self.prbl.init_inventory[self.factory])

        for t in self.prbl.base_problem.time_periods:
            if t in self.prbl.pickup_times[self.factory]:
                temp_inventory -= sum(self.prbl.demands[self.factory][t])
            inventory_cost += temp_inventory * self.prbl.base_problem.inventory_unit_costs[self.factory]
            temp_inventory += sum(self._get_produced_amount(activities[t], prod_line)
                                  for prod_line, activities in self.activities.items())

        production_start_cost = (sum(prod_start_costs[self.factory, product_name[activities[0]]]
                                     for prod_line, activities in self.activities.items()
                                     if activities[0] is not None)
                                 + sum(int(activities[t - 1] != activities[t])
                                       * prod_start_costs[self.factory, product_name[activities[t]]]
                                       for prod_line, activities in self.activities.items()
                                       for t in range(1, len(activities))
                                       if activities[t] is not None))
        return production_start_cost + inventory_cost

    def _get_produced_amount(self, activity: int, prod_line: str):
        prod_cap = self.prbl.base_problem.production_max_capacities
        product_name = self.prbl.index_product_map
        if activity is None:
            return 0
        else:
            return prod_cap[prod_line, product_name[activity]]

    def print(self):

        def get_print_symbol(activity_symbol):
            if activity_symbol is None:
                return '-'
            else:
                return self.prbl.index_product_map[activity_symbol]

        activities = [[prod_line] + [get_print_symbol(activity) for activity in activities]
                      for prod_line, activities in self.activities.items()]

        print(tabulate(activities, headers=['prod_line'] + list(self.prbl.base_problem.time_periods)))
        print()


class ProductionProblemHeuristic:

    def __init__(self, prbl: ProductionProblem):
        self.prbl = prbl

    @staticmethod
    def construct_greedy(sol: ProductionProblemSolution, verbose: bool = False) -> bool:
        for t_demand in sol.prbl.pickup_times[sol.factory]:
            while not sol.demand_is_satisfied(t_demand):  # Fill in activities until demand is met
                insert_candidates = sol.get_insertion_candidates(t_demand)
                if not insert_candidates:
                    return False
                insert_candidates.sort(key=lambda item: item[3])  # sort by insertion cost
                prod_line, t_insert, product_idx, _ = insert_candidates[0]
                sol.insert_activity(prod_line, t_insert, product_idx, t_demand)
                if verbose:
                    sol.print()
        return True

    def is_feasible(self, routing_sol: Solution) -> Tuple[bool, str]:
        self.prbl.set_demands_and_pickup_times(routing_sol.get_demand_dict())
        t0 = time()
        for factory in self.prbl.base_problem.factory_nodes:
            sol = ProductionProblemSolution(self.prbl, factory)
            is_feasible = self.construct_greedy(sol)
            # sol.print()
            if not is_feasible:
                print(round(time() - t0, 1), "s (h)", sep="")
                return False, factory
        print(round(time() - t0, 1), "s (h)", sep="")
        return True, ''


def hardcode_demand_dict(prbl: ProblemData) -> Dict[Tuple[str, str, int], int]:
    # y_locks = [
    #     ('f_1', 'o_6', 4),
    #     ('f_1', 'o_7', 4),
    #     ('f_1', 'o_8', 4),
    #     ('f_1', 'o_10', 4),
    #     ('f_1', 'o_11', 4),
    #
    #     ('f_1', 'o_16', 8),
    #     ('f_1', 'o_13', 8),
    #     ('f_1', 'o_12', 8),
    #
    #     ('f_1', 'o_4', 10),
    #     ('f_1', 'o_1', 10)
    # ]
    y_locks = [
        ('f_1', 'o_6', 4),
        ('f_1', 'o_7', 4),
        ('f_1', 'o_8', 4),
        ('f_1', 'o_10', 4),
        ('f_1', 'o_11', 4),

        ('f_1', 'o_16', 13),
        ('f_1', 'o_13', 13),
        ('f_1', 'o_12', 13),
        ('f_1', 'o_4',  13),
        ('f_1', 'o_1',  13),
        ('f_1', 'o_2',  13),
        ('f_1', 'o_3',  13),

        ('f_1', 'o_14', 20),
        ('f_1', 'o_15', 20),
        ('f_1', 'o_5', 20),
        ('f_1', 'o_9', 20),
    ]


    demands: Dict[Tuple[str, str, int], int] = {(i, p, t): 0 for i in prbl.factory_nodes
                                                for p in prbl.products for t in prbl.time_periods}
    for visit in y_locks:
        for j in range(len(prbl.nodes[visit[1]].demand)):  # index in product list
            if prbl.nodes[visit[1]].demand[j] > 0:
                demands[visit[0], prbl.products[j], visit[2]] += prbl.nodes[visit[1]].demand[j]

    return demands


if __name__ == '__main__':
    file_path = '../../data/input_data/largester_testcase.xlsx'
    problem_data = ProblemData(file_path)
    problem_data_ext = ProblemDataExtended(file_path)
    demands = hardcode_demand_dict(problem_data_ext)
    production_problem = ProductionProblem(problem_data_ext, demands)
    print(production_problem.init_inventory)
    print(production_problem.demands)
    # pprint(production_problem.unmet_demands)

    solution = ProductionProblemSolution(production_problem, 'f_1')
    t0 = time()
    print(ProductionProblemHeuristic.construct_greedy(solution))
    print(round(time() - t0, 8), 's')
    solution.print()
    print(solution.get_cost())


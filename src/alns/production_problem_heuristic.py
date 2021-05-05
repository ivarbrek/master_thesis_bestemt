from src.read_problem_data import ProblemData
from typing import Dict, Tuple, List, Any
from collections import defaultdict
import itertools
import bisect
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
        self.activities: Dict[str, List[int]] = {prod_line: [None if is_producing else 'stop'
                                                             for (f, t), is_producing in
                                                             self.prbl.base_problem.production_stops.items()
                                                             if f == factory]  # list with product numbers
                                                 for f2, prod_line in
                                                 self.prbl.base_problem.production_lines_for_factories
                                                 if f2 == self.factory}
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

    def insertion_is_feasible(self, prod_line: str, t_insert: int, product: int, t_demand: int,
                              check_min_periods: bool = True) -> bool:
        if not self.activities[prod_line][t_insert] is None:
            return False

        if not self.insertion_is_product_group_change_feasible(prod_line, t_insert, product):
            return False

        if not self.insertion_is_inventory_feasible(prod_line, product, t_demand):
            return False

        if check_min_periods and not self.insertion_is_min_periods_feasible(prod_line, t_insert, product, t_demand):
            return False

        return True

    def insertion_is_inventory_feasible(self, prod_line: str, product: int, t_demand: int) -> bool:
        # Note: This check assumes that insertions are done chronologically (eariest pickup first)
        product_name = self.prbl.index_product_map[product]
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]
        return sum(self.inventory[t_demand]) + extra_production <= self.inventory_capacity

    def insertion_is_product_group_change_feasible(self, prod_line: str, t_insert: int, product: int) -> bool:
        activity_before, activity_after = self._get_activity_before_and_after(prod_line, t_insert)
        feasible_before = (self.prbl.is_same_product_group(activity_before, product)
                           or activity_before is None or activity_before == 'stop')
        feasible_after = (self.prbl.is_same_product_group(product, activity_after)
                          or activity_after is None or activity_after == 'stop')
        return feasible_before and feasible_after

    def insertion_is_min_periods_feasible(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> bool:
        product_name = self.prbl.index_product_map[product]
        min_production_periods = self.prbl.base_problem.production_line_min_times[prod_line, product_name]

        if min_production_periods < 2:  # No need to check
            return True

        activity_before, activity_after = self._get_activity_before_and_after(prod_line, t_insert)
        next_t_demand = self._get_next_pickup_time(t_insert + 1)

        if activity_before == product or activity_after == product:
            return True
        elif self.insertion_is_feasible(prod_line, t_insert - 1, product, t_demand, check_min_periods=False):
            return True
        elif self.insertion_is_feasible(prod_line, t_insert + 1, product, next_t_demand, check_min_periods=False):
            return True

        return False

    def insert_activity(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> None:
        product_name = self.prbl.index_product_map[product]
        min_production_periods = self.prbl.base_problem.production_line_min_times[prod_line, product_name]
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]
        activity_before, activity_after = self._get_activity_before_and_after(prod_line, t_insert)

        # Insert activity
        self.activities[prod_line][t_insert] = product

        # Increase inventory for all remaining pickup times
        remaining_pickups = [t for t in self.inventory.keys() if t >= t_demand]
        for t in remaining_pickups:
            self.inventory[t][product] += extra_production

        # A new insertion must be made (assumes min_production_periods in [1, 2]
        if min_production_periods > 1 and not (activity_before == product or activity_after == product):
            self.insert_min_production_periods(prod_line, t_insert, product, t_demand)

    def insert_min_production_periods(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> None:
        next_t_demand = self._get_next_pickup_time(t_insert + 1)

        # prioritize insertion before t_insert if demanded for product
        if self.inventory[t_demand][product] < 0 < t_insert:
            if self.insertion_is_feasible(prod_line, t_insert - 1, product, t_demand, check_min_periods=False):
                self.insert_activity(prod_line, t_insert - 1, product, t_demand)
            elif self.insertion_is_feasible(prod_line, t_insert + 1, product, next_t_demand, check_min_periods=False):
                self.insert_activity(prod_line, t_insert + 1, product, next_t_demand)
        else: # else, insert after, as this gives lower inventory costs
            if self.insertion_is_feasible(prod_line, t_insert + 1, product, next_t_demand, check_min_periods=False):
                self.insert_activity(prod_line, t_insert + 1, product, next_t_demand)
            elif self.insertion_is_feasible(prod_line, t_insert - 1, product, t_demand, check_min_periods=False):
                self.insert_activity(prod_line, t_insert - 1, product, t_demand)

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
                                     if activities[0] not in [None, 'stop'])
                                 + sum(int(activities[t - 1] != activities[t])
                                       * prod_start_costs[self.factory, product_name[activities[t]]]
                                       for prod_line, activities in self.activities.items()
                                       for t in range(1, len(activities))
                                       if activities[t] not in [None, 'stop']))
        return production_start_cost + inventory_cost

    def _get_produced_amount(self, activity: int, prod_line: str):
        prod_cap = self.prbl.base_problem.production_max_capacities
        product_name = self.prbl.index_product_map
        if activity is None or activity == 'stop':
            return 0
        else:
            return prod_cap[prod_line, product_name[activity]]

    def _get_next_pickup_time(self, t: int) -> int:
        """Returns the next time from t (and including t) that is a pickup time period"""
        pickup_t_list = list(self.inventory.keys())
        idx = min(bisect.bisect(pickup_t_list, t), len(pickup_t_list) - 1)
        return pickup_t_list[idx]

    def _get_activity_before_and_after(self, prod_line: str, t_insert: int) -> Tuple[Any, Any]:
        activity_before = self.activities[prod_line][t_insert - 1] if t_insert > 0 else 'stop'
        activity_after = self.activities[prod_line][t_insert + 1] if t_insert < len(self.activities[prod_line]) else 'stop'
        return activity_before, activity_after

    def print(self):

        def get_print_symbol(activity_symbol):
            if activity_symbol is None:
                return '-'
            elif activity_symbol == 'stop':
                return 'stop'
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
                print(round(time() - t0, 1), "s (h) (infeasible)", sep="")
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
    orders_demand = [
        ('f_0', 'o_1', 45),
        ('f_0', 'o_2', 45),
        ('f_0', 'o_3', 45),
        ('f_0', 'o_4', 45),
        ('f_0', 'o_5', 45),
        ('f_0', 'o_6', 45),
        ('f_0', 'o_7', 45),
        ('f_0', 'o_8', 45),
        ('f_0', 'o_9', 45),
        ('f_0', 'o_10', 60),
        ('f_0', 'o_11', 60),
        ('f_0', 'o_12', 60),
        ('f_0', 'o_13', 60),
        ('f_0', 'o_14', 60),
        ('f_0', 'o_15', 60),
        ('f_0', 'o_16', 60),
        ('f_0', 'o_17', 60),
        ('f_0', 'o_18', 60),
        ('f_0', 'o_19', 60),
    ]

    demands: Dict[Tuple[str, str, int], int] = {(i, p, t): 0 for i in prbl.factory_nodes
                                                for p in prbl.products for t in prbl.time_periods}
    for visit in orders_demand:
        for j in range(len(prbl.nodes[visit[1]].demand)):  # index in product list
            if prbl.nodes[visit[1]].demand[j] > 0:
                demands[visit[0], prbl.products[j], visit[2]] += prbl.nodes[visit[1]].demand[j]

    return demands


if __name__ == '__main__':
    file_path = '../../data/testoutputfile.xlsx'
    problem_data = ProblemData(file_path)
    problem_data_ext = ProblemDataExtended(file_path)
    demands = hardcode_demand_dict(problem_data_ext)
    production_problem = ProductionProblem(problem_data_ext, demands)
    print(production_problem.init_inventory)
    print(production_problem.demands)
    # pprint(production_problem.unmet_demands)

    solution = ProductionProblemSolution(production_problem, 'f_0')

    t0 = time()
    print(ProductionProblemHeuristic.construct_greedy(solution))
    print(round(time() - t0, 8), 's')
    solution.print()
    print(solution.get_cost())



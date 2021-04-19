from src.read_problem_data import ProblemData
from src.alns.solution import ProblemDataExtended
from typing import Dict, Tuple, List
from pprint import pprint
from collections import defaultdict
import itertools
from tabulate import tabulate


class ProductionProblem:

    def __init__(self, base_problem_data: ProblemData, demand_dict: Dict[Tuple[str, str, int], int]) -> None:
        self.base_problem = base_problem_data
        self.products_index_map = {product: i for i, product in enumerate(self.base_problem.products)}
        self.index_product_map = {i: product for i, product in enumerate(self.base_problem.products)}
        self._init_demand(demand_dict)
        self.product_permutations_same_group = {(self.products_index_map[p1], self.products_index_map[p2])
                                                for group_list in self.base_problem.product_groups.values()
                                                for p1, p2 in itertools.permutations(group_list, 2)}
        print(self.product_permutations_same_group)
        self.init_inventory = {factory: sum(inventory for (f, p), inventory in
                                            self.base_problem.factory_initial_inventories.items()
                                            if f == factory)
                               for factory in self.base_problem.factory_nodes}

    def _init_demand(self, demand_dict: Dict[Tuple[str, str, int], int]) -> None:

        demands: Dict[str, defaultdict[int, List]] = {factory: defaultdict(lambda: [0 for _ in self.base_problem.products])
                                                      for factory in self.base_problem.factory_nodes}
        for (factory, product, t_demand), demand in demand_dict.items():
            if demand > 0:
                demands[factory][t_demand][self.products_index_map[product]] = demand
        self.pickup_times = {factory: sorted(t_demand_keys) for factory, t_demand_keys in demands.items()}
        # initial_inventories = {factory: {t_demand: sum(d for (f, p), d in self.base_problem.factory_initial_inventories.items()
        #                                                if f == factory)}}

        # demands example:
        # { 'f_1': {2: [330, 55, 50, 45, 50, 300],
        #           11: [150, 0, 50, 0, 0, 200]},
        #   'f_2': {8: [220, 40, 0, 0, 65, 0]}
        # }
        # Adjust unmet_demand for initial inventory. Unmet demand reduction is set at the earliest demand_t
        unmet_demands = {factory: {t: demand[:]}  # copy demands
                         for factory, t_demand_dict in demands.items()
                         for t, demand in t_demand_dict.items()}
        remaining_inventory = {factory: [0 for _ in self.base_problem.products]
                               for factory in self.base_problem.factory_nodes}
        for (factory, product), inventory in self.base_problem.factory_initial_inventories.items():
            remaining_inventory[factory][self.products_index_map[product]] = inventory
        for factory in unmet_demands:
            for product in range(len(self.products_index_map.keys())):
                for t_demand in self.pickup_times[factory]:
                    if remaining_inventory[factory][product] <= unmet_demands[factory][t_demand][product]:
                        unmet_demands[factory][t_demand][product] -= remaining_inventory[factory][product]
                        break
                    else:  # larger inventory than demand in t
                        remaining_inventory[factory][product] -= unmet_demands[factory][t_demand][product]
                        unmet_demands[factory][t_demand][product] = 0

        self.demands = demands
        self.unmet_demands = unmet_demands
        self.demand_factories = unmet_demands.keys()

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
        self.activities: Dict[str, List[int]] = {prod_line: [None for _ in self.prbl.base_problem.time_periods]  # list with product numbers
                                                 for f, prod_line in self.prbl.base_problem.production_lines_for_factories
                                                 if f == self.factory}
        # self.insert_frontier = {prod_line: {t_demand: [t_demand - 1]}
        #                         for t_demand in self.prbl.unmet_demands[self.factory].keys()
        #                         for prod_line in self.prbl.base_problem.production_lines}

        self.unmet_demand: Dict[int, List[int]] = {t_demand: demands for t_demand, demands in
                                                   self.prbl.unmet_demands[self.factory].items()}
        self.inventory = {self.prbl.pickup_times[i]: (init_inventory
                                                      - self.prbl.demands[self.factory][self.prbl.pickup_times[i - 1]])
                          for factory, init_inventory in self.prbl.init_inventory.items()
                          for i in range(1, len(self.prbl.pickup_times))
                          if factory == self.factory}
        # TODO: Inventories

    def construct_greedy(self) -> bool:
        for t_demand in self.prbl.pickup_times[self.factory]:
            print()
            print("pickup at ", t_demand, ":", sep="")
            while sum(self.unmet_demand[t_demand]) > 0:  # Fill in activities until demand is met
                self.print()
                insert_candidates = self.get_insertion_candidates(t_demand)
                if not insert_candidates:
                    return False
                insert_candidates.sort(key=lambda item: item[3])  # sort by insertion cost
                prod_line, t_insert, product_idx, _ = insert_candidates[0]
                self.insert_activity(prod_line, t_insert, product_idx, t_demand)
        return True

    def get_insertion_candidates(self, t_demand: int) -> List[Tuple[str, int, int, int]]:
        unmet_demand_product_idxs = [i for i, d in enumerate(self.unmet_demand[t_demand]) if d > 0]
        candidates = [(prod_line, t_insert, product_idx,
                       self.get_insertion_cost(prod_line, t_insert, product_idx, t_demand))
                      for prod_line in self.activities.keys()
                      for product_idx in unmet_demand_product_idxs
                      for t_insert in range(0, t_demand)  # may insert until t_demand
                      if self.insertion_is_feasible(prod_line, t_insert, product_idx)]
        return candidates

    def insertion_is_feasible(self, prod_line: str, t_insert: int, product: int) -> bool:
        if not self.activities[prod_line][t_insert] is None:
            return False

        if not self.insertion_is_inventory_feasible(prod_line, t_insert, product):
            return False

        if not self.insertion_is_product_group_change_feasible(prod_line, t_insert, product):
            return False

        return True

    def insertion_is_inventory_feasible(self, prod_line: str, t_insert: int, product: int) -> bool:
        product_name = self.prbl.index_product_map[product]
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]

        return True

    def insertion_is_product_group_change_feasible(self, prod_line: str, t_insert: int, product: int) -> bool:
        activity_before = self.activities[prod_line][t_insert - 1] if t_insert > 0 else None
        activity_after = self.activities[prod_line][t_insert + 1] if t_insert < len(self.activities[prod_line]) else None
        feasible_before = (activity_before == product
                           or self.prbl.is_same_product_group(activity_before, product)
                           or activity_before is None)
        feasible_after = (product == activity_after
                          or self.prbl.is_same_product_group(product, activity_after)
                          or activity_after is None)
        return feasible_before and feasible_after

    def insert_activity(self, prod_line: str, t_insert: int, product: int, t_demand: int):
        product_name = self.prbl.index_product_map[product]
        self.activities[prod_line][t_insert] = product
        extra_production = self.prbl.base_problem.production_max_capacities[prod_line, product_name]
        # self.inventory += extra_production  # TODO

        # reduce unmet demand for current and (potentially) for future t_demand
        for t in self.prbl.pickup_times[self.factory]:
            if t >= t_demand:
                if extra_production <= self.unmet_demand[t][product]:
                    self.unmet_demand[t][product] -= extra_production
                    break
                else:
                    extra_production -= self.unmet_demand[t][product]
                    self.unmet_demand[t][product] = 0

    def get_insertion_cost(self, prod_line: str, t_insert: int, product: int, t_demand: int) -> int:
        product_name = self.prbl.index_product_map[product]
        t_before = max(0, t_insert - 1)
        t_after = min(len(self.activities[prod_line]), t_insert + 1)
        new_serie = self.activities[prod_line][t_before] != product and self.activities[prod_line][t_after] != product

        production_start_cost = (new_serie * self.prbl.base_problem.production_start_costs[self.factory, product_name])
        inventory_cost = ((t_demand - t_insert) * self.prbl.base_problem.inventory_unit_costs[self.factory]
                          * self.prbl.base_problem.production_max_capacities[prod_line, product_name])
        return production_start_cost + inventory_cost

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


    # def get_prod_series_start_end(self, prod_line: str) -> List[Tuple[int, int, int]]:
    #     production_series = []  # [(product, start, end)]
    #     t = 0
    #     while t < len(self.activities[prod_line]):
    #         if self.activities[prod_line][t] is None:
    #             t += 1
    #         else:
    #             product = self.activities[prod_line][t]
    #             serie_start = t
    #             serie_end = -1
    #             for t2 in range(t + 1, len(self.activities[prod_line])):
    #                 if self.activities[prod_line][t2] != product:
    #                     serie_end = t2 - 1
    #             assert serie_end != -1, "Silly! This production series never ends"
    #             production_series.append((product, serie_start, serie_end))
    #     return production_series

    # optimize for min inventory + startup costs
    # constrained by maximum inventory and demand

class ProductionProblemHeuristic:
    pass


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
    #     ('f_1', 'o_4', 11),
    #     ('f_1', 'o_1', 11)
    # ]
    y_locks = [
        ('f_1', 'o_6', 4),
        ('f_1', 'o_7', 4),
        ('f_1', 'o_8', 4),
        ('f_1', 'o_10', 4),
        ('f_1', 'o_11', 4),

        ('f_1', 'o_16', 8),
        ('f_1', 'o_13', 8),
        ('f_1', 'o_12', 8),

        ('f_1', 'o_4', 10),
        ('f_1', 'o_1', 10)
    ]


    demands: Dict[Tuple[str, str, int], int] = {(i, p, t): 0 for i in prbl.factory_nodes
                                                for p in prbl.products for t in prbl.time_periods}
    for visit in y_locks:
        for j in range(len(prbl.nodes[visit[1]].demand)):  # index in product list
            if prbl.nodes[visit[1]].demand[j] > 0:
                demands[visit[0], prbl.products[j], visit[2]] += prbl.nodes[visit[1]].demand[j]

    return demands


if __name__ == '__main__':
    problem_data = ProblemData('../../data/input_data/larger_testcase2.xlsx')
    problem_data_ext = ProblemDataExtended('../../data/input_data/larger_testcase2.xlsx')
    demands = hardcode_demand_dict(problem_data_ext)
    production_problem = ProductionProblem(problem_data, demands)
    pprint(production_problem.unmet_demands)

    solution = ProductionProblemSolution(production_problem, 'f_1')
    print(solution.construct_greedy())


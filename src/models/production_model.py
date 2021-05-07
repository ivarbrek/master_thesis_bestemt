import pyomo.environ as pyo
from typing import Dict, Tuple
from time import time
import math
import logging
from tabulate import tabulate
from pyomo.opt import SolverStatus, TerminationCondition
from src.alns.solution import ProblemDataExtended, Solution

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class ProductionModel:
    def __init__(self,
                 prbl: ProblemDataExtended,
                 demands: Dict[Tuple[str, str, int], int],
                 inventory_reward_extension: bool = False) -> None:

        # GENERAL MODEL SETUP
        self.m = pyo.ConcreteModel()
        self.solver_factory = pyo.SolverFactory('gurobi')
        self.results = None
        self.solution = None
        self.inventory_reward_extension = inventory_reward_extension

        ################################################################################################################
        # SETS #########################################################################################################

        self.m.FACTORY_NODES = pyo.Set(initialize=[node_id for node_id in prbl.factory_nodes.keys()])
        self.m.PRODUCTS = pyo.Set(initialize=prbl.products)
        self.m.TIME_PERIODS = pyo.Set(initialize=prbl.time_periods)
        self.m.PRODUCTION_LINES = pyo.Set(initialize=prbl.production_lines)
        self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=prbl.production_lines_for_factories)

        products_within_same_product_group_tup = [(prod1, prod2)
                                                  for product_group in prbl.product_groups.keys()
                                                  for prod1 in prbl.product_groups[product_group]
                                                  for prod2 in prbl.product_groups[product_group]]
        self.m.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP = pyo.Set(dimen=2,
                                                                initialize=products_within_same_product_group_tup)

        ################################################################################################################
        # PARAMETERS ###################################################################################################

        # self.m.production_min_capacities = pyo.Param(self.m.PRODUCTION_LINES,
        #                                              self.m.PRODUCTS,
        #                                              initialize=prbl.production_min_capacities)

        self.m.production_max_capacities = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=prbl.production_max_capacities)

        self.m.production_start_costs = pyo.Param(self.m.FACTORY_NODES,
                                                  self.m.PRODUCTS,
                                                  initialize=prbl.production_start_costs)

        self.m.production_stops = pyo.Param(self.m.FACTORY_NODES,
                                            self.m.TIME_PERIODS,
                                            initialize=prbl.production_stops)

        self.m.factory_inventory_capacities = pyo.Param(self.m.FACTORY_NODES,
                                                        initialize=prbl.factory_inventory_capacities)

        self.m.factory_initial_inventories = pyo.Param(self.m.FACTORY_NODES,
                                                       self.m.PRODUCTS,
                                                       initialize=prbl.factory_initial_inventories)

        self.m.inventory_unit_costs = pyo.Param(self.m.FACTORY_NODES,
                                                initialize=prbl.inventory_unit_costs)

        self.m.production_line_min_times = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=prbl.production_line_min_times)

        self.m.demands = pyo.Param(self.m.FACTORY_NODES,
                                   self.m.PRODUCTS,
                                   self.m.TIME_PERIODS,
                                   initialize=demands,
                                   mutable=True)

        if self.inventory_reward_extension:
            self.m.inventory_targets = pyo.Param(self.m.FACTORY_NODES,
                                                 self.m.PRODUCTS,
                                                 initialize=prbl.inventory_targets)

            self.m.inventory_unit_rewards = pyo.Param(self.m.FACTORY_NODES,
                                                      initialize=prbl.inventory_unit_rewards)

        ################################################################################################################
        # VARIABLES ####################################################################################################

        # self.m.q = pyo.Var(self.m.PRODUCTION_LINES,
        #                    self.m.PRODUCTS,
        #                    self.m.TIME_PERIODS,
        #                    domain=pyo.NonNegativeReals,
        #                    initialize=0)

        self.m.g = pyo.Var(self.m.PRODUCTION_LINES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.s = pyo.Var(self.m.FACTORY_NODES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals,
                           initialize=0)

        self.m.a = pyo.Var(self.m.PRODUCTION_LINES,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.delta = pyo.Var(self.m.PRODUCTION_LINES,
                               self.m.PRODUCTS,
                               self.m.TIME_PERIODS,
                               domain=pyo.Boolean,
                               initialize=0)

        if self.inventory_reward_extension:
            self.m.s_plus = pyo.Var(self.m.FACTORY_NODES,
                                    self.m.PRODUCTS,
                                    domain=pyo.NonNegativeReals,
                                    initialize=0)

        ################################################################################################################
        # OBJECTIVE ####################################################################################################

        def obj(model):
            return (sum(model.inventory_unit_costs[i] * model.s[i, p, t]
                        for t in model.TIME_PERIODS
                        for p in model.PRODUCTS
                        for i in model.FACTORY_NODES)
                    + sum(model.production_start_costs[i, p] * model.delta[l, p, t]
                          for i in model.FACTORY_NODES
                          for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if i == ii
                          for p in model.PRODUCTS
                          for t in model.TIME_PERIODS))

        def obj_extended(model):
            return (obj(model)
                    - sum(model.inventory_unit_rewards[i] * model.s_plus[i, p]
                          for i in model.FACTORY_NODES
                          for p in model.PRODUCTS))

        if self.inventory_reward_extension:
            self.m.objective = pyo.Objective(rule=obj_extended, sense=pyo.minimize)
        else:
            self.m.objective = pyo.Objective(rule=obj, sense=pyo.minimize)

        ################################################################################################################
        # CONSTRAINTS ##################################################################################################

        def constr_inventory_below_capacity(model, i, t):
            return sum(model.s[i, p, t] for p in model.PRODUCTS) <= model.factory_inventory_capacities[i]

        self.m.constr_inventory_below_capacity = pyo.Constraint(self.m.FACTORY_NODES,
                                                                self.m.TIME_PERIODS,
                                                                rule=constr_inventory_below_capacity)

        def constr_initial_inventory(model, i, p):
            return model.s[i, p, 0] == (model.factory_initial_inventories[i, p] - model.demands[i, p, 0])

        self.m.constr_initial_inventory = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         rule=constr_initial_inventory)

        def constr_inventory_balance(model, i, p, t):
            if t == 0:
                return pyo.Constraint.Feasible
            return (model.s[i, p, t] == (model.s[i, p, (t - 1)] +
                    sum(model.production_max_capacities[l, p] * model.g[l, p, (t-1)]
                        for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i) -
                    model.demands[i, p, t]))

        self.m.constr_inventory_balance = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_inventory_balance)

        def constr_production_below_max_capacity(model, i, l, p, t):
            return model.g[l, p, t] <= model.production_stops[i, t]

        self.m.constr_production_below_max_capacity = pyo.Constraint(self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_below_max_capacity)

        # def constr_production_above_min_capacity(model, l, p, t):
        #     return model.q[l, p, t] >= model.production_min_capacities[l, p] * model.g[l, p, t]
        #
        # self.m.constr_production_above_min_capacity = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                              self.m.PRODUCTS,
        #                                                              self.m.TIME_PERIODS,
        #                                                              rule=constr_production_above_min_capacity)

        def constr_activate_delta(model, l, p, t):
            if t == 0:
                return pyo.Constraint.Feasible
            return model.g[l, p, t] - model.g[l, p, t - 1] <= model.delta[l, p, t]

        self.m.constr_activate_delta = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                      self.m.PRODUCTS,
                                                      self.m.TIME_PERIODS,
                                                      rule=constr_activate_delta)

        def constr_initial_production_start(model, l, p):
            return model.delta[l, p, 0] == model.g[l, p, 0]

        self.m.constr_initial_production_start = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                self.m.PRODUCTS,
                                                                rule=constr_initial_production_start)

        def constr_produce_minimum_number_of_periods(model, l, p, t):
            relevant_time_periods = {tau for tau in model.TIME_PERIODS if
                                     t <= tau <= t + model.production_line_min_times[l, p] - 1}
            return (model.production_line_min_times[l, p] * model.delta[l, p, t]
                    <=
                    sum(model.g[l, p, tau] for tau in relevant_time_periods))

        self.m.constr_produce_minimum_number_of_periods = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                         self.m.PRODUCTS,
                                                                         self.m.TIME_PERIODS,
                                                                         rule=constr_produce_minimum_number_of_periods)

        def constr_production_line_availability(model, l, t):
            return model.a[l, t] + sum(model.g[l, p, t] for p in model.PRODUCTS) == 1

        self.m.constr_production_line_availability = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                    self.m.TIME_PERIODS,
                                                                    rule=constr_production_line_availability)

        def constr_production_shift(model, l, p, t):
            if t == 0:
                return pyo.Constraint.Feasible
            relevant_products = {q for (qq, q) in model.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP if qq == p}
            return model.g[l, p, (t - 1)] <= model.a[l, t] + sum(model.g[l, q, t] for q in relevant_products)

        self.m.constr_production_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                        self.m.PRODUCTS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_production_shift)

        # Extension
        if self.inventory_reward_extension:
            def constr_rewarded_inventory_below_inventory_level(model, i, p):
                return model.s_plus[i, p] <= model.s[i, p, max(model.TIME_PERIODS)]

            self.m.constr_rewarded_inventory_below_inventory_level = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                    self.m.PRODUCTS,
                                                                                    rule=constr_rewarded_inventory_below_inventory_level)

            def constr_rewarded_inventory_below_inventory_target(model, i, p):
                return model.s_plus[i, p] <= model.inventory_targets[i, p]

            self.m.constr_rewarded_inventory_below_inventory_target = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                     self.m.PRODUCTS,
                                                                                     rule=constr_rewarded_inventory_below_inventory_target)

    def solve(self, verbose: bool = True, time_limit: int = None, feasibility_check: bool = False) -> None:
        self.solver_factory.options['SolutionLimit'] = 1 if feasibility_check else 1e8
        self.solver_factory.options['TimeLimit'] = time_limit if time_limit else 1e6
        t = time()
        self.results = self.solver_factory.solve(self.m, tee=verbose, load_solutions=(not feasibility_check))
        if verbose:
            if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
                print("Not optimal termination condition: ", self.results.solver.termination_condition)
            print("Solve time: ", round(time() - t, 1))

    def print_solution(self):
        for i in self.m.FACTORY_NODES:
            relevant_production_lines = {l for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i}
            table = []
            print("Factory", i)
            for p in self.m.PRODUCTS:
                row = [p, "prod"]
                for t in self.m.TIME_PERIODS:
                    if sum(self.m.g[l, p, t]() for l in relevant_production_lines) > 0.5:
                        row.append(round(sum(self.m.production_max_capacities[l, p] * self.m.g[l, p, t]() for l in
                                             relevant_production_lines)))  # + " [" + str(self.m.s[i, p, t]()) + "]")
                    else:
                        row.append(" ")
                table.append(row)
                row = ["\"", "inv"]
                for t in self.m.TIME_PERIODS:
                    if t == 0:
                        row.append(round(self.m.s[i, p, 0]()))
                    elif abs(self.m.s[i, p, t]() - self.m.s[i, p, t - 1]()) > 0.5:
                        row.append(round(self.m.s[i, p, t]()))  # + " [" + str(self.m.s[i, p, t]()) + "]")
                    else:
                        row.append(" ")
                table.append(row)
                table.append(["____"] * (len(list(self.m.TIME_PERIODS)) + 2))
            print(tabulate(table, headers=["product", "prod/inv"] + list(self.m.TIME_PERIODS)))
            print()

    def print_solution2(self):
        for i in self.m.FACTORY_NODES:
            prod_line_activities = {l: [] for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i}
            for l in prod_line_activities:
                for t in self.m.TIME_PERIODS:
                    has_activity = False
                    if not self.m.production_stops[i, t]:
                        prod_line_activities[l].append('stop')
                        has_activity = True
                    else:
                        for p in self.m.PRODUCTS:
                            if self.m.g[l, p, t]() > 0.5:
                                prod_line_activities[l].append(p)
                                has_activity = True
                    if not has_activity:
                        prod_line_activities[l].append(' ')
            activities = [[prod_line] + activities for prod_line, activities in prod_line_activities.items()]
            print(tabulate(activities, headers=['prod_line'] + list(self.m.TIME_PERIODS)))
            print()

    def reconstruct_demand(self, new_demands: Dict[Tuple[str, str, int], int]) -> None:
        """ Helper function for get_production_cost and for production feasibility check of initial solution """
        self.m.demands.reconstruct(new_demands)
        self.m.constr_initial_inventory.reconstruct()
        self.m.constr_inventory_balance.reconstruct()

    def get_production_cost(self, sol: Solution, verbose: bool = False, time_limit: int = 60) -> float:
        demand = sol.get_demand_dict()
        self.reconstruct_demand(demand)
        self.solve(verbose=verbose, time_limit=time_limit)

        if self.results.solver.termination_condition in [TerminationCondition.maxTimeLimit, TerminationCondition.optimal]:
            return self.m.objective()
        else:
            return math.inf

    def is_feasible(self, sol: Solution, verbose: bool = False) -> bool:
        demand = sol.get_demand_dict()
        self.reconstruct_demand(demand)
        t0 = time()
        self.solve(verbose=verbose, time_limit=5, feasibility_check=True)
        print(round(time() - t0, 1), "s", sep="")
        if self.results.solver.termination_condition not in [TerminationCondition.infeasible, TerminationCondition.maxTimeLimit]:
            return True
        else:
            return False


def hardcode_demand_dict(prbl: ProblemDataExtended) -> Dict[Tuple[str, str, int], int]:
    y_locks = [
        ('f_1', 'o_6', 2),
        ('f_1', 'o_7', 2),
        ('f_1', 'o_8', 2),
        ('f_1', 'o_10', 2),
        ('f_1', 'o_11', 2),

        ('f_2', 'o_16', 8),
        ('f_2', 'o_13', 8),
        ('f_2', 'o_12', 8),

        ('f_1', 'o_4', 11),
        ('f_1', 'o_1', 11)
    ]

    demands: Dict[Tuple[str, str, int], int] = {(i, p, t): 0 for i in prbl.factory_nodes
                                                for p in prbl.products for t in prbl.time_periods}
    for visit in y_locks:
        for j in range(len(prbl.nodes[visit[1]].demand)):  # index in product list
            if prbl.nodes[visit[1]].demand[j] > 0:
                demands[visit[0], prbl.products[j], visit[2]] += prbl.nodes[visit[1]].demand[j]

    return demands


if __name__ == '__main__':
    pass

    # prbl = ProblemDataExtended('../../data/input_data/larger_testcase.xlsx', precedence=False)
    # demands = hardcode_demand_dict(prbl)
    #
    # pp_model = ProductionModel(prbl=prbl, demands=demands, inventory_reward_extension=False)
    # pp_model.solve(verbose=False, time_limit=100)
    #
    # print("\nCOST RESULT")
    # try:
    #     print(pyo.value(pp_model.m.objective))
    # except AttributeError:
    #     print("Infeasible problem")

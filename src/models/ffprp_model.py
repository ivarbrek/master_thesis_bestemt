import pyomo.environ as pyo
from typing import Dict, List, Tuple
from time import time
from pyomo.core import Constraint  # TODO: Remove this and use pyo.Constraint.Feasible/Skip
from tabulate import tabulate
from pyomo.util.infeasible import log_infeasible_constraints

from src.alns.solution import ProblemDataExtended
from src.read_problem_data import ProblemData
import src.alns.alns


# TODO IN THIS FILE
# ----------------------------------------------------------------------------------------------------------------------------
# TODO: Check code marked with TODO


class FfprpModel:
    def __init__(self,
                 prbl: ProblemData,
                 extended_model: bool = False,
                 y_init_dict: Dict[Tuple[str, str, int], int] = None,
                 ) -> None:

        # GENERAL MODEL SETUP
        self.m = pyo.ConcreteModel()
        self.solver_factory = pyo.SolverFactory('gurobi')
        self.results = None
        self.solution = None
        self.extended_model = extended_model

        ################################################################################################################
        # SETS #########################################################################################################

        # NODE SETS
        self.m.NODES = pyo.Set(initialize=prbl.nodes)
        self.m.NODES_INCLUDING_DUMMIES = pyo.Set(
            initialize=prbl.nodes + ['d_0', 'd_-1'])  # d_0 is dummy origin, d_-1 is dummy destination
        self.m.NODES_INCLUDING_DUMMY_START = pyo.Set(initialize=prbl.nodes + ['d_0'])
        self.m.NODES_INCLUDING_DUMMY_END = pyo.Set(initialize=prbl.nodes + ['d_-1'])
        self.m.FACTORY_NODES = pyo.Set(initialize=prbl.factory_nodes)
        self.m.ORDER_NODES = pyo.Set(initialize=prbl.order_nodes)

        self.m.PRODUCTS = pyo.Set(initialize=prbl.products)
        self.m.VESSELS = pyo.Set(initialize=prbl.vessels)

        # ARCS
        self.m.ARCS = pyo.Set(self.m.VESSELS,
                              initialize=prbl.arcs_for_vessels)
        arcs_for_vessels_trip = [(v, i, j) for v in self.m.VESSELS for i, j in prbl.arcs_for_vessels[v]]
        self.m.ARCS_FOR_VESSELS_TRIP = pyo.Set(initialize=arcs_for_vessels_trip)

        # TIME PERIOD SETS
        self.m.TIME_PERIODS = pyo.Set(initialize=prbl.time_periods)

        # TUPLE SETS
        orders_related_to_nodes_tup = [(factory_node, order_node)
                                       for factory_node in prbl.factory_nodes
                                       for order_node in prbl.order_nodes] + [
                                          (order_node, order_node) for order_node in prbl.order_nodes]

        self.m.ORDERS_RELATED_TO_NODES_TUP = pyo.Set(dimen=2, initialize=orders_related_to_nodes_tup)

        nodes_for_vessels_tup = [(vessel, node)
                                 for vessel, node in prbl.nodes_for_vessels.keys()
                                 if prbl.nodes_for_vessels[vessel, node] == 1]
        self.m.NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=nodes_for_vessels_tup)

        factory_nodes_for_vessels_tup = [(vessel, node)
                                         for vessel, node in nodes_for_vessels_tup
                                         if node in prbl.factory_nodes]
        self.m.FACTORY_NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=factory_nodes_for_vessels_tup)

        order_nodes_for_vessels_tup = [(vessel, node)
                                       for vessel, node in nodes_for_vessels_tup
                                       if node in prbl.order_nodes]
        self.m.ORDER_NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=order_nodes_for_vessels_tup)

        vessels_relevantnodes_ordernodes = [(vessel, relevant_node, order_node)
                                            for vessel, order_node in order_nodes_for_vessels_tup
                                            for relevant_node, order_node2 in orders_related_to_nodes_tup
                                            if order_node2 == order_node
                                            and (vessel, relevant_node) in nodes_for_vessels_tup
                                            ]

        self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3,
                                                                     initialize=vessels_relevantnodes_ordernodes)

        vessels_factorynodes_ordernodes = [(vessel, factory_node, order_node)
                                           for vessel, order_node in order_nodes_for_vessels_tup
                                           for vessel2, factory_node in factory_nodes_for_vessels_tup
                                           if vessel == vessel2
                                           ]

        self.m.ORDER_NODES_FACTORY_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3, initialize=vessels_factorynodes_ordernodes)

        vessels_for_factory_nodes_tup = [(node, vessel)
                                         for vessel, node in nodes_for_vessels_tup
                                         if node in prbl.factory_nodes]
        self.m.VESSELS_FOR_FACTORY_NODES_TUP = pyo.Set(dimen=2, initialize=vessels_for_factory_nodes_tup)

        time_windows_for_orders_tup = [(order, time_period)
                                       for order, time_period in prbl.time_windows_for_orders.keys()
                                       if prbl.time_windows_for_orders[order, time_period] == 1]
        self.m.TIME_WINDOWS_FOR_ORDERS_TUP = pyo.Set(dimen=2, initialize=time_windows_for_orders_tup)

        self.m.PRODUCTION_LINES = pyo.Set(initialize=prbl.production_lines)

        self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=prbl.production_lines_for_factories)

        products_within_same_product_group_tup = [(prod1, prod2)
                                                  for product_group in prbl.product_groups.keys()
                                                  for prod1 in prbl.product_groups[product_group]
                                                  for prod2 in prbl.product_groups[product_group]]
        self.m.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP = pyo.Set(dimen=2,
                                                                initialize=products_within_same_product_group_tup)

        self.m.ZONES = pyo.Set(initialize=prbl.orders_for_zones.keys())

        orders_for_zones_tup = [(zone, order)
                                for zone, li in prbl.orders_for_zones.items()
                                for order in li]
        self.m.ORDERS_FOR_ZONES_TUP = pyo.Set(dimen=2, initialize=orders_for_zones_tup)

        green_nodes_for_vessel_tup = [(vessel, node)
                                      for vessel, node in nodes_for_vessels_tup
                                      if node in prbl.orders_for_zones['green']]
        self.m.GREEN_NODES_FOR_VESSEL_TUP = pyo.Set(dimen=2, initialize=green_nodes_for_vessel_tup)

        green_and_yellow_nodes_for_vessel_tup = [(vessel, node)
                                                 for vessel, node in nodes_for_vessels_tup
                                                 if node in prbl.orders_for_zones['green'] + prbl.orders_for_zones[
                                                     'yellow']]
        self.m.GREEN_AND_YELLOW_NODES_FOR_VESSEL_TUP = pyo.Set(initialize=green_and_yellow_nodes_for_vessel_tup)

        # sick_arcs_tup = list(set([(orig, dest)
        #                           for (v, orig, dest) in prbl.min_wait_if_sick.keys()
        #                           if prbl.min_wait_if_sick[v, orig, dest] > 0]))
        # self.m.WAIT_EDGES = pyo.Set(dimen=2,
        #                             initialize=sick_arcs_tup)  # prbl.min_wait_if_sick.keys())

        wait_edges_for_vessels_trip = [(v, i, j)
                                       for v in self.m.VESSELS
                                       for u, i, j in prbl.min_wait_if_sick.keys()
                                       if u == v and (i, j) in self.m.ARCS[v]]

        self.m.WAIT_EDGES_FOR_VESSEL_TRIP = pyo.Set(dimen=3, initialize=wait_edges_for_vessels_trip)

        # Extension
        if extended_model:
            self.m.TIME_WINDOW_VIOLATIONS = pyo.Set(
                initialize=[i for i in range(-prbl.max_tw_violation, prbl.max_tw_violation + 1)])

        print("Done setting sets!")

        ################################################################################################################
        # PARAMETERS ###################################################################################################

        self.m.vessel_ton_capacities = pyo.Param(self.m.VESSELS,
                                                 initialize=prbl.vessel_ton_capacities)

        self.m.vessel_nprod_capacities = pyo.Param(self.m.VESSELS,
                                                   initialize=prbl.vessel_nprod_capacities)

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

        self.m.transport_unit_costs = pyo.Param(self.m.VESSELS,
                                                initialize=prbl.transport_unit_costs)

        self.m.transport_times = pyo.Param(self.m.VESSELS,
                                           self.m.NODES_INCLUDING_DUMMY_START,
                                           self.m.NODES_INCLUDING_DUMMY_END,
                                           initialize=prbl.transport_times)

        self.m.transport_times_exact = pyo.Param(self.m.VESSELS,
                                                 self.m.NODES_INCLUDING_DUMMY_START,
                                                 self.m.NODES_INCLUDING_DUMMY_END,
                                                 initialize=prbl.transport_times_exact)

        self.m.loading_unloading_times = pyo.Param(self.m.VESSELS,
                                                   self.m.NODES,
                                                   initialize=prbl.loading_unloading_times)

        self.m.demands = pyo.Param(self.m.ORDERS_RELATED_TO_NODES_TUP,
                                   self.m.PRODUCTS,
                                   initialize=prbl.demands)

        self.m.production_line_min_times = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=prbl.production_line_min_times)

        self.m.start_time_for_vessels = pyo.Param(self.m.VESSELS,
                                                  initialize=prbl.start_times_for_vessels)

        self.m.vessel_initial_locations = pyo.Param(self.m.VESSELS,
                                                    initialize=prbl.vessel_initial_locations,
                                                    within=pyo.Any)

        self.m.factory_max_vessels_destination = pyo.Param(self.m.FACTORY_NODES,
                                                           initialize=prbl.factory_max_vessels_destination)

        self.m.factory_max_vessels_loading = pyo.Param(self.m.FACTORY_NODES,
                                                       self.m.TIME_PERIODS,
                                                       initialize=prbl.factory_max_vessels_loading)

        self.m.external_delivery_penalties = pyo.Param(self.m.ORDER_NODES,
                                                       initialize=prbl.external_delivery_penalties)

        self.m.min_wait_if_sick = pyo.Param(self.m.WAIT_EDGES_FOR_VESSEL_TRIP,
                                            initialize=prbl.min_wait_if_sick)

        # self.m.warm_start = pyo.Param({0},
        #                               initialize={0:False},
        #                               mutable=True)

        # Extension
        if extended_model:
            tw_violation_unit_cost = {k: prbl.tw_violation_unit_cost * abs(k) for k in self.m.TIME_WINDOW_VIOLATIONS}
            self.m.time_window_violation_cost = pyo.Param(self.m.TIME_WINDOW_VIOLATIONS,
                                                          initialize=tw_violation_unit_cost)

            # Fetch first (min) and last (max) time period within the time window of each order
            tw_min, tw_max = {}, {}
            for i in self.m.ORDER_NODES:
                tw_min[i] = min(t for i2, t in time_windows_for_orders_tup if i == i2)
                tw_max[i] = max(t for i2, t in time_windows_for_orders_tup if i == i2)

            self.m.tw_min = pyo.Param(self.m.ORDER_NODES, initialize=tw_min)
            self.m.tw_max = pyo.Param(self.m.ORDER_NODES, initialize=tw_max)

            # self.m.inventory_targets = pyo.Param(self.m.FACTORY_NODES,
            #                                      self.m.PRODUCTS,
            #                                      initialize=prbl.inventory_targets)

            self.m.inventory_unit_rewards = pyo.Param(self.m.FACTORY_NODES,
                                                      initialize=prbl.inventory_unit_rewards)
        print("Done setting parameters!")

        ################################################################################################################
        # VARIABLES ####################################################################################################
        self.m.x = pyo.Var(self.m.ARCS_FOR_VESSELS_TRIP,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.w = pyo.Var(self.m.VESSELS,
                           self.m.NODES,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        # def y_init(m, v, i, t):
        #     return y_init_dict[(v, i, t)] if (y_init_dict is not None and prbl.is_order_node(i)) else 0

        self.m.y = pyo.Var(self.m.NODES_FOR_VESSELS_TUP,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.z = pyo.Var(self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.l = pyo.Var(self.m.VESSELS,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals,
                           initialize=0)

        self.m.h = pyo.Var(self.m.VESSELS,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

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

        self.m.e = pyo.Var(self.m.ORDER_NODES,
                           domain=pyo.Boolean,
                           initialize=0)

        # Extension
        if extended_model:
            self.m.lambd = pyo.Var(self.m.ORDER_NODES,
                                   self.m.TIME_WINDOW_VIOLATIONS,
                                   domain=pyo.Boolean,
                                   initialize=0)

            self.m.s_plus = pyo.Var(self.m.FACTORY_NODES,
                                    self.m.PRODUCTS,
                                    domain=pyo.NonNegativeReals,
                                    initialize=0)

        print("Done setting variables!")

        ################################################################################################################
        # OBJECTIVE ####################################################################################################
        def obj(model):
            return (sum(model.inventory_unit_costs[i] * model.s[i, p, t]
                        for t in model.TIME_PERIODS
                        for p in model.PRODUCTS
                        for i in model.FACTORY_NODES)
                    + sum(model.transport_unit_costs[v] * model.transport_times_exact[v, i, j] * model.x[v, i, j, t]
                          for t in model.TIME_PERIODS
                          for v in model.VESSELS
                          for i, j in model.ARCS[v]
                          if i != 'd_0' and j != 'd_-1')
                    + sum(model.production_start_costs[i, p] * model.delta[l, p, t]
                          for i in model.FACTORY_NODES
                          for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if i == ii
                          for p in model.PRODUCTS
                          for t in model.TIME_PERIODS)
                    + sum(model.external_delivery_penalties[i] * model.e[i] for i in model.ORDER_NODES))

        def obj_extended(model):
            return (obj(model)
                    + sum(model.time_window_violation_cost[k] * model.lambd[i, k]
                          for i in model.ORDER_NODES
                          for k in model.TIME_WINDOW_VIOLATIONS)
                    - sum(model.inventory_unit_rewards[i] * model.s_plus[i, p]
                          for i in model.FACTORY_NODES
                          for p in model.PRODUCTS))

        if extended_model:
            self.m.objective = pyo.Objective(rule=obj_extended, sense=pyo.minimize)
        else:
            self.m.objective = pyo.Objective(rule=obj, sense=pyo.minimize)

        print("Done setting objective!")

        ################################################################################################################
        # CONSTRAINTS ##################################################################################################

        # def constr_y_heuristic_flying_start(model, v, i, t):
        #     if prbl.is_factory_node(node_id=i) or not model.warm_start[0]:
        #         return Constraint.Skip
        #     return model.y[v, i, t] == int(y_init_dict[v, i, t])
        #
        # self.m.constr_y_heuristic_flying_start = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
        #                                                         self.m.TIME_PERIODS,
        #                                                         rule=constr_y_heuristic_flying_start,
        #                                                         name="constr_y_heuristic_flying_start")

        # def constr_max_one_activity(model, v, t):
        #     relevant_nodes = {n for (vessel, n) in model.NODES_FOR_VESSELS_TUP if vessel == v}
        #
        #     return (sum(model.y_minus[v, i, t] +
        #                 model.y_plus[v, i, t] +
        #                 sum(model.x[v, i, j, t] for j in relevant_nodes) +
        #                 model.w[v, i, t]
        #                 for i in relevant_nodes)
        #             <= 1)

        def constr_max_one_activity(model, v, t):
            if t < model.start_time_for_vessels[v]:  # Skip constraint if vessel has not become available in t
                return pyo.Constraint.Skip
            else:
                return (sum(model.y[v, i, tau]
                            for v2, i in model.NODES_FOR_VESSELS_TUP if v2 == v
                            for tau in range(max(0, t - model.loading_unloading_times[v, i] + 1), t + 1))
                        + sum(model.x[v, i, j, tau]
                              for i, j in model.ARCS[v]
                              for tau in range(max(0, t - model.transport_times[v, i, j] + 1), t + 1))
                        + sum(model.w[v, i, t]
                              for v2, i in model.NODES_FOR_VESSELS_TUP if v2 == v)
                        == 1)

        self.m.constr_max_one_activity = pyo.Constraint(self.m.VESSELS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_max_one_activity,
                                                        name="constr_max_one_activity")

        def constr_max_m_vessels_loading(model, i, t):
            return (sum(model.y[v, i, tau]
                        for i2, v in model.VESSELS_FOR_FACTORY_NODES_TUP if i2 == i
                        for tau in range(max(0, t - model.loading_unloading_times[v, i] + 1), t + 1))
                    <= model.factory_max_vessels_loading[i, t])

        self.m.constr_max_m_vessels_loading = pyo.Constraint(self.m.FACTORY_NODES,
                                                             self.m.TIME_PERIODS,
                                                             rule=constr_max_m_vessels_loading,
                                                             name="constr_max_m_vessels_loading")

        def constr_delivery_within_time_window(model, i):
            relevant_vessels = {vessel for (vessel, j) in model.ORDER_NODES_FOR_VESSELS_TUP if j == i}
            relevant_time_periods = {t for (j, t) in model.TIME_WINDOWS_FOR_ORDERS_TUP if j == i}
            return (sum(model.y[v, i, t] for v in relevant_vessels for t in relevant_time_periods) + model.e[i]
                    == 1)

        self.m.constr_delivery_within_time_window = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_delivery_within_time_window,
                                                                   name="constr_delivery_within_time_window")

        def constr_sailing_after_loading_unloading(model, v, i, t):
            loading_unloading_time = pyo.value(model.loading_unloading_times[v, i])
            relevant_destination_nodes = [j for i2, j in model.ARCS[v]
                                          if i2 == i
                                          and j != 'd_-1']
            if t < loading_unloading_time:
                return 0 == sum(model.x[v, i, j, t] for j in relevant_destination_nodes)
            else:
                return (model.y[v, i, (t - loading_unloading_time)]
                        ==
                        sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

        self.m.constr_sailing_after_loading_unloading = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                       self.m.TIME_PERIODS,
                                                                       rule=constr_sailing_after_loading_unloading,
                                                                       name="constr_sailing_after_loading_unloading")

        def constr_wait_load_unload_after_sailing(model, v, i, t):
            relevant_nodes = [j for j, i2 in model.ARCS[v]
                              if i2 == i
                              and model.transport_times[v, j, i] <= t]
            # Only allow sailing from i to dummy end node if arc is defined
            x_to_dummy_end = model.x[v, i, 'd_-1', t] if (i, 'd_-1') in model.ARCS[v] else 0
            if t == 0:  # exclude w_t-1
                return (sum(
                    model.x[v, j, i, (t - model.transport_times[v, j, i])] for j in relevant_nodes)
                        ==
                        model.y[v, i, t] + model.w[v, i, t] + x_to_dummy_end)
            else:
                return (sum(
                    model.x[v, j, i, (t - model.transport_times[v, j, i])] for j in relevant_nodes)
                        + model.w[v, i, (t - 1)]
                        ==
                        model.y[v, i, t] + model.w[v, i, t] + x_to_dummy_end)

        self.m.constr_wait_load_unload_after_sailing = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_wait_load_unload_after_sailing,
                                                                      name="constr_wait_load_unload_after_sailing")

        def constr_start_route(model, v):
            return model.x[v, 'd_0', model.vessel_initial_locations[v], model.start_time_for_vessels[v]] == 1

        self.m.constr_start_route = pyo.Constraint(self.m.VESSELS, rule=constr_start_route,
                                                   name="constr_start_route")

        def constr_start_route_once(model, v):
            return (sum(model.x[v, 'd_0', j, t]
                        for i, j in model.ARCS[v] if i == 'd_0'
                        for t in model.TIME_PERIODS)
                    == 1)

        self.m.constr_start_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_start_route_once,
                                                        name="constr_start_route_once")

        def constr_end_route_once(model, v):
            return (sum(model.x[v, i, 'd_-1', t]
                        for t in model.TIME_PERIODS
                        for vessel, i in model.FACTORY_NODES_FOR_VESSELS_TUP
                        if vessel == v)
                    == 1)

        self.m.constr_end_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_end_route_once,
                                                      name="constr_end_route_once")

        def constr_maximum_vessels_at_end_destination(model, i):
            return (sum(model.x[v, i, 'd_-1', t]
                        for v in model.VESSELS
                        for t in model.TIME_PERIODS)
                    <= model.factory_max_vessels_destination[i])

        self.m.constr_maximum_vessels_at_end_destination = pyo.Constraint(self.m.FACTORY_NODES,
                                                                          rule=constr_maximum_vessels_at_end_destination,
                                                                          name="constr_maximum_vessels_at_end_destination")

        def constr_pickup_requires_factory_visit(model, v, i, j, t):
            return model.z[v, i, j, t] <= model.y[v, i, t]

        self.m.constr_pickup_requires_factory_visit = pyo.Constraint(self.m.ORDER_NODES_FACTORY_NODES_FOR_VESSELS_TRIP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_pickup_requires_factory_visit,
                                                                     name="constr_pickup_requires_factory_visit")

        def constr_delivery_requires_order_visit(model, v, i, t):
            return model.z[v, i, i, t] == model.y[v, i, t]

        self.m.constr_delivery_requires_order_visit = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_delivery_requires_order_visit,
                                                                     name="constr_delivery_requires_order_visit")

        def constr_vessel_initial_load(model, v, p):
            return (model.l[v, p, 0] ==
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v2, i, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if v2 == v))

        self.m.constr_vessel_initial_load = pyo.Constraint(self.m.VESSELS,
                                                           self.m.PRODUCTS,
                                                           rule=constr_vessel_initial_load,
                                                           name="constr_vessel_initial_load")

        def constr_load_balance(model, v, p, t):
            if t == 0:
                return Constraint.Feasible
            return (model.l[v, p, t] == model.l[v, p, (t - 1)] -
                    sum(model.demands[i, j, p] * model.z[v, i, j, t]
                        for (v2, i, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if v2 == v))

        self.m.constr_load_balance = pyo.Constraint(self.m.VESSELS,
                                                    self.m.PRODUCTS,
                                                    self.m.TIME_PERIODS,
                                                    rule=constr_load_balance,
                                                    name="constr_load_balance")

        def constr_product_load_binary_activator(model, v, p, t):
            return model.l[v, p, t] <= model.vessel_ton_capacities[v] * model.h[v, p, t]

        self.m.constr_product_load_binary_activator = pyo.Constraint(self.m.VESSELS,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_product_load_binary_activator,
                                                                     name="constr_product_load_binary_activator")

        def constr_load_below_vessel_ton_capacity(model, v, t):
            if t == 0:
                return Constraint.Feasible
            return sum(model.l[v, p, (t - 1)] for p in model.PRODUCTS) <= (model.vessel_ton_capacities[v] *
                                                                           (1 - sum(model.y[v, i, t] for i in
                                                                                    model.FACTORY_NODES)))

        self.m.constr_load_below_vessel_ton_capacity = pyo.Constraint(self.m.VESSELS,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_load_below_vessel_ton_capacity,
                                                                      name="constr_load_below_vessel_ton_capacity")

        def constr_load_below_vessel_nprod_capacity(model, v, t):
            return sum(model.h[v, p, t] for p in model.PRODUCTS) <= model.vessel_nprod_capacities[v]

        self.m.constr_load_below_vessel_nprod_capacity = pyo.Constraint(self.m.VESSELS,
                                                                        self.m.TIME_PERIODS,
                                                                        rule=constr_load_below_vessel_nprod_capacity,
                                                                        name="constr_load_below_vessel_nprod_capacity")

        # def constr_zero_final_load(model, v, p):
        #     return model.l[v, p, max(model.TIME_PERIODS)] == 0
        #
        # self.m.constr_zero_final_load = pyo.Constraint(self.m.VESSELS,
        #                                                self.m.PRODUCTS,
        #                                                rule=constr_zero_final_load)

        def constr_inventory_below_capacity(model, i, t):
            return sum(model.s[i, p, t] for p in model.PRODUCTS) <= model.factory_inventory_capacities[i]

        self.m.constr_inventory_below_capacity = pyo.Constraint(self.m.FACTORY_NODES,
                                                                self.m.TIME_PERIODS,
                                                                rule=constr_inventory_below_capacity,
                                                                name="constr_inventory_below_capacity")

        def constr_initial_inventory(model, i, p):
            return (model.s[i, p, 0] == model.factory_initial_inventories[i, p] +
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v, i2, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if i2 == i))

        self.m.constr_initial_inventory = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         rule=constr_initial_inventory,
                                                         name="constr_initial_inventory")

        def constr_inventory_balance(model, i, p, t):
            if t == 0:
                return Constraint.Feasible
            return (model.s[i, p, t] == model.s[i, p, (t - 1)]
                    + sum(model.production_max_capacities[l, p] * model.g[l, p, (t - 1)]
                          for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i)
                    + sum(model.demands[i, j, p] * model.z[v, i, j, t]
                          for v, i2, j in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                          if i2 == i))
        # sum(model.q[l, p, t - 1] for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i) +

        self.m.constr_inventory_balance = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_inventory_balance,
                                                         name="constr_inventory_balance")

        def constr_production_below_max_capacity(model, i, l, p, t):
            # return (model.q[l, p, t]
            #         == model.production_stops[i, t] * model.production_max_capacities[l, p] * model.g[l, p, t])
            return model.g[l, p, t] <= model.production_stops[i, t]

        self.m.constr_production_below_max_capacity = pyo.Constraint(self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_below_max_capacity,
                                                                     name="constr_production_below_max_capacity")

        # def constr_production_above_min_capacity(model, l, p, t):
        #     return model.q[l, p, t] >= model.production_min_capacities[l, p] * model.g[l, p, t]
        #
        # self.m.constr_production_above_min_capacity = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                              self.m.PRODUCTS,
        #                                                              self.m.TIME_PERIODS,
        #                                                              rule=constr_production_above_min_capacity)

        def constr_activate_delta(model, l, p, t):
            if t == 0:
                return Constraint.Feasible
            return model.g[l, p, t] - model.g[l, p, t - 1] <= model.delta[l, p, t]

        self.m.constr_activate_delta = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                      self.m.PRODUCTS,
                                                      self.m.TIME_PERIODS,
                                                      rule=constr_activate_delta,
                                                      name="constr_activate_delta")

        def constr_initial_production_start(model, l, p):
            return model.delta[l, p, 0] == model.g[l, p, 0]

        self.m.constr_initial_production_start = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                self.m.PRODUCTS,
                                                                rule=constr_initial_production_start,
                                                                name="constr_initial_production_start")

        def constr_produce_minimum_number_of_periods(model, l, p, t):
            relevant_time_periods = {tau for tau in model.TIME_PERIODS if
                                     t <= tau <= t + model.production_line_min_times[l, p] - 1}
            return (model.production_line_min_times[l, p] * model.delta[l, p, t]
                    <=
                    sum(model.g[l, p, tau] for tau in relevant_time_periods))

        self.m.constr_produce_minimum_number_of_periods = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                         self.m.PRODUCTS,
                                                                         self.m.TIME_PERIODS,
                                                                         rule=constr_produce_minimum_number_of_periods,
                                                                         name="constr_produce_minimum_number_of_periods")

        def constr_production_line_availability(model, l, t):
            return model.a[l, t] + sum(model.g[l, p, t] for p in model.PRODUCTS) == 1

        self.m.constr_production_line_availability = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                    self.m.TIME_PERIODS,
                                                                    rule=constr_production_line_availability,
                                                                    name="constr_production_line_availability")

        def constr_production_shift(model, l, p, t):
            if t == 0:
                return Constraint.Feasible
            relevant_products = {q for (qq, q) in model.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP if qq == p}
            return model.g[l, p, (t - 1)] <= model.a[l, t] + sum(model.g[l, q, t] for q in relevant_products)

        self.m.constr_production_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                        self.m.PRODUCTS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_production_shift,
                                                        name="constr_production_shift")

        def constr_wait_if_visit_sick_farm(model, v, i, j, t):
            if model.transport_times[v, i, j] <= t <= len(model.TIME_PERIODS) - model.min_wait_if_sick[v, i, j]:
                return (model.min_wait_if_sick[v, i, j] * model.x[v, i, j, t - model.transport_times[v, i, j]]
                        <=
                        sum(model.w[v, j, tau] for tau in range(t, t + model.min_wait_if_sick[v, i, j])))
            else:
                return Constraint.Feasible

        self.m.constr_wait_if_visit_sick_farm = pyo.Constraint(self.m.WAIT_EDGES_FOR_VESSEL_TRIP,
                                                               self.m.TIME_PERIODS,
                                                               rule=constr_wait_if_visit_sick_farm,
                                                               name="constr_wait_if_visit_sick_farm")

        # Extension
        if extended_model:
            def constr_delivery_no_tw_violation(model, i):
                return (sum(model.y[v, i, t]
                            for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j
                            for j, t in model.TIME_WINDOWS_FOR_ORDERS_TUP if i == j)
                        ==
                        model.lambd[i, 0])

            self.m.constr_delivery_within_time_window.deactivate()  # Deactivate the current delivery constraint
            self.m.constr_delivery_no_tw_violation = pyo.Constraint(self.m.ORDER_NODES,
                                                                    rule=constr_delivery_no_tw_violation,
                                                                    name="constr_delivery_no_tw_violation")

            def constr_delivery_tw_violation_earlier(model, i, k):
                if k < 0 and self.m.tw_min[i] + k in model.TIME_PERIODS:
                    return (sum(model.y[v, i, self.m.tw_min[i] + k]
                                for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j)
                            ==
                            model.lambd[i, k])
                else:
                    return Constraint.Skip

            self.m.constr_delivery_tw_violation_earlier = pyo.Constraint(self.m.ORDER_NODES,
                                                                         self.m.TIME_WINDOW_VIOLATIONS,
                                                                         rule=constr_delivery_tw_violation_earlier,
                                                                         name="constr_delivery_tw_violation_earlier")

            def constr_delivery_tw_violation_later(model, i, k):
                if k > 0 and self.m.tw_max[i] + k in model.TIME_PERIODS:
                    return (sum(model.y[v, i, self.m.tw_max[i] + k]
                                for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j)
                            ==
                            model.lambd[i, k])
                else:
                    return Constraint.Skip

            self.m.constr_delivery_tw_violation_later = pyo.Constraint(self.m.ORDER_NODES,
                                                                       self.m.TIME_WINDOW_VIOLATIONS,
                                                                       rule=constr_delivery_tw_violation_later,
                                                                       name="constr_delivery_tw_violation_later")

            def constr_choose_one_tw_violation(model, i):
                return (sum(model.lambd[i, k]
                            for k in model.TIME_WINDOW_VIOLATIONS
                            if model.tw_max[i] + k in model.TIME_PERIODS
                            and model.tw_min[i] + k in model.TIME_PERIODS)
                        + model.e[i]
                        == 1)

            self.m.constr_choose_one_tw_violation = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_choose_one_tw_violation,
                                                                   name="constr_choose_one_tw_violation")

            def constr_rewarded_inventory_below_inventory_level(model, i, p):
                return model.s_plus[i, p] <= model.s[i, p, max(model.TIME_PERIODS)]

            self.m.constr_rewarded_inventory_below_inventory_level = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                    self.m.PRODUCTS,
                                                                                    rule=constr_rewarded_inventory_below_inventory_level,
                                                                                    name="constr_rewarded_inventory_below_inventory_level")

            def constr_rewarded_inventory_below_inventory_target(model, i, p):
                return model.s_plus[i, p] <= model.inventory_targets[i, p]

            self.m.constr_rewarded_inventory_below_inventory_target = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                     self.m.PRODUCTS,
                                                                                     rule=constr_rewarded_inventory_below_inventory_target,
                                                                                     name="constr_rewarded_inventory_below_inventory_target")

        print("Done setting constraints!")

    def solve(self, verbose: bool = True, time_limit: int = None, warm_start: bool = False) -> None:
        print("Solver running...")

        t = time()
        # t_warm_solve = 0

        # if warm_start:
        #     print(f"Preparing for warm-start...")
        #     self.solver_factory.options['TimeLimit'] = time_limit
        #     self.m.warm_start.reconstruct({0: True})
        #     self.m.constr_y_heuristic_flying_start.reconstruct()
        #     self.solver_factory.options['SolutionLimit'] = 1
        #     try:
        #         self.results = self.solver_factory.solve(self.m, tee=verbose)
        #         # self.m.write("debug.lp")
        #         if self.results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        #             warm_start = False
        #             print(f"Initial ALNS solution was regarded infeasible")
        #     except ValueError:
        #         print(f"No warm-start initial solution found within time limit")
        #         warm_start = False
        #
        #     self.solver_factory.options['SolutionLimit'] = 2000000000  # Gurobi's default value
        #     print(f"...warm-start model completed!")
        #
        #     self.m.warm_start.reconstruct({0: False})
        #     self.m.constr_y_heuristic_flying_start.reconstruct()
        #     t_warm_solve = time() - t

        if time_limit:
            remaining_time_limit = time_limit  # max(time_limit - t_warm_solve, 60) if time_limit > 60 else time_limit - t_warm_solve
            print(f"{round(remaining_time_limit, 1)} seconds remains out of the total of {time_limit} seconds")
            self.solver_factory.options['TimeLimit'] = remaining_time_limit  # time limit in seconds

        try:
            print(f"Solving model...")
            self.results = self.solver_factory.solve(self.m, tee=verbose, warmstart=warm_start)
            # logfile=f'../../log_files/console_output_{log_name}.log'
            print(f"...model solved!")
            print("Termination condition", self.results.solver.termination_condition)
            if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
                print("Not optimal termination condition: ", self.results.solver.termination_condition)
                # log_infeasible_constraints(self.m, log_variables=True, log_expression=True)
            print("Solve time: ", round(time() - t, 1))
        except ValueError:
            print(f"No solution found within time limit of {time_limit} seconds")

    def print_result(self):

        def print_result_variablewise():
            def print_vessel_routing():
                print("VESSEL ROUTING (x variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for i, j in self.m.ARCS[v]:
                            if pyo.value(self.m.x[v, i, j, t]) == 1:
                                print("Vessel", v, "travels from", i, "to", j, "in time period", t)
                    print()
                print()

            def print_waiting():
                print("WAITING (w variable)")
                for v in self.m.VESSELS:
                    for i in self.m.ORDER_NODES:
                        for t in self.m.TIME_PERIODS:
                            if pyo.value(self.m.w[v, i, t]) >= 0.5:
                                print("Vessel", v, "waits to deliver order", i, "in time period", t)
                print()

            def print_order_delivery_and_pickup():
                print("ORDER DELIVERY AND PICKUP (y variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for (vv, i) in self.m.NODES_FOR_VESSELS_TUP:
                            if v == vv:
                                activity_str = "loading" if i in self.m.FACTORY_NODES else "unloading"
                                if pyo.value(self.m.y[v, i, t]) >= 0.5:
                                    print(t, ": vessel ", v, " starts ", activity_str, " in node ", i, sep="")
                print()

            def print_factory_production():
                print("FACTORY PRODUCTION (q variable)")
                production = False
                for t in self.m.TIME_PERIODS:
                    for l in self.m.PRODUCTION_LINES:
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.g[l, p, t]) >= 0.5:
                                print("Production line", l, "produces",
                                      pyo.value(self.m.production_max_capacities[l, p]), "tons of product",
                                      p,
                                      "in time period", t)
                                production = True
                if not production:
                    print("Nothing is produced")
                print()

            def print_factory_inventory():
                print("FACTORY INVENTORY (r variable)")
                for t in self.m.TIME_PERIODS:
                    for i in self.m.FACTORY_NODES:
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.s[i, p, t]) >= 0.5:
                                print("Factory", i, "holds", pyo.value(self.m.s[i, p, t]), "tons of product", p,
                                      "as inventory in time period", t)
                print()

            def print_factory_pickup():
                print("FACTORY PICKUPS (z variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for j in self.m.ORDER_NODES:
                            for i in {n for (n, o) in self.m.ORDERS_RELATED_TO_NODES_TUP if o == j}:
                                if pyo.value(self.m.z[v, i, j, t]) >= 0.5:
                                    print("Vessel", v, "handles order", j, "in node", i, "in time period", t)
                print()

            def print_vessel_load():
                print("VESSEL LOAD (l variable)")
                for v in self.m.VESSELS:
                    for p in self.m.PRODUCTS:
                        for t in self.m.TIME_PERIODS:
                            if pyo.value(self.m.l[v, p, t]) >= 0.5:
                                print("Vessel", v, "carries", pyo.value(self.m.l[v, p, t]), "tons of product", p,
                                      "in time period", t)
                print()

            def print_orders_not_delivered():
                all_delivered = True
                print("ORDERS NOT DELIVERED (e variable)")
                for i in self.m.ORDER_NODES:
                    if pyo.value(self.m.e[i]) >= 0.5:
                        print("Order", i, "is not delivered")
                        all_delivered = False
                if all_delivered:
                    print("All orders have been delivered")
                print()

            def print_production_starts():
                print("PRODUCTION START (delta variable)")
                for i in self.m.FACTORY_NODES:
                    relevant_production_lines = {l for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i}
                    for l in relevant_production_lines:
                        print("Production line", l, "at factory", i)
                        for t in self.m.TIME_PERIODS:
                            for p in self.m.PRODUCTS:
                                if pyo.value(self.m.delta[l, p, t]) >= 0.5:
                                    print(t, ": production of product ", p, " is started, imposing a cost of ",
                                          pyo.value(self.m.production_start_costs[i, p]), ", and ",
                                          pyo.value(self.m.production_max_capacities[l, p]), " is produced", sep="")
                        print()

            def print_production_happens():
                print("PRODUCTION HAPPENS (g variable)")
                for i in self.m.FACTORY_NODES:
                    relevant_production_lines = {l for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i}
                    for l in relevant_production_lines:
                        print("Production line", l, "at factory", i)
                        for t in self.m.TIME_PERIODS:
                            for p in self.m.PRODUCTS:
                                if pyo.value(self.m.g[l, p, t]) >= 0.5:
                                    print(t, ": production of product ", p, " happens ", sep="")
                        print()

            def print_final_inventory():
                print("FINAL INVENTORY AND TARGETS (r_plus variable)")
                for i in self.m.FACTORY_NODES:
                    print("Factory", i)
                    for p in self.m.PRODUCTS:
                        if self.extended_model:
                            print("Rewarded inventory of product ", p, " is ", pyo.value(self.m.s_plus[i, p]),
                                  ", total final inventory is ", pyo.value(self.m.s[i, p, (max(self.m.TIME_PERIODS))]),
                                  " and its target is ", pyo.value(self.m.inventory_targets[i, p]), sep="")
                        else:
                            print("Final inventory for", p, "is", pyo.value(self.m.s[i, p, (max(self.m.TIME_PERIODS))]))
                    print()

            def print_available_production_lines():
                print("AVAILABLE PRODUCTION LINES")
                for ll in self.m.PRODUCTION_LINES:
                    for t in self.m.TIME_PERIODS:
                        print(t, ": production line ", ll, " has value ", pyo.value(self.m.a[ll, t]), sep="")
                    print()

            def print_time_window_violations():
                if self.extended_model:
                    for k in self.m.TIME_WINDOW_VIOLATIONS:
                        orders_with_k_violation = [i for i in self.m.ORDER_NODES if self.m.lambd[i, k]() > 0.5]
                        s = " " if k <= 0 else "+"
                        print(s + str(k), "violation:", orders_with_k_violation)
                else:
                    print("No time window violation, extension is not applied")
                print()

            # PRINTING
            print()
            # print_factory_production()
            # print_factory_inventory()
            # print_vessel_routing()
            # print_order_delivery_and_pickup()
            # print_factory_pickup()
            # print_waiting()
            # print_vessel_load()
            print_orders_not_delivered()
            # print_production_starts()
            # print_production_happens()
            # print_final_inventory()
            # print_available_production_lines()
            # print_time_window_violations()

        def print_result_eventwise():

            def print_routes_simple():
                table = []

                for v in self.m.VESSELS:
                    row = [v]
                    for t in self.m.TIME_PERIODS:
                        action_in_period = False
                        for i in self.m.NODES:
                            # Check if node may be visited by vessel
                            if i not in [i2 for v2, i2 in self.m.NODES_FOR_VESSELS_TUP if v2 == v]:
                                continue
                            if self.m.y[v, i, t]() > 0.5:
                                row.append(i)  # load or unload
                                action_in_period = True
                            if self.m.w[v, i, t]() > 0.5:
                                row.append('.')  # wait
                                action_in_period = True
                            if i in self.m.FACTORY_NODES and self.m.x[v, i, 'd_-1', t]() >= 0.5:  # route ends
                                row.append(i)
                                action_in_period = True

                        for i, j in self.m.ARCS[v]:
                            if self.m.x[v, i, j, t]() > 0.5 and i != 'd_0' and j != 'd_-1':
                                row.append(">")  # sail
                                action_in_period = True
                        if not action_in_period:
                            row.append(" ")
                    table.append(row)
                print(tabulate(table, headers=["vessel"] + list(self.m.TIME_PERIODS)))
                print()

            def print_y():
                active_y_s = [(v, i, t)
                              for v in self.m.VESSELS
                              for v2, i in self.m.NODES_FOR_VESSELS_TUP
                              for t in self.m.TIME_PERIODS
                              if v2 == v and self.m.y[v, i, t]() > 0.5]
                print("Active y's:", active_y_s)

            def print_routing(include_loads=True):
                for v in self.m.VESSELS:
                    print("ROUTING OF VESSEL", v)
                    for t in self.m.TIME_PERIODS:
                        curr_load = [round(self.m.l[v, p, t]()) for p in self.m.PRODUCTS]
                        # x variable
                        for i, j in self.m.ARCS[v]:
                            if pyo.value(self.m.x[v, i, j, t]) >= 0.5:
                                print(t, ": ", i, " --> ", j, sep="")
                                if include_loads and i != 'd_0':
                                    print("   load: ", curr_load)
                        # w variable
                        for i in self.m.NODES:
                            if pyo.value(self.m.w[v, i, t]) >= 0.5:
                                print(t, ": waits to go to ", i, sep="")
                                if include_loads:
                                    print("   load: ", curr_load)
                        for i in [j for (vessel, j) in self.m.NODES_FOR_VESSELS_TUP if vessel == v]:
                            # y variable
                            if pyo.value(self.m.y[v, i, t]) >= 0.5:
                                activity_str = "loads" if i in self.m.FACTORY_NODES else "unloads"
                                print(t, ": ", activity_str, " in node ", i, sep="")
                                if include_loads:
                                    print("   load: ", curr_load)
                        # z variable
                        for (v2, n, o) in self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP:
                            if v2 == v and pyo.value(self.m.z[v, n, o, t]) >= 0.5:
                                print("   [handles order ", o, " in node ", n, "]", sep="")
                    print()

            def print_vessel_load():
                for v in self.m.VESSELS:
                    print("LOAD AT VESSEL", v)
                    for t in self.m.TIME_PERIODS:
                        curr_load = [round(self.m.l[v, p, t]()) for p in self.m.PRODUCTS]
                        if sum(curr_load) > 0.5:
                            print(t, ": ", curr_load, sep="")
                    print()

            def print_production_and_inventory():
                for i in self.m.FACTORY_NODES:
                    print("PRODUCTION AND INVENTORY AT FACTORY", i)
                    for t in self.m.TIME_PERIODS:
                        relevant_production_lines = {l for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if
                                                     ii == i}
                        production = [round(sum(self.m.g[l, p, t]() * self.m.production_max_capacities[l, p]
                                                for l in relevant_production_lines))
                                      for p in sorted(self.m.PRODUCTS)]
                        inventory = [round(self.m.s[i, p, t]()) for p in sorted(self.m.PRODUCTS)]
                        if sum(production) > 0.5:
                            print(t, ": production: \t", production, sep="")
                        print(t, ": inventory: \t", inventory, sep="")
                        # for p in self.m.PRODUCTS:
                        #     if pyo.value(self.m.q[i, p, t]) >= 0.5:
                        #         print(t, ": production of ", round(pyo.value(self.m.q[i, p, t]), 1),
                        #               " tons of product ",
                        #               p, sep="")
                        #     if pyo.value(self.m.s[i, p, t]) >= 0.5:
                        #         print(t, ": inventory level is ", round(pyo.value(self.m.s[i, p, t]), 1),
                        #               " tons of product ", p, sep="")
                        #     relevant_order_nodes = {j for (f, j) in self.m.ORDER_NODES_FOR_FACTORIES_TUP if f == i}
                        #     loaded_onto_vessels = pyo.value(sum(
                        #         self.m.demands[i, j, p] * self.m.z[v, i, j, t]
                        #         for (v, i2, j) in self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TUP if i == i2))
                        #     if loaded_onto_vessels >= 0.5:
                        #         print(t, ": ", round(loaded_onto_vessels, 1), " tons of product ", p,
                        #               " is loaded onto vessels ", sep="")
                    print()

            def print_production_simple():
                for i in self.m.FACTORY_NODES:
                    relevant_production_lines = {l for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i}
                    table = []
                    print("Factory", i)
                    for p in self.m.PRODUCTS:
                        row = [p, "prod"]
                        for t in self.m.TIME_PERIODS:
                            if sum(self.m.g[l, p, t]() for l in relevant_production_lines) > 0.5:
                                row.append(
                                    round(sum(self.m.g[l, p, t]() * self.m.production_max_capacities[l, p] for l in
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

            # print_routing(include_loads=False)
            # print_vessel_load()
            # print_production_and_inventory()
            print_production_simple()
            print_y()
            print_routes_simple()

        def print_objective_function_components():
            inventory_cost = (sum(self.m.inventory_unit_costs[i] * pyo.value(self.m.s[i, p, t])
                                  for t in self.m.TIME_PERIODS
                                  for p in self.m.PRODUCTS
                                  for i in self.m.FACTORY_NODES))
            transport_cost = (
                sum(self.m.transport_unit_costs[v] * self.m.transport_times[v, i, j] * pyo.value(self.m.x[v, i, j, t])
                    for t in self.m.TIME_PERIODS
                    for v in self.m.VESSELS
                    for i, j in self.m.ARCS[v]
                    if i != 'd_0' and j != 'd_-1'))

            production_start_cost = (sum(self.m.production_start_costs[i, p] * pyo.value(self.m.delta[l, p, t])
                                         for i in self.m.FACTORY_NODES
                                         for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if i == ii
                                         for p in self.m.PRODUCTS
                                         for t in self.m.TIME_PERIODS))
            unmet_order_cost = (sum(self.m.external_delivery_penalties[i] * pyo.value(self.m.e[i])
                                    for i in self.m.ORDER_NODES))

            sum_obj = inventory_cost + transport_cost + unmet_order_cost + production_start_cost
            if self.extended_model:
                time_window_violation_cost = (sum(self.m.time_window_violation_cost[k] * self.m.lambd[i, k]()
                                                  for i in self.m.ORDER_NODES
                                                  for k in self.m.TIME_WINDOW_VIOLATIONS))
                final_inventory_reward = (-sum(self.m.inventory_unit_rewards[i] * pyo.value(self.m.s_plus[i, p])
                                               for i in self.m.FACTORY_NODES
                                               for p in self.m.PRODUCTS))
                sum_obj += time_window_violation_cost + final_inventory_reward
                print("Time window violation cost:", round(time_window_violation_cost, 2))
                print("Final inventory reward (negative cost):", round(final_inventory_reward, 2))

            print("Inventory cost:", round(inventory_cost, 2))
            print("Transport cost:", round(transport_cost, 2))
            print("Production start cost:", round(production_start_cost, 2))
            print("Unmet order cost:", round(unmet_order_cost, 2))

            print("Sum of above cost components:", round(sum_obj, 15))
            print("Objective value (from Gurobi):", pyo.value(self.m.objective))

        print_result_variablewise()
        print_result_eventwise()
        print_objective_function_components()


if __name__ == '__main__':
    # problem_data = ProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
    # problem_data = ProblemData('../../data/input_data/small_testcase.xlsx')
    # problem_data = ProblemData('../../data/input_data/medium_testcase.xlsx')
    # problem_data = ProblemData('../../data/input_data/large_testcase.xlsx')
    # problem_data = ProblemData('../../data/input_data/larger_testcase.xlsx')
    # problem_data = ProblemData('../../data/input_data/larger_testcase_4vessels.xlsx')

    # PARAMETERS TO CHANGE ###
    file_path = '../../data/input_data/gurobi_testing/f1-v3-o20-t72-tw4.xlsx'
    time_limit = 1800
    partial_warm_start = False
    num_alns_iterations = 200  # only used if partial_warm_start = True
    extensions = False  # extensions _not_ supported in generated test files

    # PARAMETERS NOT TO CHANGE ###
    problem_data = ProblemData(file_path)
    problem_data.soft_tw = extensions

    y_init_dict = None
    if partial_warm_start:
        problem_data_ext = ProblemDataExtended(file_path)  # TODO: Fix to avoid to prbl reads (problem with nodes field)
        problem_data_ext.soft_tw = extensions
        t = time()
        y_init_dict = src.alns.alns.run_alns(prbl=problem_data_ext, iterations=num_alns_iterations,
                                             skip_production_problem_postprocess=partial_warm_start, verbose=False)
        print(f"ALNS warmup time {round(time() - t, 1)}")

    model = FfprpModel(problem_data, extended_model=extensions, y_init_dict=y_init_dict)
    model.solve(time_limit=time_limit, warm_start=partial_warm_start)
    model.print_result()

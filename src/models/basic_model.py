import pyomo.environ as pyo
from typing import Dict, List
from time import time
from pyomo.core import Constraint
from tabulate import tabulate


# TODO IN THIS FILE
# ----------------------------------------------------------------------------------------------------------------------------
# TODO: Have not made any difference between different start/end time periods (so for now, not possible to have different time period lengths)
# TODO: Check code marked with TODO

class BasicModel:
    def __init__(self,
                 nodes: List,
                 factory_nodes: List,
                 order_nodes: List,
                 nodes_for_vessels: Dict,
                 products: List,
                 vessels: List,
                 time_periods: List,
                 time_periods_for_vessels: Dict,
                 vessel_initial_locations: Dict,
                 time_windows_for_orders: Dict,
                 vessel_ton_capacities: Dict,
                 vessel_nprod_capacities: Dict,
                 vessel_initial_loads: Dict,
                 factory_inventory_capacities: Dict,
                 factory_initial_inventories: Dict,
                 inventory_unit_costs: Dict,
                 transport_unit_costs: Dict,
                 transport_times: Dict,
                 unloading_times: Dict,
                 loading_times: Dict,
                 demands: Dict,
                 production_stops: Dict,
                 production_start_costs: Dict,
                 production_min_capacities: Dict,
                 production_max_capacities: Dict,
                 production_lines: List,
                 production_lines_for_factories: List,
                 production_line_min_times: Dict,
                 product_groups: Dict,
                 factory_max_vessels_destination: Dict,
                 factory_max_vessels_loading: Dict,
                 external_delivery_penalty: int,

                 # Extensions
                 orders_for_zones: Dict,
                 min_wait_if_sick: Dict,
                 max_tw_violation: int,
                 tw_violation_unit_cost: float,
                 inventory_targets: Dict,
                 inventory_unit_rewards: Dict,
                 extended_model=False,

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
        self.m.NODES = pyo.Set(initialize=nodes)
        self.m.NODES_INCLUDING_DUMMIES = pyo.Set(
            initialize=nodes + ['d_0', 'd_-1'])  # d_0 is dummy origin, d_-1 is dummy destination
        self.m.NODES_INCLUDING_DUMMY_START = pyo.Set(initialize=nodes + ['d_0'])
        self.m.NODES_INCLUDING_DUMMY_END = pyo.Set(initialize=nodes + ['d_-1'])
        self.m.FACTORY_NODES = pyo.Set(initialize=factory_nodes)
        self.m.ORDER_NODES = pyo.Set(initialize=order_nodes)

        self.m.PRODUCTS = pyo.Set(initialize=products)
        self.m.VESSELS = pyo.Set(initialize=vessels)

        # TIME PERIOD SETS
        self.m.TIME_PERIODS = pyo.Set(initialize=time_periods)

        # TUPLE SETS
        orders_related_to_nodes_tup = [(factory_node, order_node)
                                       for factory_node in factory_nodes
                                       for order_node in order_nodes] + [
                                          (order_node, order_node) for order_node in order_nodes]

        self.m.ORDERS_RELATED_TO_NODES_TUP = pyo.Set(dimen=2, initialize=orders_related_to_nodes_tup)

        nodes_for_vessels_tup = [(vessel, node)
                                 for vessel, node in nodes_for_vessels.keys()
                                 if nodes_for_vessels[vessel, node] == 1]
        self.m.NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=nodes_for_vessels_tup)

        factory_nodes_for_vessels_tup = [(vessel, node)
                                         for vessel, node in nodes_for_vessels_tup
                                         if node in factory_nodes]
        self.m.FACTORY_NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=factory_nodes_for_vessels_tup)

        order_nodes_for_vessels_tup = [(vessel, node)
                                       for vessel, node in nodes_for_vessels_tup
                                       if node in order_nodes]
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
                                         if node in factory_nodes]
        self.m.VESSELS_FOR_FACTORY_NODES_TUP = pyo.Set(dimen=2, initialize=vessels_for_factory_nodes_tup)

        time_windows_for_orders_tup = [(order, time_period)
                                       for order, time_period in time_windows_for_orders.keys()
                                       if time_windows_for_orders[order, time_period] == 1]
        self.m.TIME_WINDOWS_FOR_ORDERS_TUP = pyo.Set(dimen=2, initialize=time_windows_for_orders_tup)

        self.m.PRODUCTION_LINES = pyo.Set(initialize=production_lines)

        self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=production_lines_for_factories)

        products_within_same_product_group_tup = [(prod1, prod2)
                                                  for product_group in product_groups.keys()
                                                  for prod1 in product_groups[product_group]
                                                  for prod2 in product_groups[product_group]]
        self.m.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP = pyo.Set(dimen=2,
                                                                initialize=products_within_same_product_group_tup)

        # Extension
        if extended_model:
            self.m.ZONES = pyo.Set(initialize=orders_for_zones.keys())

            orders_for_zones_tup = [(zone, order)
                                    for zone, li in orders_for_zones.items()
                                    for order in li]
            self.m.ORDERS_FOR_ZONES_TUP = pyo.Set(dimen=2, initialize=orders_for_zones_tup)

            green_nodes_for_vessel_tup = [(vessel, node)
                                          for vessel, node in nodes_for_vessels_tup
                                          if node in orders_for_zones['green']]
            self.m.GREEN_NODES_FOR_VESSEL_TUP = pyo.Set(dimen=2, initialize=green_nodes_for_vessel_tup)

            green_and_yellow_nodes_for_vessel_tup = [(vessel, node)
                                                     for vessel, node in nodes_for_vessels_tup
                                                     if node in orders_for_zones['green'] + orders_for_zones['yellow']]
            self.m.GREEN_AND_YELLOW_NODES_FOR_VESSEL_TUP = pyo.Set(initialize=green_and_yellow_nodes_for_vessel_tup)

            self.m.WAIT_EDGES = pyo.Set(dimen=2, initialize=min_wait_if_sick.keys())

            wait_edges_for_vessels_trip = [(v, i, j)
                                           for v in self.m.VESSELS
                                           for i, j in self.m.WAIT_EDGES
                                           if (v, i) in self.m.ORDER_NODES_FOR_VESSELS_TUP
                                           and (v, j) in self.m.ORDER_NODES_FOR_VESSELS_TUP]

            self.m.WAIT_EDGES_FOR_VESSEL_TRIP = pyo.Set(dimen=3, initialize=wait_edges_for_vessels_trip)

            self.m.TIME_WINDOW_VIOLATIONS = pyo.Set(initialize=[i for i in range(-max_tw_violation, max_tw_violation+1)])

        print("Done setting sets!")

        ################################################################################################################
        # PARAMETERS ###################################################################################################

        self.m.vessel_ton_capacities = pyo.Param(self.m.VESSELS,
                                                 initialize=vessel_ton_capacities)

        self.m.vessel_nprod_capacities = pyo.Param(self.m.VESSELS,
                                                   initialize=vessel_nprod_capacities)

        self.m.vessel_initial_loads = pyo.Param(self.m.VESSELS,
                                                self.m.PRODUCTS,
                                                initialize=vessel_initial_loads)

        self.m.production_min_capacities = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=production_min_capacities)

        self.m.production_max_capacities = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=production_max_capacities)

        self.m.production_start_costs = pyo.Param(self.m.FACTORY_NODES,
                                                  self.m.PRODUCTS,
                                                  initialize=production_start_costs)

        self.m.production_stops = pyo.Param(self.m.FACTORY_NODES,
                                            self.m.TIME_PERIODS,
                                            initialize=production_stops)

        self.m.factory_inventory_capacities = pyo.Param(self.m.FACTORY_NODES,
                                                        initialize=factory_inventory_capacities)

        self.m.factory_initial_inventories = pyo.Param(self.m.FACTORY_NODES,
                                                       self.m.PRODUCTS,
                                                       initialize=factory_initial_inventories)

        self.m.inventory_unit_costs = pyo.Param(self.m.FACTORY_NODES,
                                                initialize=inventory_unit_costs)

        self.m.transport_unit_costs = pyo.Param(self.m.VESSELS,
                                                initialize=transport_unit_costs)

        self.m.transport_times = pyo.Param(self.m.NODES_INCLUDING_DUMMY_START,
                                           self.m.NODES,
                                           initialize=transport_times)

        self.m.unloading_times = pyo.Param(self.m.VESSELS,
                                           self.m.NODES,
                                           initialize=unloading_times)

        self.m.loading_times = pyo.Param(self.m.VESSELS,
                                         self.m.NODES,
                                         initialize=loading_times)

        self.m.demands = pyo.Param(self.m.ORDERS_RELATED_TO_NODES_TUP,
                                   self.m.PRODUCTS,
                                   initialize=demands)

        self.m.production_line_min_times = pyo.Param(self.m.PRODUCTION_LINES,
                                                     self.m.PRODUCTS,
                                                     initialize=production_line_min_times)

        self.m.time_period_for_vessels = pyo.Param(self.m.VESSELS,
                                                   initialize=time_periods_for_vessels)

        self.m.vessel_initial_locations = pyo.Param(self.m.VESSELS,
                                                    initialize=vessel_initial_locations,
                                                    within=pyo.Any)

        self.m.factory_max_vessels_destination = pyo.Param(self.m.FACTORY_NODES,
                                                           initialize=factory_max_vessels_destination)

        self.m.factory_max_vessels_loading = pyo.Param(self.m.FACTORY_NODES,
                                                       self.m.TIME_PERIODS,
                                                       initialize=factory_max_vessels_loading)

        self.m.external_delivery_penalty = pyo.Param(initialize=external_delivery_penalty)

        # Extension
        if extended_model:
            tw_violation_unit_cost = {k: tw_violation_unit_cost * abs(k) for k in self.m.TIME_WINDOW_VIOLATIONS}
            self.m.time_window_violation_cost = pyo.Param(self.m.TIME_WINDOW_VIOLATIONS,
                                                          initialize=tw_violation_unit_cost)

            # Fetch first (min) and last (max) time period within the time window of each order
            tw_min, tw_max = {}, {}
            for i in self.m.ORDER_NODES:
                tw_min[i] = min(t for i2, t in time_windows_for_orders_tup if i == i2)
                tw_max[i] = max(t for i2, t in time_windows_for_orders_tup if i == i2)

            self.m.tw_min = pyo.Param(self.m.ORDER_NODES, initialize=tw_min)
            self.m.tw_max = pyo.Param(self.m.ORDER_NODES, initialize=tw_max)

            self.m.min_wait_if_sick = pyo.Param(self.m.WAIT_EDGES,
                                                initialize=min_wait_if_sick)

            self.m.inventory_targets = pyo.Param(self.m.FACTORY_NODES,
                                                 self.m.PRODUCTS,
                                                 initialize=inventory_targets)

            self.m.inventory_unit_rewards = pyo.Param(self.m.FACTORY_NODES,
                                                      initialize=inventory_unit_rewards)
        print("Done setting parameters!")

        ################################################################################################################
        # VARIABLES ####################################################################################################
        self.m.x = pyo.Var(self.m.VESSELS,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.w = pyo.Var(self.m.VESSELS,
                           self.m.NODES,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.y_plus = pyo.Var(self.m.NODES_FOR_VESSELS_TUP,
                                self.m.TIME_PERIODS,
                                domain=pyo.Boolean,
                                initialize=0)

        self.m.y_minus = pyo.Var(self.m.NODES_FOR_VESSELS_TUP,
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
                           domain=pyo.NonNegativeReals)

        self.m.h = pyo.Var(self.m.VESSELS,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.q = pyo.Var(self.m.PRODUCTION_LINES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals,
                           initialize=0)

        self.m.g = pyo.Var(self.m.PRODUCTION_LINES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.r = pyo.Var(self.m.FACTORY_NODES,
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

        self.m.e = pyo.Var(self.m.ORDER_NODES,
                           domain=pyo.Boolean,
                           initialize=0)

        # Extension
        if extended_model:
            self.m.lambd = pyo.Var(self.m.ORDER_NODES,
                                   self.m.TIME_WINDOW_VIOLATIONS,
                                   domain=pyo.Boolean,
                                   initialize=0)

            self.m.r_plus = pyo.Var(self.m.FACTORY_NODES,
                                    self.m.PRODUCTS,
                                    domain=pyo.NonNegativeReals)



        print("Done setting variables!")

        ################################################################################################################
        # OBJECTIVE ####################################################################################################
        def obj(model):
            return (sum(model.inventory_unit_costs[i] * model.r[i, p, t]
                        for t in model.TIME_PERIODS
                        for p in model.PRODUCTS
                        for i in model.FACTORY_NODES)
                    + sum(model.transport_unit_costs[v] * model.transport_times[i, j] * model.x[v, i, j, t]
                          for t in model.TIME_PERIODS
                          for j in model.NODES
                          for i in model.NODES
                          for v in model.VESSELS)
                    + sum(model.production_start_costs[i, p] * model.delta[l, p, t]
                          for i in model.FACTORY_NODES
                          for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if i == ii
                          for p in model.PRODUCTS
                          for t in model.TIME_PERIODS)
                    + sum(model.external_delivery_penalty * model.e[i] for i in model.ORDER_NODES))

        def obj_extended(model):
            return (obj(model)
                    + sum(model.time_window_violation_cost[k] * model.lambd[i, k]
                          for i in model.ORDER_NODES
                          for k in model.TIME_WINDOW_VIOLATIONS)
                    - sum(model.inventory_unit_rewards[i] * model.r_plus[i, p]
                          for i in model.FACTORY_NODES
                          for p in model.PRODUCTS))

        if extended_model:
            self.m.objective = pyo.Objective(rule=obj_extended, sense=pyo.minimize)
        else:
            self.m.objective = pyo.Objective(rule=obj, sense=pyo.minimize)

        print("Done setting objective")

        ################################################################################################################
        # CONSTRAINTS ##################################################################################################

        def constr_max_one_activity(model, v, t):
            relevant_nodes = {n for (vessel, n) in model.NODES_FOR_VESSELS_TUP if vessel == v}

            return (sum(model.y_minus[v, i, t] +
                        model.y_plus[v, i, t] +
                        sum(model.x[v, i, j, t] for j in relevant_nodes) +
                        model.w[v, i, t]
                        for i in relevant_nodes)
                    <= 1)

        self.m.constr_max_one_activity = pyo.Constraint(self.m.VESSELS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_max_one_activity)

        def constr_max_m_vessels_loading(model, i, v, t):
            relevant_vessels = {vessel for (j, vessel) in model.VESSELS_FOR_FACTORY_NODES_TUP if j == i and vessel != v}
            relevant_time_periods = {tau for tau in model.TIME_PERIODS if (t <= tau < t + model.loading_times[v, i])}   # TODO: Sjekk at < og ikke <= er good
            return (sum(model.y_plus[u, i, tau]
                        for tau in relevant_time_periods
                        for u in relevant_vessels)
                    <= model.factory_max_vessels_loading[i, t] - model.y_plus[v, i, t])

        self.m.constr_max_m_vessels_loading = pyo.Constraint(self.m.VESSELS_FOR_FACTORY_NODES_TUP,
                                                             self.m.TIME_PERIODS,
                                                             rule=constr_max_m_vessels_loading)

        def constr_delivery_within_time_window(model, i):
            relevant_vessels = {vessel for (vessel, j) in model.ORDER_NODES_FOR_VESSELS_TUP if j == i}
            relevant_time_periods = {t for (j, t) in model.TIME_WINDOWS_FOR_ORDERS_TUP if j == i}
            return (sum(model.y_minus[v, i, t] for v in relevant_vessels for t in relevant_time_periods) + model.e[i]
                    == 1)

        self.m.constr_delivery_within_time_window = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_delivery_within_time_window)

        def constr_sailing_after_loading_unloading(model, v, i, t):
            unloading_time = pyo.value(model.unloading_times[v, i])
            loading_time = pyo.value(model.loading_times[v, i])
            relevant_destination_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                                          if vessel == v}

            if t < unloading_time and t < loading_time:  # Neither unloading_time nor loading_time is valid
                return Constraint.Feasible

            elif t < unloading_time:  # Only loading_time is valid
                return (model.y_plus[v, i, (t - loading_time)]
                        ==
                        sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

            elif t < loading_time:  # Only unloading_time is valid
                return (model.y_minus[v, i, (t - unloading_time)]
                        ==
                        sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

            else:
                return ((model.y_minus[v, i, (t - unloading_time)] + model.y_plus[v, i, (t - loading_time)])
                        ==
                        sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

        self.m.constr_sailing_after_loading_unloading = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                       self.m.TIME_PERIODS,
                                                                       rule=constr_sailing_after_loading_unloading)

        def constr_wait_load_unload_after_sailing(model, v, i, t):
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                              if vessel == v
                              and model.transport_times[j, i] <= t}.union({'d_0'})
            if t == 0:  # exclude w_t-1
                return (sum(model.x[v, j, i, (t - model.transport_times[j, i])] for j in relevant_nodes)
                        ==
                        model.y_minus[v, i, t] + model.y_plus[v, i, t] + model.w[v, i, t] + model.x[v, i, 'd_-1', t])
            else:
                return (sum(model.x[v, j, i, (t - model.transport_times[j, i])] for j in relevant_nodes)
                        + model.w[v, i, (t - 1)]
                        ==
                        model.y_minus[v, i, t] + model.y_plus[v, i, t] + model.w[v, i, t] + model.x[v, i, 'd_-1', t])

        self.m.constr_wait_load_unload_after_sailing = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_wait_load_unload_after_sailing)

        def constr_start_route(model, v):
            return model.x[v, 'd_0', model.vessel_initial_locations[v], model.time_period_for_vessels[v]] == 1

        self.m.constr_start_route = pyo.Constraint(self.m.VESSELS, rule=constr_start_route)

        def constr_start_route_once(model, v):
            return (sum(model.x[v, 'd_0', j, t]
                        for j in model.NODES_INCLUDING_DUMMIES
                        for t in model.TIME_PERIODS)
                    == 1)

        self.m.constr_start_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_start_route_once)

        def constr_end_route_once(model, v):
            return (sum(model.x[v, i, 'd_-1', t]
                        for t in model.TIME_PERIODS
                        for vessel, i in model.FACTORY_NODES_FOR_VESSELS_TUP
                        if vessel == v)
                    == 1)

        self.m.constr_end_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_end_route_once)

        def constr_maximum_vessels_at_end_destination(model, i):
            return (sum(model.x[v, i, 'd_-1', t]
                        for v in model.VESSELS
                        for t in model.TIME_PERIODS)
                    <= model.factory_max_vessels_destination[i])

        self.m.constr_maximum_vessels_at_end_destination = pyo.Constraint(self.m.FACTORY_NODES,
                                                                          rule=constr_maximum_vessels_at_end_destination)

        def constr_pickup_requires_factory_visit(model, v, i, j, t):
            return model.z[v, i, j, t] <= model.y_plus[v, i, t]

        self.m.constr_pickup_requires_factory_visit = pyo.Constraint(self.m.ORDER_NODES_FACTORY_NODES_FOR_VESSELS_TRIP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_pickup_requires_factory_visit)

        def constr_delivery_requires_order_visit(model, v, i, t):
            return model.z[v, i, i, t] == model.y_minus[v, i, t]

        self.m.constr_delivery_requires_order_visit = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_delivery_requires_order_visit)

        def constr_vessel_initial_load(model, v, p):
            return (model.l[v, p, 0] == model.vessel_initial_loads[v, p] -
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v2, i, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if v2 == v))

        self.m.constr_vessel_initial_load = pyo.Constraint(self.m.VESSELS,
                                                           self.m.PRODUCTS,
                                                           rule=constr_vessel_initial_load)

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
                                                    rule=constr_load_balance)

        def constr_product_load_binary_activator(model, v, p, t):
            return model.l[v, p, t] <= model.vessel_ton_capacities[v] * model.h[v, p, t]

        self.m.constr_product_load_binary_activator = pyo.Constraint(self.m.VESSELS,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_product_load_binary_activator)

        def constr_load_below_vessel_ton_capacity(model, v, t):
            return sum(model.l[v, p, t] for p in model.PRODUCTS) <= model.vessel_ton_capacities[v]

        self.m.constr_load_below_vessel_ton_capacity = pyo.Constraint(self.m.VESSELS,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_load_below_vessel_ton_capacity)

        def constr_load_below_vessel_nprod_capacity(model, v, t):
            return sum(model.h[v, p, t] for p in model.PRODUCTS) <= model.vessel_nprod_capacities[v]

        self.m.constr_load_below_vessel_nprod_capacity = pyo.Constraint(self.m.VESSELS,
                                                                        self.m.TIME_PERIODS,
                                                                        rule=constr_load_below_vessel_nprod_capacity)

        def constr_zero_final_load(model, v, p):
            return model.l[v, p, max(model.TIME_PERIODS)] == 0

        self.m.constr_zero_final_load = pyo.Constraint(self.m.VESSELS,
                                                       self.m.PRODUCTS,
                                                       rule=constr_zero_final_load)

        def constr_inventory_below_capacity(model, i, t):
            return sum(model.r[i, p, t] for p in model.PRODUCTS) <= model.factory_inventory_capacities[i]

        self.m.constr_inventory_below_capacity = pyo.Constraint(self.m.FACTORY_NODES,
                                                                self.m.TIME_PERIODS,
                                                                rule=constr_inventory_below_capacity)

        def constr_initial_inventory(model, i, p):
            return (model.r[i, p, 0] == model.factory_initial_inventories[i, p] +
                    sum(model.q[l, p, 0] for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i) +
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v, i2, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if i2 == i))  # Changed

        self.m.constr_initial_inventory = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         rule=constr_initial_inventory)

        def constr_inventory_balance(model, i, p, t):
            if t == 0:
                return Constraint.Feasible
            return (model.r[i, p, t] == model.r[i, p, (t - 1)] +
                    sum(model.q[l, p, t] for (ii, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if ii == i) +
                    sum(model.demands[i, j, p] * model.z[v, i, j, t]
                        for v, i2, j in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if i2 == i))

        self.m.constr_inventory_balance = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_inventory_balance)

        def constr_production_below_max_capacity(model, i, l, p, t):
            return (model.q[l, p, t]
                    <= model.production_stops[i, t] * model.production_max_capacities[l, p] * model.g[l, p, t])

        self.m.constr_production_below_max_capacity = pyo.Constraint(self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_below_max_capacity)

        def constr_production_above_min_capacity(model, l, p, t):
            return model.q[l, p, t] >= model.production_min_capacities[l, p] * model.g[l, p, t]

        self.m.constr_production_above_min_capacity = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_above_min_capacity)

        def constr_activate_delta(model, l, p, t):
            if t == 0:
                return Constraint.Feasible
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
                return Constraint.Feasible
            relevant_products = {q for (qq, q) in model.PRODUCTS_WITHIN_SAME_PRODUCT_GROUP_TUP if qq == p}
            return model.g[l, p, (t - 1)] <= model.a[l, t] + sum(model.g[l, q, t] for q in relevant_products)

        self.m.constr_production_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                        self.m.PRODUCTS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_production_shift)

        # Extension
        if extended_model:
            def constr_delivery_no_tw_violation(model, i):
                return (sum(model.y_minus[v, i, t]
                            for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j
                            for j, t in model.TIME_WINDOWS_FOR_ORDERS_TUP if i == j)
                        ==
                        model.lambd[i, 0])

            self.m.constr_delivery_within_time_window.deactivate()  # Deactivate the current delivery constraint
            self.m.constr_delivery_no_tw_violation = pyo.Constraint(self.m.ORDER_NODES,
                                                                    rule=constr_delivery_no_tw_violation)

            def constr_delivery_tw_violation_earlier(model, i, k):
                if k < 0 and self.m.tw_min[i] + k in model.TIME_PERIODS:
                    return (sum(model.y_minus[v, i, self.m.tw_min[i] + k]
                                for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j)
                            ==
                            model.lambd[i, k])
                else:
                    return Constraint.Skip

            self.m.constr_delivery_tw_violation_earlier = pyo.Constraint(self.m.ORDER_NODES,
                                                                         self.m.TIME_WINDOW_VIOLATIONS,
                                                                         rule=constr_delivery_tw_violation_earlier)

            def constr_delivery_tw_violation_later(model, i, k):
                if k > 0 and self.m.tw_max[i] + k in model.TIME_PERIODS:
                    return (sum(model.y_minus[v, i, self.m.tw_max[i] + k]
                                for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if i == j)
                            ==
                            model.lambd[i, k])
                else:
                    return Constraint.Skip

            self.m.constr_delivery_tw_violation_later = pyo.Constraint(self.m.ORDER_NODES,
                                                                       self.m.TIME_WINDOW_VIOLATIONS,
                                                                       rule=constr_delivery_tw_violation_later)

            def constr_choose_one_tw_violation(model, i):
                return (sum(model.lambd[i, k]
                            for k in model.TIME_WINDOW_VIOLATIONS
                            if model.tw_max[i] + k in model.TIME_PERIODS
                            and model.tw_min[i] + k in model.TIME_PERIODS)
                        + model.e[i]
                        == 1)

            self.m.constr_choose_one_tw_violation = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_choose_one_tw_violation)

            def constr_wait_if_visit_sick_farm(model, v, i, j, t):
                if model.transport_times[i, j] <= t <= len(model.TIME_PERIODS) - model.min_wait_if_sick[i, j]:
                    return (model.min_wait_if_sick[i, j] * model.x[v, i, j, t - model.transport_times[i, j]]
                            <=
                            sum(model.w[v, j, tau] for tau in range(t, t + model.min_wait_if_sick[i, j])))
                else:
                    return Constraint.Feasible

            self.m.constr_wait_if_visit_sick_farm = pyo.Constraint(self.m.WAIT_EDGES_FOR_VESSEL_TRIP,
                                                                   self.m.TIME_PERIODS,
                                                                   rule=constr_wait_if_visit_sick_farm)

            def constr_rewarded_inventory_below_inventory_level(model, i, p):
                return model.r_plus[i, p] <= model.r[i, p, max(model.TIME_PERIODS)]

            self.m.constr_rewarded_inventory_below_inventory_level = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                    self.m.PRODUCTS,
                                                                                    rule=constr_rewarded_inventory_below_inventory_level)

            def constr_rewarded_inventory_below_inventory_target(model, i, p):
                return model.r_plus[i, p] <= model.inventory_targets[i, p]

            self.m.constr_rewarded_inventory_below_inventory_target = pyo.Constraint(self.m.FACTORY_NODES,
                                                                                     self.m.PRODUCTS,
                                                                                     rule=constr_rewarded_inventory_below_inventory_target)

        print("Done setting constraints!")

        # UNUSED CONSTRAINTS

        # def constr_one_product_per_production_line(model, ll, t):
        #     return sum(model.g[ll, p, t] for p in model.PRODUCTS) <= 1
        #
        # self.m.constr_one_product_per_production_line = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                                self.m.TIME_PERIODS,
        #                                                                rule=constr_one_product_per_production_line)

        # def constr_production_start_or_shift(model, l, q, t):
        #     if t == 0:
        #         return Constraint.Feasible
        #     return (model.delta[l, q, t]
        #             <=
        #             model.a[l, t - 1] + sum(model.gamma[l, p, q, t] for p in model.PRODUCTS if q != p))
        #
        # self.m.constr_production_start_or_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                          self.m.PRODUCTS,
        #                                                          self.m.TIME_PERIODS,
        #                                                          rule=constr_production_start_or_shift)
        #
        # def constr_activate_product_shift(model, l, p, t):
        #     if t == 0:
        #         return Constraint.Feasible
        #     return sum(model.gamma[l, p, q, t] for q in model.PRODUCTS) <= model.g[l, p, t - 1]
        #
        # self.m.constr_activate_product_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                       self.m.PRODUCTS,
        #                                                       self.m.TIME_PERIODS,
        #                                                       rule=constr_activate_product_shift)

        # def constr_restrict_product_shift(model, l, q, t):
        #     return sum(model.gamma[l, p, q, t] for p in model.PRODUCTS) <= model.delta[l, q, t]
        #
        # self.m.constr_restrict_product_shift = pyo.Constraint(self.m.PRODUCTION_LINES,
        #                                                       self.m.PRODUCTS,
        #                                                       self.m.TIME_PERIODS,
        #                                                       rule=constr_restrict_product_shift)

        # def constr_max_one_delivery(model, i):
        #     return (sum(model.y_minus[v, i, t]
        #                 for v, j in model.ORDER_NODES_FOR_VESSELS_TUP if j == i
        #                 for t in model.TIME_PERIODS)
        #             <= 1)
        #
        # self.m.constr_max_one_delivery = pyo.Constraint(self.m.ORDER_NODES,
        #                                                 rule=constr_max_one_delivery)

        # self.m.b = pyo.Var(self.m.ORDER_NODES_FOR_VESSELS_TUP,
        #                    self.m.TIME_PERIODS,
        #                    domain=pyo.Boolean,
        #                    initialize=0)
        #
        # def constr_precedence_yellow(model, v, i, t):
        #     return (sum(model.y_minus[v, j, t] for zone, j in model.ORDERS_FOR_ZONES_TUP if zone == 'yellow')
        #             <=
        #             1 - model.b[v, i, t])
        #
        # self.m.constr_precedence_yellow = pyo.Constraint(self.m.GREEN_NODES_FOR_VESSEL_TUP,
        #                                                  self.m.TIME_PERIODS,
        #                                                  rule=constr_precedence_yellow)
        #
        # def constr_precedence_red(model, v, i, t):
        #     return (sum(model.y_minus[v, j, t] for zone, j in model.ORDERS_FOR_ZONES_TUP if zone == 'red')
        #             <=
        #             1 - model.b[v, i, t])
        #
        # self.m.constr_precedence_red = pyo.Constraint(self.m.GREEN_AND_YELLOW_NODES_FOR_VESSEL_TUP,
        #                                               self.m.TIME_PERIODS,
        #                                               rule=constr_precedence_red)
        #
        # def constr_precedence_allow_visit(model, v, i, t):
        #     return model.y_minus[v, i, t] <= model.b[v, i, t]
        #
        # self.m.constr_precedence_allow_visit = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
        #                                                       self.m.TIME_PERIODS,
        #                                                       rule=constr_precedence_allow_visit)
        #
        # def constr_precedence_visit_after_wait(model, v, i, t):
        #     if t == 0:
        #         return Constraint.Skip
        #     else:
        #         return (model.b[v, i, t]
        #                 <=
        #                 sum(model.y_plus[v, j, t] for v2, j in model.FACTORY_NODES_FOR_VESSELS_TUP if v2 == v) +
        #                 # sum(model.x[v, 'W', j, t] for v2, j in model.NODES_FOR_VESSELS_TUP if v2 == v) +
        #                 model.b[v, i, t-1])
        #
        #
        # self.m.constr_precedence_visit_after_wait = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
        #                                                            self.m.TIME_PERIODS,
        #                                                            rule=constr_precedence_visit_after_wait)

    def solve(self, verbose: bool = True, time_limit: int = None) -> None:
        print("Solver running...")
        if time_limit:
            self.solver_factory.options['TimeLimit'] = time_limit  # time limit in seconds
        t = time()
        self.results = self.solver_factory.solve(self.m, tee=verbose)  # logfile=f'../../log_files/console_output_{log_name}.log'
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Not optimal termination condition: ", self.results.solver.termination_condition)
        print("Solve time: ", round(time() - t, 1))

    def print_result(self):

        def print_result_variablewise():
            def print_vessel_routing():
                print("VESSEL ROUTING (x variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for i in self.m.NODES_INCLUDING_DUMMIES:
                            for j in self.m.NODES_INCLUDING_DUMMIES:
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

            def print_order_delivery():
                print("ORDER DELIVERY (y_minus variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for i in self.m.ORDER_NODES:
                            if pyo.value(self.m.y_minus[v, i, t]) >= 0.5:
                                print("Vessel", v, "starts unloading order", i, "in time period", t)
                print()

            def print_order_pickup():
                print("ORDER PICKUP (y_plus variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS:
                        for i in self.m.FACTORY_NODES:
                            if pyo.value(self.m.y_plus[v, i, t]) >= 0.5:
                                print("Vessel", v, "starts loading at factory", i, "in time period", t)
                print()

            def print_factory_production():
                print("FACTORY PRODUCTION (q variable)")
                production = False
                for t in self.m.TIME_PERIODS:
                    for l in self.m.PRODUCTION_LINES:
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.q[l, p, t]) >= 0.5:
                                print("Production line", l, "produces", pyo.value(self.m.q[l, p, t]), "tons of product",
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
                            if pyo.value(self.m.r[i, p, t]) >= 0.5:
                                print("Factory", i, "holds", pyo.value(self.m.r[i, p, t]), "tons of product", p,
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
                                          pyo.value(self.m.q[l, p, t]), " is produced", sep="")
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
                            print("Rewarded inventory of product ", p, " is ", pyo.value(self.m.r_plus[i, p]),
                                  ", total final inventory is ", pyo.value(self.m.r[i, p, (max(self.m.TIME_PERIODS))]),
                                  " and its target is ", pyo.value(self.m.inventory_targets[i, p]), sep="")
                        else:
                            print("Final inventory for", p, "is", pyo.value(self.m.r[i, p, (max(self.m.TIME_PERIODS))]))
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
                        s = "" if k <= 0 else "+"
                        print(s + str(k), "violation:", orders_with_k_violation)
                else:
                    print("No time window violation, extension is not applied")
                print()

            # PRINTING
            print()
            # print_factory_production()
            # print_factory_inventory()
            # print_vessel_routing()
            # print_order_delivery()
            # print_order_pickup()
            # print_factory_pickup()
            # print_waiting()
            # print_vessel_load()
            print_orders_not_delivered()
            print_production_starts()
            print_production_happens()
            print_final_inventory()
            # print_available_production_lines()
            print_time_window_violations()

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
                            if self.m.y_plus[v, i, t]() > 0.5 or self.m.y_minus[v, i, t]() > 0.5:
                                row.append(i)  # load
                                action_in_period = True
                            if self.m.w[v, i, t]() > 0.5:
                                row.append('.')  # wait
                                action_in_period = True
                            for j in self.m.NODES:
                                if self.m.x[v, i, j, t]() > 0.5:
                                    row.append(">")  # sail
                                    action_in_period = True
                            if self.m.x[v, i, 'd_-1', t]() >= 0.5:  # route ends
                                row.append(i)
                                action_in_period = True
                        if not action_in_period:
                            row.append(" ")


                    table.append(row)
                print(tabulate(table, headers=["vessel"] + list(self.m.TIME_PERIODS)))
                print()

            def print_routing(include_loads=True):
                for v in self.m.VESSELS:
                    print("ROUTING OF VESSEL", v)
                    for t in self.m.TIME_PERIODS:
                        curr_load = [round(self.m.l[v, p, t]()) for p in self.m.PRODUCTS]
                        # x variable
                        for i in self.m.NODES_INCLUDING_DUMMIES:
                            for j in self.m.NODES_INCLUDING_DUMMIES:
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
                            # y_plus variable
                            if pyo.value(self.m.y_plus[v, i, t]) >= 0.5:
                                print(t, ": loads in node ", i, sep="")
                                if include_loads:
                                    print("   load: ", curr_load)
                            # y_minus variable
                            if pyo.value(self.m.y_minus[v, i, t]) >= 0.5:
                                print(t, ": unloads in node ", i, sep="")
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
                        production = [round(sum(self.m.q[l, p, t]() for l in relevant_production_lines))
                                      for p in sorted(self.m.PRODUCTS)]
                        inventory = [round(self.m.r[i, p, t]()) for p in sorted(self.m.PRODUCTS)]
                        if sum(production) > 0.5:
                            print(t, ": production: \t", production, sep="")
                        print(t, ": inventory: \t", inventory, sep="")
                        # for p in self.m.PRODUCTS:
                        #     if pyo.value(self.m.q[i, p, t]) >= 0.5:
                        #         print(t, ": production of ", round(pyo.value(self.m.q[i, p, t]), 1),
                        #               " tons of product ",
                        #               p, sep="")
                        #     if pyo.value(self.m.r[i, p, t]) >= 0.5:
                        #         print(t, ": inventory level is ", round(pyo.value(self.m.r[i, p, t]), 1),
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
                                row.append(round(sum(self.m.q[l, p, t]() for l in relevant_production_lines))) # + " [" + str(self.m.r[i, p, t]()) + "]")
                            else:
                                row.append(" ")
                        table.append(row)
                        row = ["\"", "inv"]
                        for t in self.m.TIME_PERIODS:
                            if t == 0:
                                row.append(round(self.m.r[i, p, 0]()))
                            elif abs(self.m.r[i, p, t]() - self.m.r[i, p, t-1]()) > 0.5:
                                row.append(round(self.m.r[i, p, t]()))  # + " [" + str(self.m.r[i, p, t]()) + "]")
                            else:
                                row.append(" ")
                        table.append(row)
                        table.append(["____"] * (len(list(self.m.TIME_PERIODS)) + 2))
                    print(tabulate(table, headers=["product", "prod/inv"] + list(self.m.TIME_PERIODS)))
                    print()

            print_routing(include_loads=False)
            # print_vessel_load()
            # print_production_and_inventory()
            print_production_simple()
            print_routes_simple()

        def print_objective_function_components():
            inventory_cost = (sum(self.m.inventory_unit_costs[i] * pyo.value(self.m.r[i, p, t])
                                  for t in self.m.TIME_PERIODS
                                  for p in self.m.PRODUCTS
                                  for i in self.m.FACTORY_NODES))
            transport_cost = (
                sum(self.m.transport_unit_costs[v] * self.m.transport_times[i, j] * pyo.value(self.m.x[v, i, j, t])
                    for t in self.m.TIME_PERIODS
                    for j in self.m.NODES
                    for i in self.m.NODES
                    for v in self.m.VESSELS))

            production_start_cost = (sum(self.m.production_start_costs[i, p] * pyo.value(self.m.delta[l, p, t])
                                         for i in self.m.FACTORY_NODES
                                         for (ii, l) in self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP if i == ii
                                         for p in self.m.PRODUCTS
                                         for t in self.m.TIME_PERIODS))
            unmet_order_cost = (sum(self.m.external_delivery_penalty * pyo.value(self.m.e[i])
                                    for i in self.m.ORDER_NODES))

            sum_obj = inventory_cost + transport_cost + unmet_order_cost + production_start_cost
            if self.extended_model:
                time_window_violation_cost = (sum(self.m.time_window_violation_cost[k] * self.m.lambd[i, k]()
                                                  for i in self.m.ORDER_NODES
                                                  for k in self.m.TIME_WINDOW_VIOLATIONS))
                final_inventory_reward = (-sum(self.m.inventory_unit_rewards[i] * pyo.value(self.m.r_plus[i, p])
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

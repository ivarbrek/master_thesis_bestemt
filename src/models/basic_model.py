import pyomo.environ as pyo
from typing import Dict, List
from time import time
from pyomo.core import Constraint


# TODO IN THIS FILE
# ----------------------------------------------------------------------------------------------------------------------------

# TODO: pipenv with Python 3.8?
# TODO: Have not made any difference between different start/end time periods (so for now, not possible to have different time period lengths)
# TODO: Check code marked with TODO
# TODO: Add features (extensions)

class BasicModel:

    def __init__(self,
                 nodes: List,
                 factory_nodes: List,
                 order_nodes: List,
                 order_nodes_for_factories: Dict,
                 nodes_for_vessels: Dict,
                 products: List,
                 vessels: List,
                 time_periods: List,
                 time_periods_for_vessels: Dict,
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
                 production_unit_costs: Dict,
                 production_min_capacities: Dict,
                 production_max_capacities: Dict,
                 production_lines: List,
                 production_lines_for_factories: List,
                 production_line_min_times: Dict
                 ) -> None:

        # GENERAL MODEL SETUP
        self.m = pyo.ConcreteModel()

        # TODO: Make sure that this is included
        # self._create_sets(nodes=nodes,
        #                   factory_nodes=factory_nodes,
        #                   order_nodes=order_nodes,
        #                   order_nodes_for_factories=order_nodes_for_factories,
        #                   nodes_for_vessels=nodes_for_vessels,
        #                   products=products,
        #                   vessels=vessels,
        #                   time_periods=time_periods,
        #                   time_periods_for_vessels=time_periods_for_vessels,
        #                   time_windows_for_orders=time_windows_for_orders,
        #                   production_lines=production_lines,
        #                   production_lines_for_factories=production_lines_for_factories)

        # self._set_parameters(vessel_ton_capacities=vessel_ton_capacities,
        #                      vessel_nprod_capacities=vessel_nprod_capacities,
        #                      vessel_initial_loads=vessel_initial_loads,
        #                      production_min_capacities=production_min_capacities,
        #                      production_max_capacities=production_max_capacities,
        #                      production_unit_costs=production_unit_costs,
        #                      production_line_min_times=production_line_min_times,
        #                      # C^S,
        #                      factory_inventory_capacities=factory_inventory_capacities,
        #                      factory_initial_inventories=factory_initial_inventories,
        #                      inventory_unit_costs=inventory_unit_costs,
        #                      demands=demands,
        #                      transport_unit_costs=transport_unit_costs,
        #                      transport_times=transport_times,
        #                      # M^V,
        #                      # M^D,
        #                      unloading_times=unloading_times,
        #                      loading_times=loading_times
        #                      )

        # self._set_variables()
        # self._set_objective()
        # self._set_constraints()

        self.solver_factory = pyo.SolverFactory('gurobi')
        self.results = None
        self.solution = None

        ################################################################################################################
        # SETS #########################################################################################################
        self.m.NODES = pyo.Set(initialize=nodes)
        self.m.NODES_INCLUDING_DUMMIES = pyo.Set(
            initialize=nodes + ['d_0', 'd_-1'])  # d_0 is dummy origin, d_-1 is dummy destination
        self.m.NODES_INCLUDING_DUMMY_START = pyo.Set(initialize=nodes + ['d_0'])
        self.m.NODES_INCLUDING_DUMMY_END = pyo.Set(initialize=nodes + ['d_-1'])
        self.m.FACTORY_NODES = pyo.Set(initialize=factory_nodes)
        self.m.ORDER_NODES = pyo.Set(initialize=order_nodes)

        self.m.PRODUCTS = pyo.Set(initialize=products)
        self.m.VESSELS = pyo.Set(initialize=vessels)
        self.m.TIME_PERIODS = pyo.Set(initialize=time_periods)

        last_time_period = max(time_periods)
        last_dummy_time_period = last_time_period + max(transport_times.values())
        dummy_time_periods = [i for i in range(max(time_periods) + 1, last_dummy_time_period + 1)]
        self.m.DUMMY_TIME_PERIODS = pyo.Set(initialize=dummy_time_periods)

        self.m.TIME_PERIODS_INCLUDING_DUMMY = pyo.Set(
            initialize=time_periods + dummy_time_periods)

        order_nodes_for_factories_tup = [(factory, order_node)
                                         for (factory, order_node) in
                                         order_nodes_for_factories.keys()
                                         if order_nodes_for_factories[factory, order_node] == 1]
        self.m.ORDER_NODES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=order_nodes_for_factories_tup)

        nodes_for_orders_tup = [(order_node, factory_node)
                                for (factory_node, order_node) in order_nodes_for_factories_tup] + [
                                   (order_node, order_node) for order_node in order_nodes]
        self.m.NODES_FOR_ORDERS_TUP = pyo.Set(dimen=2, initialize=nodes_for_orders_tup)

        orders_related_to_nodes_tup = [(node, order_node)
                                       for (order_node, node) in nodes_for_orders_tup]
        self.m.ORDERS_RELATED_TO_NODES_TUP = pyo.Set(dimen=2, initialize=orders_related_to_nodes_tup)

        nodes_for_vessels_tup = [(vessel, node)
                                 for (vessel, node) in nodes_for_vessels.keys()
                                 if nodes_for_vessels[vessel, node] == 1]
        self.m.NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=nodes_for_vessels_tup)

        nodes_nodes_for_vessels_trip = [(vessel, node, node) for (vessel, node) in nodes_for_vessels_tup] + [
            (vessel, 'd_0', node) for (vessel, node) in nodes_for_vessels_tup]
        self.m.NODES_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3, initialize=nodes_nodes_for_vessels_trip)

        factory_nodes_for_vessels_tup = [(vessel, node)
                                         for (vessel, node) in
                                         nodes_for_vessels_tup
                                         if node in factory_nodes]
        self.m.FACTORY_NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=factory_nodes_for_vessels_tup)

        order_nodes_for_vessels_tup = [(vessel, node)
                                       for (vessel, node) in
                                       nodes_for_vessels_tup
                                       if node in order_nodes]
        self.m.ORDER_NODES_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=order_nodes_for_vessels_tup)

        vessels_relevantnodes_ordernodes = [(vessel, relevant_node, order_node)
                                            for vessel, order_node in order_nodes_for_vessels_tup
                                            for order_node2, relevant_node in nodes_for_orders_tup
                                            if order_node2 == order_node
                                            and (vessel, relevant_node) in nodes_for_vessels_tup
                                            ]

        self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3,
                                                                     initialize=vessels_relevantnodes_ordernodes)

        vessels_factorynodes_ordernodes = [(vessel, factory_node, order_node)
                                           for vessel, order_node in order_nodes_for_vessels_tup
                                           for factory_node, order_node2 in order_nodes_for_factories_tup
                                           if order_node2 == order_node
                                           and (vessel, factory_node) in nodes_for_vessels_tup
                                           ]

        self.m.ORDER_NODES_FACTORY_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3, initialize=vessels_factorynodes_ordernodes)

        vessels_for_nodes_tup = [(node, vessel)
                                 for (vessel, node) in nodes_for_vessels_tup]
        self.m.VESSELS_FOR_NODES_TUP = pyo.Set(dimen=2, initialize=vessels_for_nodes_tup)

        vessels_for_factory_nodes_tup = [(node, vessel)
                                         for (vessel, node) in nodes_for_vessels_tup
                                         if node in factory_nodes]
        self.m.VESSELS_FOR_FACTORY_NODES_TUP = pyo.Set(dimen=2, initialize=vessels_for_factory_nodes_tup)

        order_nodes_for_factory_nodes_for_vessels_trip = [(vessel, factory_node, order_node)
                                                          for vessel in vessels
                                                          for (factory_node, v) in vessels_for_factory_nodes_tup if
                                                          v == vessel
                                                          for order_node in
                                                          {o for (u, o) in self.m.ORDER_NODES_FOR_VESSELS_TUP if
                                                           u == vessel}.intersection(
                                                              {o for (f, o) in self.m.ORDER_NODES_FOR_FACTORIES_TUP if
                                                               f == factory_node})]

        self.m.ORDER_NODES_FOR_FACTORY_NODES_FOR_VESSELS_TRIP = pyo.Set(dimen=3,
                                                                        initialize=order_nodes_for_factory_nodes_for_vessels_trip)

        # TODO: Comment in or remove
        # order_nodes_for_factory_nodes_not_vessels = [(vessel, factory_node, order_node)
        #                                              for vessel in vessels
        #                                              for (factory_node, v) in vessels_for_factory_nodes_tup
        #                                              if v == vessel
        #                                              for order_node in
        #                                              ]
        # self.m.ORDER_NODES_FOR_FACTORY_NODES_NOT_FOR_VESSELS = pyo.Set(dimen=3,
        #                                                                initialize=order_nodes_for_factory_nodes_not_vessels)


        time_windows_for_orders_tup = [(order, time_period)
                                       for (order, time_period) in
                                       time_windows_for_orders.keys()
                                       if time_windows_for_orders[order, time_period] == 1]
        self.m.TIME_WINDOWS_FOR_ORDERS_TUP = pyo.Set(dimen=2, initialize=time_windows_for_orders_tup)

        time_periods_for_vessels_tup = [(vessel, time_period)
                                        for (vessel, time_period) in
                                        time_periods_for_vessels.keys()
                                        if time_periods_for_vessels[vessel, time_period] == 1]
        self.m.TIME_PERIODS_FOR_VESSELS_TUP = pyo.Set(dimen=2, initialize=time_periods_for_vessels_tup)

        nodes_time_periods_for_vessels_tup = [(vessel, node, time_period)
                                              for vessel, node in vessels_for_nodes_tup
                                              for vessel2, time_period in time_periods_for_vessels_tup
                                              if vessel == vessel2]
        self.m.NODES_TIME_PERIODS_FOR_VESSELS_TUP = pyo.Set(dimen=3, initialize=nodes_time_periods_for_vessels_tup)

        self.m.PRODUCTION_LINES = pyo.Set(initialize=production_lines)

        self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=production_lines_for_factories)

        print("Done setting sets!")

        # TODO: Make sure that this is included
    # def _set_parameters(self,
    #                     vessel_ton_capacities,
    #                     vessel_nprod_capacities,
    #                     vessel_initial_loads,
    #                     factory_inventory_capacities,
    #                     factory_initial_inventories,
    #                     inventory_unit_costs,
    #                     transport_unit_costs,
    #                     transport_times,
    #                     unloading_times,
    #                     loading_times,
    #                     demands,
    #                     production_unit_costs,
    #                     production_min_capacities,
    #                     production_max_capacities,
    #                     production_line_min_times):

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

        self.m.production_unit_costs = pyo.Param(self.m.FACTORY_NODES,
                                                 self.m.PRODUCTS,
                                                 initialize=production_unit_costs)

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

        print("Done setting parameters!")

        ################################################################################################################
        # VARIABLES ####################################################################################################
        self.m.x = pyo.Var(self.m.VESSELS,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.TIME_PERIODS_INCLUDING_DUMMY,
                           domain=pyo.Boolean,
                           initialize=0)  # OK

        self.m.w = pyo.Var(self.m.VESSELS,
                           self.m.NODES,
                           self.m.TIME_PERIODS_INCLUDING_DUMMY,
                           domain=pyo.Boolean,
                           initialize=0)  # Remove dummy from T?

        self.m.y_plus = pyo.Var(self.m.NODES_FOR_VESSELS_TUP,
                                self.m.TIME_PERIODS,
                                domain=pyo.Boolean,
                                initialize=0)  # OK

        self.m.y_minus = pyo.Var(self.m.NODES_FOR_VESSELS_TUP,
                                 self.m.TIME_PERIODS,
                                 domain=pyo.Boolean,
                                 initialize=0)  # OK


        # TODO: Change name
        #self.m.z = pyo.Var(self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP,  # CHANGED: ORDERS_RELATED_TO_NODES_TUP
        self.m.z = pyo.Var(self.m.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TUP,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.l = pyo.Var(self.m.VESSELS,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals)  # OK

        self.m.h = pyo.Var(self.m.VESSELS,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)  # OK

        self.m.q = pyo.Var(self.m.FACTORY_NODES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals,
                           initialize=0)  # OK

        self.m.g = pyo.Var(self.m.PRODUCTION_LINES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.Boolean,
                           initialize=0)  # OK

        self.m.r = pyo.Var(self.m.FACTORY_NODES,
                           self.m.PRODUCTS,
                           self.m.TIME_PERIODS,
                           domain=pyo.NonNegativeReals)  # OK

        self.m.e = pyo.Var(self.m.ORDER_NODES,
                           domain=pyo.Boolean,
                           initialize=0)  # To be removed, implemented to avoid infeasibility during testing

        # TODO: Add variables, a, delta, gamma

        print("Done setting variables!")

        ################################################################################################################
        # OBJECTIVE ####################################################################################################
        def obj(model):
            return (sum(model.production_unit_costs[i, p] * model.q[i, p, t] +
                        model.inventory_unit_costs[i] * model.r[i, p, t]
                        for t in model.TIME_PERIODS
                        for p in model.PRODUCTS
                        for i in model.FACTORY_NODES)
                    + sum(model.transport_unit_costs[v] * model.transport_times[i, j] * model.x[v, i, j, t]
                          for t in model.TIME_PERIODS_INCLUDING_DUMMY
                          for j in model.NODES
                          # TODO: Perhaps change this set so that only allowed nodes for v are summed over
                          for i in model.NODES
                          for v in model.VESSELS)
                    + sum(100000000000 * model.e[i] for i in model.ORDER_NODES))  # TODO: Remove when done with testing
            # TODO: Add to objective: cost of switching products

        self.m.objective = pyo.Objective(rule=obj, sense=pyo.minimize)
        # self.m.objective.pprint()

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

        def constr_max_one_activity_after_planning_horizon(model, v, t):
            nodes = {n for (vessel, n) in model.NODES_FOR_VESSELS_TUP if vessel == v}.union({'d_-1'})
            return sum(model.x[v, i, j, t] for j in nodes for i in nodes) <= 1

        self.m.constr_max_one_activity_after_planning_horizon = pyo.Constraint(self.m.VESSELS,
                                                                               self.m.TIME_PERIODS_INCLUDING_DUMMY,
                                                                               rule=constr_max_one_activity_after_planning_horizon)

        def constr_max_one_vessel_loading(model, i, v, t):
            relevant_vessels = {vessel for (j, vessel) in model.VESSELS_FOR_FACTORY_NODES_TUP if j == i and vessel != v}
            relevant_time_periods = {tau for tau in model.TIME_PERIODS if (t <= tau <= t + model.loading_times[v, i])}
            return (sum(model.y_plus[u, i, tau]
                        for tau in relevant_time_periods
                        for u in relevant_vessels)
                    <= 1 - model.y_plus[v, i, t])  # TODO: Add parameter M^V, otherwise OK

        self.m.constr_max_one_vessel_loading = pyo.Constraint(self.m.VESSELS_FOR_FACTORY_NODES_TUP,
                                                              self.m.TIME_PERIODS,
                                                              rule=constr_max_one_vessel_loading)

        def constr_delivery_within_time_window(model, i):  # TODO: Remove e when done with testing
            relevant_vessels = {vessel for (vessel, j) in model.ORDER_NODES_FOR_VESSELS_TUP if j == i}
            relevant_time_periods = {t for (j, t) in model.TIME_WINDOWS_FOR_ORDERS_TUP if j == i}
            return (sum(model.y_minus[v, i, t] for v in relevant_vessels for t in relevant_time_periods) + model.e[i]
                    == 1)

        self.m.constr_delivery_within_time_window = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_delivery_within_time_window)  # OK

        def constr_sailing_after_loading_unloading(model, v, i, t):
            unloading_time = pyo.value(model.unloading_times[v, i])
            loading_time = pyo.value(model.loading_times[v, i])
            relevant_destination_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP if vessel == v}
            relevant_destination_nodes.add('d_-1')  # Adds the dummy end node

            # This simplification didn't work out...
            # if max(model.loading_times[v, i], model.unloading_times[v, i]) > t or t == -1:
            #     return Constraint.Feasible
            # else:
            #     return ((model.y_minus[v, i, (t - unloading_time)] + model.y_plus[v, i, (t - loading_time)])
            #             == sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

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
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP if
                              vessel == v and model.transport_times[j, i] <= t}.union({'d_0'})
            if t == 0:
                return (sum(model.x[v, j, i, (t - model.transport_times[j, i])] for j in relevant_nodes)
                        ==
                        model.y_minus[v, i, t] + model.y_plus[v, i, t] + model.w[v, i, t] + model.x[v, i, 'd_-1', t])
            else:
                return (sum(model.x[v, j, i, (t - model.transport_times[j, i])] for j in relevant_nodes) +
                        model.w[v, i, (t - 1)]
                        ==
                        model.y_minus[v, i, t] + model.y_plus[v, i, t] + model.w[v, i, t] + model.x[v, i, 'd_-1', t])

        self.m.constr_wait_load_unload_after_sailing = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_wait_load_unload_after_sailing)  # OK

        # TODO: Comment in or remove
        # Note: Due to readability this constraint is not in the Overleaf formulation
        # def constr_no_waiting_in_dummy_end_period(model):
        #     return sum(model.w[v, i, -1] for v in model.VESSELS for i in model.NODES) == 0

        def constr_sail_to_dest_after_planning_horizon(model, v, i, t):
            return (sum(model.x[v, j, i, (t - model.transport_times[j, i])]
                        for vessel, j in model.NODES_FOR_VESSELS_TUP
                        if vessel == v)
                    ==
                    model.x[v, i, 'd_-1', t])

        self.m.constr_sail_to_dest_after_planning_horizon = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                           self.m.DUMMY_TIME_PERIODS,
                                                                           rule=constr_sail_to_dest_after_planning_horizon)

        def constr_start_route(model, v):
            return model.x[v, 'd_0', 'f_1', 0] == 1  # TODO: Replace with vessel dependent parameters

        self.m.constr_start_route = pyo.Constraint(self.m.VESSELS, rule=constr_start_route)

        def constr_start_route_once(model, v):
            return (sum(model.x[v, 'd_0', j, t]
                        for j in model.NODES_INCLUDING_DUMMIES
                        for t in model.TIME_PERIODS_INCLUDING_DUMMY)
                    == 1)


        # def constr_only_sail_in_relevant_time_periods(model, v):
        #     relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
        #                       if vessel == v}
        #     relevant_time_periods = {t for (vessel, t) in model.TIME_PERIODS_FOR_VESSELS_TUP
        #                              if vessel == v}
        #     irrelevant_time_periods = set(model.TIME_PERIODS_INCLUDING_DUMMY) - relevant_time_periods
        #     return (sum(model.x[v, 'd_0', j, t] for t in irrelevant_time_periods for j in relevant_nodes)
        #             == 0)

        # self.m.constr_only_sail_in_relevant_time_periods = pyo.Constraint(self.m.VESSELS,
        #                                                                   rule=constr_only_sail_in_relevant_time_periods)  # OK

        self.m.constr_start_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_start_route_once)

        def constr_end_route_once(model, v):
            relevant_nodes = {i for (vessel, i) in model.FACTORY_NODES_FOR_VESSELS_TUP
                              if vessel == v}
            return (sum(model.x[v, i, 'd_-1', t] for t in model.TIME_PERIODS_INCLUDING_DUMMY for i in relevant_nodes))


        def constr_end_route_once(model, v):
            return (sum(model.x[v, i, 'd_-1', t]
                        for t in model.TIME_PERIODS_INCLUDING_DUMMY
                        for vessel, i in model.FACTORY_NODES_FOR_VESSELS_TUP
                        if vessel == v) == 1)

        self.m.constr_end_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_end_route_once)  # OK

        # TODO: Many new constraints!
        # TODO: Add constraint (3.10), restrictions on number of vessels with a factory as destination

        def constr_pickup_requires_factory_visit(model, v, i, j, t):
            return model.z[v, i, j, t] <= model.y_plus[v, i, t]

        self.m.constr_pickup_requires_factory_visit = pyo.Constraint(self.m.ORDER_NODES_FACTORY_NODES_FOR_VESSELS_TRIP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_pickup_requires_factory_visit)  # OK

        def constr_delivery_requires_order_visit(model, v, i, t):
            return model.z[v, i, i, t] == model.y_minus[v, i, t]

        self.m.constr_delivery_requires_order_visit = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_delivery_requires_order_visit)  # OK

        def constr_vessel_initial_load(model, v, p):
            return (model.l[v, p, 0] == model.vessel_initial_loads[v, p] -
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v2, i, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if v2 == v))  # TODO: Check if correct (but I think good)

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

        def constr_inventory_below_capacity(model, i, t):
            return sum(model.r[i, p, t] for p in model.PRODUCTS) <= model.factory_inventory_capacities[i]

        self.m.constr_inventory_below_capacity = pyo.Constraint(self.m.FACTORY_NODES,
                                                                self.m.TIME_PERIODS,
                                                                rule=constr_inventory_below_capacity)

        def constr_zero_final_load(model, v, p): # Added
            return model.l[v, p, max(model.TIME_PERIODS)] == 0

        self.m.contr_zero_final_load = pyo.Constraint(self.m.VESSELS,
                                                      self.m.PRODUCTS,
                                                      rule=constr_zero_final_load)

        def constr_initial_inventory(model, i, p):
            return (model.r[i, p, 0] == model.factory_initial_inventories[i, p] + model.q[i, p, 0] +
                    sum(model.demands[i, j, p] * model.z[v, i, j, 0]
                        for (v, i2, j) in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if i2 == i)) # Changed

        self.m.constr_initial_inventory = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         rule=constr_initial_inventory)

        def constr_inventory_balance(model, i, p, t):
            if t == 0:
                return Constraint.Feasible
            return (model.r[i, p, t] == model.r[i, p, (t - 1)] + model.q[i, p, t] +
                    sum(model.demands[i, j, p] * model.z[v, i, j, t]
                        for v, i2, j in model.ORDER_NODES_RELEVANT_NODES_FOR_VESSELS_TRIP
                        if i2 == i))

        self.m.constr_inventory_balance = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_inventory_balance)

        def constr_production_below_max_capacity(model, i, p, t):
            relevant_production_lines = {l for (factory, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if factory == i}
            return (model.q[i, p, t]
                    <=
                    sum(model.production_max_capacities[ll, p] * model.g[ll, p, t] for ll in relevant_production_lines))

        self.m.constr_production_below_max_capacity = pyo.Constraint(self.m.FACTORY_NODES,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_below_max_capacity)

        def constr_production_above_min_capacity(model, i, ll, p, t):
            return model.q[i, p, t] >= model.production_min_capacities[ll, p] * model.g[ll, p, t]

        self.m.constr_production_above_min_capacity = pyo.Constraint(self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP,
                                                                     self.m.PRODUCTS,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_production_above_min_capacity)

        def constr_one_product_per_production_line(model, ll, t):
            return sum(model.g[ll, p, t] for p in model.PRODUCTS) <= 1

        self.m.constr_one_product_per_production_line = pyo.Constraint(self.m.PRODUCTION_LINES,
                                                                       self.m.TIME_PERIODS,
                                                                       rule=constr_one_product_per_production_line)

        # TODO: Add more constraints (new production features)

        print("Done setting constraints!")

    def solve(self):
        print("Solver running...")
        t = time()
        self.results = self.solver_factory.solve(self.m, tee=True)
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("Termination condition not optimal, ", self.results.solver.termination_condition)
        print("Solve time: ", round(time() - t, 1))
        for v in self.m.VESSELS:
            print(v, ": ", self.m.x[v, 'd_0', 'f_1', 3]())

    def print_result(self):

        def print_result_variablewise():
            def print_vessel_routing():
                print("VESSEL ROUTING (x variable)")
                for v in self.m.VESSELS:
                    for t in self.m.TIME_PERIODS_INCLUDING_DUMMY:
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
                    for i in self.m.FACTORY_NODES:
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.q[i, p, t]) >= 0.5:
                                print("Factory", i, "produces", pyo.value(self.m.q[i, p, t]), "tons of product", p,
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
                            for i in {n for (o, n) in self.m.NODES_FOR_ORDERS_TUP if o == j}:
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

            def print_orders_not_delivered():  # TODO: Remove when done with testing
                all_delivered = True
                print("ORDERS NOT DELIVERED (e variable)")
                for i in self.m.ORDER_NODES:
                    if pyo.value(self.m.e[i]) >= 0.5:
                        print("Order", i, "is not delivered")
                        all_delivered = False
                if all_delivered:
                    print("All orders have been delivered")
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

        def print_result_eventwise():

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

                    # x variable is defined also for the dummy end node, d_-1
                    for i in self.m.NODES_INCLUDING_DUMMIES:
                        for j in self.m.NODES_INCLUDING_DUMMIES:
                            for t in self.m.DUMMY_TIME_PERIODS:
                                if pyo.value(self.m.x[v, i, j, t]) >= 0.5:
                                    print(t, ": ", i, " --> ", j, sep="")
                    # Space before next vessel
                    print()

            def print_vessel_load():
                for v in self.m.VESSELS:
                    print("LOAD AT VESSEL", v)
                    for t in self.m.TIME_PERIODS:
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.l[v, p, t]) >= 0.5:
                                print(t, ": load of ", round(pyo.value(self.m.l[v, p, t]), 1), " tons of product ", p,
                                      sep="")
                    print()

            def print_production_and_inventory():
                for i in self.m.FACTORY_NODES:
                    print("PRODUCTION AND INVENTORY AT FACTORY", i)
                    for t in self.m.TIME_PERIODS:
                        production = [round(self.m.q[i, p, t]()) for p in self.m.PRODUCTS]
                        inventory = [round(self.m.r[i, p, t]()) for p in self.m.PRODUCTS]
                        if sum(production) > 0.5:
                            print(t, ": production:", production, sep="")
                        print(t, ": inventory:", inventory, sep="")
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

            print_routing()
            print_vessel_load()
            print_production_and_inventory()

        print_result_variablewise()
        print_result_eventwise()

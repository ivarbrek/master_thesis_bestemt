import pyomo.environ as pyo
from typing import Dict, List
from time import time
from pyomo.core import Constraint

# TODO IN THIS FILE
# ----------------------------------------------------------------------------------------------------------------------------
# TODO: pipenv with Python 3.8?
# TODO: Should we have a cost on vessel load? As for now, it is free to have "inventories" on the vessels. Results in some quite strange routing (e.g. when initial factory inventories are large).
# TODO: Suggest to skip lead time (currently not included here)
# TODO: Have not made any difference between different start/end time periods (so for now, not possible to have different time period lengths)
# TODO: Remove the e variable (and the related stuff) when done with testing
# TODO: Some other comments in the code (typically suggestions to formulation fixes)
# TODO: Constraint constr_delivery_requires_pickup - see comments there and fix 'em
# TODO: Problem with routing still... A vessel may 1) sail from dummy start to a node n_1, 2) sail from n_1 to another node n_2 and 3) start unloading in a node n_1 in the same time period
    # TODO cont.: In the next time period it does operations in two nodes (e.g. loading/unloading), n_1 and n_2
    # TODO cond.: Also, sometimes a n_1 --> n_1 flow happens, which messes things up...
# TODO: Big M stuff
# TODO: Check load OK?


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
                 production_lines_for_factories: List
                 ) -> None:
        self.m = pyo.ConcreteModel()
        self._create_sets(nodes=nodes,
                          factory_nodes=factory_nodes,
                          order_nodes=order_nodes,
                          order_nodes_for_factories=order_nodes_for_factories,
                          nodes_for_vessels=nodes_for_vessels,
                          products=products,
                          vessels=vessels,
                          time_periods=time_periods,
                          time_periods_for_vessels=time_periods_for_vessels,
                          time_windows_for_orders=time_windows_for_orders,
                          production_lines=production_lines,
                          production_lines_for_factories=production_lines_for_factories)

        self._set_parameters(vessel_ton_capacities=vessel_ton_capacities,
                             vessel_nprod_capacities=vessel_nprod_capacities,
                             vessel_initial_loads=vessel_initial_loads,
                             factory_inventory_capacities=factory_inventory_capacities,
                             factory_initial_inventories=factory_initial_inventories,
                             inventory_unit_costs=inventory_unit_costs,
                             transport_unit_costs=transport_unit_costs,
                             transport_times=transport_times,
                             unloading_times=unloading_times,
                             loading_times=loading_times,
                             demands=demands,
                             production_unit_costs=production_unit_costs,
                             production_min_capacities=production_min_capacities,
                             production_max_capacities=production_max_capacities)

        self._set_variables()
        self._set_objective()
        self._set_constraints()
        self.solver_factory = pyo.SolverFactory('gurobi')
        self.results = None
        self.solution = None

    def _create_sets(self,
                     factory_nodes,
                     order_nodes,
                     nodes,
                     order_nodes_for_factories,
                     nodes_for_vessels,
                     products,
                     vessels,
                     time_periods,
                     time_periods_for_vessels,
                     time_windows_for_orders,
                     production_lines,
                     production_lines_for_factories):
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

        order_nodes_for_factories_tup = [(factory, order_node)
                                         for (factory, order_node) in
                                         order_nodes_for_factories.keys()
                                         if order_nodes_for_factories[factory, order_node] == 1]
        self.m.ORDER_NODES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=order_nodes_for_factories_tup)

        nodes_for_orders_tup = [(order_node, factory_node)
                                       for (factory_node, order_node) in order_nodes_for_factories_tup] + [
                                          (order_node, order_node)
                                          for order_node in
                                          order_nodes]
        self.m.NODES_FOR_ORDERS_TUP = pyo.Set(dimen=2, initialize=nodes_for_orders_tup)

        orders_related_to_nodes_tup = [(node, order_node)
                                       for (order_node, node) in nodes_for_orders_tup]
        self.m.ORDERS_RELATED_TO_NODES_TUP = pyo.Set(dimen=2, initialize=orders_related_to_nodes_tup)

        nodes_for_vessels_tup = [(vessel, node)
                                 for (vessel, node) in
                                 nodes_for_vessels.keys()
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

        self.m.TIME_PERIODS_INCLUDING_DUMMY = pyo.Set(
            initialize=time_periods + [-1])  # The dummy end period is denoted -1

        self.m.PRODUCTION_LINES = pyo.Set(initialize=production_lines)

        self.m.PRODUCTION_LINES_FOR_FACTORIES_TUP = pyo.Set(dimen=2, initialize=production_lines_for_factories)

        print("Done setting sets!")

    def _set_parameters(self,
                        vessel_ton_capacities,
                        vessel_nprod_capacities,
                        vessel_initial_loads,
                        factory_inventory_capacities,
                        factory_initial_inventories,
                        inventory_unit_costs,
                        transport_unit_costs,
                        transport_times,
                        unloading_times,
                        loading_times,
                        demands,
                        production_unit_costs,
                        production_min_capacities,
                        production_max_capacities):

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

        # TODO: Fix in model file - must have nodes here, not order_nodes/factory_nodes (i.e. also for loading_times)
        self.m.unloading_times = pyo.Param(self.m.VESSELS,
                                           self.m.NODES,
                                           initialize=unloading_times)

        self.m.loading_times = pyo.Param(self.m.VESSELS,
                                         self.m.NODES,
                                         initialize=loading_times)

        self.m.demands = pyo.Param(self.m.ORDERS_RELATED_TO_NODES_TUP,
                                   self.m.PRODUCTS,
                                   initialize=demands)

        print("Done setting parameters!")

    def _set_variables(self):
        self.m.x = pyo.Var(self.m.VESSELS,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.NODES_INCLUDING_DUMMIES,
                           self.m.TIME_PERIODS_INCLUDING_DUMMY,
                           domain=pyo.Boolean,
                           initialize=0)

        self.m.w = pyo.Var(self.m.VESSELS,
                           self.m.NODES,
                           self.m.TIME_PERIODS_INCLUDING_DUMMY,
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

        self.m.z = pyo.Var(self.m.VESSELS,
                           self.m.ORDERS_RELATED_TO_NODES_TUP,
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

        self.m.q = pyo.Var(self.m.FACTORY_NODES,
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
                           domain=pyo.NonNegativeReals)

        self.m.e = pyo.Var(self.m.ORDER_NODES,
                           domain=pyo.Boolean,
                           initialize=0)  # To be removed, implemented to avoid infeasibility during testing

        print("Done setting variables!")

    def _set_objective(self):
        def obj(model):
            return (sum(sum(sum(
                model.production_unit_costs[i, p] * model.q[i, p, t] + model.inventory_unit_costs[i] * model.r[i, p, t]
                for t in model.TIME_PERIODS) for p in model.PRODUCTS) for i in model.FACTORY_NODES)
                    + sum(sum(sum(sum(
                        model.transport_unit_costs[v] * model.transport_times[i, j] * model.x[v, i, j, t] for t in
                        model.TIME_PERIODS) for j in model.NODES) for i in model.NODES) for v in model.VESSELS)
                    + sum(1000000000 * model.e[i] for i in model.ORDER_NODES))  # TODO: Remove when done with testing

        self.m.objective = pyo.Objective(rule=obj, sense=pyo.minimize)

        print("Done setting objective")

    def _set_constraints(self):

        def constr_max_one_activity(model, v, t):
            relevant_nodes = {n for (vessel, n) in model.NODES_FOR_VESSELS_TUP if vessel == v}

            return (sum(model.y_minus[v, i, t] for i in relevant_nodes) +
                    sum(model.y_plus[v, i, t] for i in relevant_nodes) +
                    sum(sum(model.x[v, i, j, t] for i in relevant_nodes) for j in relevant_nodes) +
                    sum(model.w[v, i, t] for i in relevant_nodes)
                    <= 1)

        self.m.constr_max_one_activity = pyo.Constraint(self.m.VESSELS,
                                                        self.m.TIME_PERIODS,
                                                        rule=constr_max_one_activity)

        def constr_max_one_vessel_loading(model, i, v, t):
            relevant_vessels = {vessel for (j, vessel) in model.VESSELS_FOR_FACTORY_NODES_TUP
                                if j == i and vessel != v}
            relevant_time_periods = {tau for tau in model.TIME_PERIODS
                                     if (t <= tau <= t + model.loading_times[v, i])}
            return (sum(sum(model.y_plus[u, i, tau] for tau in relevant_time_periods) for u in relevant_vessels)
                    <= 1 - model.y_plus[v, i, t])

        self.m.constr_max_one_vessel_loading = pyo.Constraint(self.m.VESSELS_FOR_FACTORY_NODES_TUP,
                                                              # (factory_node, vessel) tuples
                                                              self.m.TIME_PERIODS,
                                                              rule=constr_max_one_vessel_loading)

        def constr_delivery_within_time_window(model, i):
            relevant_vessels = {vessel for (vessel, j) in model.ORDER_NODES_FOR_VESSELS_TUP
                                if j == i}
            relevant_time_periods = {t for (j, t) in model.TIME_WINDOWS_FOR_ORDERS_TUP
                                     if j == i}
            return (sum(sum(model.y_minus[v, i, t] for v in relevant_vessels) for t in relevant_time_periods) + model.e[
                i]  # TODO: Remove e when done with testing
                    == 1)

        self.m.constr_delivery_within_time_window = pyo.Constraint(self.m.ORDER_NODES,
                                                                   rule=constr_delivery_within_time_window)

        def constr_sailing_after_loading_unloading(model, v, i, t):
            unloading_time = pyo.value(model.unloading_times[v, i])
            loading_time = pyo.value(model.loading_times[v, i])
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                                          if vessel == v}
            relevant_destination_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                                          if vessel == v}
            relevant_destination_nodes.add('d_-1')  # Adds the dummy end node

            if t < unloading_time and t < loading_time:  # Neither unloading_time nor loading_time is valid
                return Constraint.Feasible

            elif t < unloading_time:  # Only loading_time is valid
                return (model.y_plus[v, i, (t - loading_time)] == sum(
                    model.x[v, i, j, t] for j in relevant_destination_nodes))

            elif t < loading_time:  # Only unloading_time is valid
                return (model.y_minus[v, i, (t - unloading_time)] == sum(
                    model.x[v, i, j, t] for j in relevant_destination_nodes))

            else:
                return ((model.y_minus[v, i, (t - unloading_time)] + model.y_plus[v, i, (t - loading_time)])
                        == sum(model.x[v, i, j, t] for j in relevant_destination_nodes))

        self.m.constr_sailing_after_loading_unloading = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                       self.m.TIME_PERIODS_INCLUDING_DUMMY,
                                                                       rule=constr_sailing_after_loading_unloading)

        # TODO: Not in model formulation - make sure it is correct and add?
        def constr_node_balance(model, v, i):
            return (sum(sum(model.x[v, i, j, t] for j in model.NODES_INCLUDING_DUMMY_END) for t in model.TIME_PERIODS_INCLUDING_DUMMY)
                    == sum(sum(model.x[v, j, i, t] for j in model.NODES_INCLUDING_DUMMY_START) for t in model.TIME_PERIODS))

        self.m.constr_node_balance = pyo.Constraint(self.m.VESSELS,
                                                    self.m.NODES,
                                                    rule=constr_node_balance)

        def constr_wait_load_unload_after_sailing(model, v, i, t):
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP if
                              vessel == v and model.transport_times[j, i] <= t}.union({'d_0'})
            return (sum(model.x[v, j, i, (t - model.transport_times[j, i])] for j in relevant_nodes) + model.w[
                v, i, (t - 1)]
                    == model.y_minus[v, i, t] + model.y_plus[v, i, t] + model.w[v, i, t])

        self.m.constr_wait_load_unload_after_sailing = pyo.Constraint(self.m.NODES_FOR_VESSELS_TUP,
                                                                      self.m.TIME_PERIODS,
                                                                      rule=constr_wait_load_unload_after_sailing)

        def constr_no_waiting_in_dummy_end_period(model):
            return sum(sum(model.w[v, i, -1] for v in model.VESSELS) for i in model.NODES) == 0

        self.m.constr_no_waiting_in_dummy_end_period = pyo.Constraint(rule=constr_no_waiting_in_dummy_end_period)

        def constr_start_route_once(model, v):
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                              if vessel == v}
            relevant_time_periods = {t for (vessel, t) in model.TIME_PERIODS_FOR_VESSELS_TUP
                                     if vessel == v}
            return (sum(sum(model.x[v, 'd_0', j, t] for t in relevant_time_periods) for j in relevant_nodes)
                    == 1)
        self.m.constr_start_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_start_route_once)

        # TODO: Add to formulation, if OK
        def constr_start_route_first(model, v, t):
            relevant_nodes = {j for (vessel, j) in model.NODES_FOR_VESSELS_TUP
                              if vessel == v}
            relevant_nodes_including_dummy_end = relevant_nodes.union({'d_-1'})
            relevant_time_periods = {tau
                                     for (vessel, tau) in model.TIME_PERIODS_FOR_VESSELS_TUP
                                     if vessel == v and tau < t}  # .union({-1})

            return (1000000000000000 * (1 - sum(model.x[v, 'd_0', j, t] for j in relevant_nodes_including_dummy_end))
                    >= sum(sum(sum(model.x[v, i, j, tau] for j in relevant_nodes_including_dummy_end) + model.y_plus[
                        v, i, tau] + model.y_minus[v, i, tau] + model.w[v, i, tau] for i in relevant_nodes) for tau in
                           relevant_time_periods))  # TODO: Fix this pretty bad big M thing

        self.m.constr_start_route_first = pyo.Constraint(self.m.VESSELS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_start_route_first)

        def constr_end_route_once(model, v):
            relevant_nodes = {i for (vessel, i) in model.NODES_FOR_VESSELS_TUP
                              if vessel == v}
            return (sum(
                sum(model.x[v, i, 'd_-1', t] for t in model.TIME_PERIODS_INCLUDING_DUMMY) for i in relevant_nodes)
                    == 1)

        self.m.constr_end_route_once = pyo.Constraint(self.m.VESSELS, rule=constr_end_route_once)

        # TODO: Add to formulation, if OK
        def constr_end_route_last(model, v, t):
            if t == -1:
                return Constraint.Feasible  # If route is ended in the last period, there are no restrictions upon activity in the previous time periods
            relevant_nodes = {i
                              for (vessel, i) in model.NODES_FOR_VESSELS_TUP if vessel == v}
            relevant_nodes_including_dummy_start = relevant_nodes.union({'d_0'})
            relevant_time_periods = {tau
                                     for (vessel, tau) in model.TIME_PERIODS_FOR_VESSELS_TUP
                                     if vessel == v and tau >= t}  # .union({-1})

            return (100000000000000000 * (  # TODO: Fix this pretty bad big M thing
                    1 - sum(model.x[v, i, 'd_-1', t] for i in relevant_nodes_including_dummy_start))
                    >=
                    sum(sum(sum(model.x[v, i, j, tau] + model.y_plus[v, j, tau] + model.y_minus[v, j, tau] +
                                model.w[v, j, tau]
                                for tau in relevant_time_periods)
                            for i in relevant_nodes_including_dummy_start)
                        for j in relevant_nodes))

        self.m.constr_end_route_last = pyo.Constraint(self.m.VESSELS,
                                                      self.m.TIME_PERIODS_INCLUDING_DUMMY,
                                                      rule=constr_end_route_last)

        def constr_pickup_requires_factory_visit(model, v, i, j, t):
            return model.z[v, i, j, t] <= model.y_plus[v, i, t]

        self.m.constr_pickup_requires_factory_visit = pyo.Constraint(
            self.m.ORDER_NODES_FOR_FACTORY_NODES_FOR_VESSELS_TRIP,  # (v, f, o)
            self.m.TIME_PERIODS,
            rule=constr_pickup_requires_factory_visit)

        def constr_delivery_requires_order_visit(model, v, i, t):
            return model.z[v, i, i, t] == model.y_minus[v, i, t]

        self.m.constr_delivery_requires_order_visit = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,
                                                                     self.m.TIME_PERIODS,
                                                                     rule=constr_delivery_requires_order_visit)

        # TODO: This constraint is not in the model - make sure it is correct and add it
        # TODO: NB! Assuming here that a vessel cannot use existing load (e.g. initial load) to serve an order, which is maybe not so good...
        # TODO: Error in this constraint: pickup factory is not linked to unloading! May be mistakes here!
        # TODO: In conclusion: look at this one...
        def constr_delivery_requires_pickup(model, v, i, t):
            relevant_factories = {f for (vessel, f) in model.FACTORY_NODES_FOR_VESSELS_TUP if vessel == v}.intersection({
                n for (n, o) in model.ORDERS_RELATED_TO_NODES_TUP if i == o})
            return (sum(sum(model.z[v, j, i, tau] for j in relevant_factories) for tau in self.m.TIME_PERIODS if tau < t)
                    >=
                    model.y_minus[v, i, t])

        #self.m.constr_delivery_requires_pickup = pyo.Constraint(self.m.ORDER_NODES_FOR_VESSELS_TUP,  # (v, o)
        #                                                        self.m.TIME_PERIODS,
        #                                                        rule=constr_delivery_requires_pickup)

        def constr_vessel_initial_load(model, v, p):
            return model.l[v, p, 0] == model.vessel_initial_loads[v, p]

        self.m.constr_vessel_initial_load = pyo.Constraint(self.m.VESSELS,
                                                           self.m.PRODUCTS,
                                                           rule=constr_vessel_initial_load)

        def constr_load_balance(model, v, p, t):
            if t == 0:  # Initial load handled by above constraint
                return Constraint.Feasible
            relevant_order_nodes = {j for (vessel, j) in model.ORDER_NODES_FOR_VESSELS_TUP
                                    if vessel == v}
            return (model.l[v, p, t] == model.l[v, p, (t - 1)] - sum(
                sum(model.demands[i, j, p] * model.z[v, i, j, t] for i in
                    {n for (n, o) in model.ORDERS_RELATED_TO_NODES_TUP if o == j}) for j in relevant_order_nodes))

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

        def constr_initial_inventory(model, i, p):
            return model.r[i, p, 0] == model.factory_initial_inventories[i, p]

        self.m.constr_initial_inventory = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         rule=constr_initial_inventory)

        def constr_inventory_balance(model, i, p, t):
            if t == 0:
                return Constraint.Feasible
            relevant_order_nodes = {j for (f, j) in model.ORDER_NODES_FOR_FACTORIES_TUP
                                    if f == i}
            return model.r[i, p, t] == model.r[i, p, (t - 1)] + model.q[i, p, (t - 1)] + sum(
                model.demands[i, j, p] * sum(model.z[v, i, j, t] for v in model.VESSELS) for j in relevant_order_nodes)

        self.m.constr_inventory_balance = pyo.Constraint(self.m.FACTORY_NODES,
                                                         self.m.PRODUCTS,
                                                         self.m.TIME_PERIODS,
                                                         rule=constr_inventory_balance)

        def constr_production_below_max_capacity(model, i, p, t):
            relevant_production_lines = {l for (factory, l) in model.PRODUCTION_LINES_FOR_FACTORIES_TUP if factory == i}
            return model.q[i, p, t] <= sum(
                (model.production_max_capacities[ll, p] * model.g[ll, p, t]) for ll in relevant_production_lines)

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

        print("Done setting constraints!")

    def solve(self):
        print("Solver running...")
        t = time()
        self.results = self.solver_factory.solve(self.m, tee=True)
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("Termination condition not optimal, ", self.results.solver.termination_condition)
        print("Solve time: ", round(time() - t, 1))

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
            #print_factory_production()
            #print_factory_inventory()
            #print_vessel_routing()
            #print_order_delivery()
            #print_order_pickup()
            #print_factory_pickup()
            #print_waiting()
            #print_vessel_load()
            print_orders_not_delivered()

        def print_result_eventwise():

            def print_routing():
                for v in self.m.VESSELS:
                    print("ROUTING OF VESSEL", v)
                    for t in self.m.TIME_PERIODS:
                        # x variable
                        for i in self.m.NODES_INCLUDING_DUMMIES:
                            for j in self.m.NODES_INCLUDING_DUMMIES:
                                if pyo.value(self.m.x[v, i, j, t]) >= 0.5:
                                    print(t, ": ", i, " --> ", j, sep="")
                        # w variable
                        for i in self.m.NODES:
                            if pyo.value(self.m.w[v, i, t]) >= 0.5:
                                print(t, ": waits to go to ", i, sep="")
                        for i in [j for (vessel, j) in self.m.NODES_FOR_VESSELS_TUP if vessel == v]:
                            # y_plus variable
                            if pyo.value(self.m.y_plus[v, i, t]) >= 0.5:
                                print(t, ": loads in node ", i, sep="")
                            # y_minus variable
                            if pyo.value(self.m.y_minus[v, i, t]) >= 0.5:
                                print(t, ": unloads in node ", i, sep="")
                        # z variable
                        for (n, o) in self.m.ORDERS_RELATED_TO_NODES_TUP:
                            if pyo.value(self.m.z[v, n, o, t]) >= 0.5:
                                print("   [handles order ", o, " in node ", n, "]", sep="")
                    # x variable is defined also for the dummy end node, d_-1
                    for i in self.m.NODES_INCLUDING_DUMMIES:
                        for j in self.m.NODES_INCLUDING_DUMMIES:
                            if pyo.value(self.m.x[v, i, j, -1]) >= 0.5:
                                print("-1: ", i, " --> ", j, sep="")
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
                        for p in self.m.PRODUCTS:
                            if pyo.value(self.m.q[i, p, t]) >= 0.5:
                                print(t, ": production of ", round(pyo.value(self.m.q[i, p, t]), 1),
                                      " tons of product ",
                                      p, sep="")
                            if pyo.value(self.m.r[i, p, t]) >= 0.5:
                                print(t, ": inventory level is ", round(pyo.value(self.m.r[i, p, t]), 1),
                                      " tons of product ", p, sep="")
                            relevant_order_nodes = {j for (f, j) in self.m.ORDER_NODES_FOR_FACTORIES_TUP if f == i}
                            loaded_onto_vessels = pyo.value(sum(
                                self.m.demands[i, j, p] * sum(self.m.z[v, i, j, t] for v in self.m.VESSELS) for j in
                                relevant_order_nodes))
                            if loaded_onto_vessels >= 0.5:
                                print(t, ": ", round(loaded_onto_vessels, 1), " tons of product ", p,
                                      " is loaded onto vessels ", sep="")
                    print()

            print_routing()
            print_vessel_load()
            print_production_and_inventory()

        print_result_variablewise()
        print_result_eventwise()

from typing import List, Dict, Tuple
from src.read_problem_data import ProblemData
import math
import numpy as np


class Node:
    id: str
    tw_start: int = -1
    tw_end: int = math.inf
    demand: List[int] = []
    zone: str = "na"

    def __init__(self, name, is_factory=False):
        self.id: str = name
        self.is_factory: bool = is_factory

    def __repr__(self):
        return f"{self.id}: ({self.tw_start}, {self.tw_end}), {self.demand}, (zone {self.zone})"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class InternalProblemData(ProblemData):

    def __init__(self, file_path: str, precedence: bool = False) -> None:
        super().__init__(file_path)
        self.precedence = precedence
        self._init_nodes()

    def _init_nodes(self) -> None:
        f_nodes = self.factory_nodes[:]
        o_nodes = self.order_nodes[:]
        self.factory_nodes: Dict[str, Node] = {}
        self.order_nodes: Dict[str, Node] = {}
        for i in f_nodes:
            node = Node(i, is_factory=True)
            node.tw_start = 1
            node.tw_end = len(self.time_periods) - 1
            self.factory_nodes[i] = node
        for i in o_nodes:
            node = Node(i)
            node.tw_start = self.tw_start[i]
            node.tw_end = self.tw_end[i]
            node.demand = [self.demands[i, i, p] for p in self.products]
            self.order_nodes[i] = node
        if self.precedence:
            for zone in self.orders_for_zones.keys():
                for order_node in self.orders_for_zones[zone]:
                    self.order_nodes[order_node].zone = zone
        self.nodes: Dict[str, Node] = {**self.factory_nodes, **self.order_nodes}


class Solution:

    def __init__(self, prbl: InternalProblemData, debug: bool = False) -> None:
        self.prbl = prbl
        self.debug = debug
        self.factory_visits: Dict[str, List[str]] = self.init_factory_visits()
        self.factory_visits_route_index: Dict[str, List[int]] = {f: [0 for _ in self.factory_visits[f]]
                                                                 for f in self.prbl.factory_nodes}
        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.e: Dict[str, List[int]] = {v: [max(1, self.prbl.start_times_for_vessels[v])] for v in self.prbl.vessels}
        self.l: Dict[str, List[int]] = {v: [math.inf] for v in self.prbl.vessels}
        self.e_factory: Dict[str, List[int]] = {v: [self.prbl.start_times_for_vessels[v]] for v in self.prbl.vessels}
        self.l_factory: Dict[str, List[int]] = {v: [math.inf] for v in self.prbl.vessels}
        # TODO: Update e and l due to factory visits

    def insert_node(self, node_id: str, vessel: str, idx: int) -> None:
        insert_node = self.prbl.nodes[node_id]
        route = self.routes[vessel]
        idx = len(route) + idx + 1 if idx < 0 else idx  # transform negative indexes
        earliest = self.get_earliest(idx, vessel, node_id)
        latest = self.get_latest(idx, vessel, node_id)
        route.insert(idx, node_id)
        self.e[vessel].insert(idx, earliest)
        self.l[vessel].insert(idx, latest)
        self.propagate_e_and_l_from_insertion(idx, vessel)

        # increase the index of all succeeding factory nodes in factory_visits_route_index
        for factory in self.prbl.factory_nodes:
            for i, v in enumerate(self.factory_visits[factory]):
                if v == vessel and idx <= self.factory_visits_route_index[factory][i]:
                    self.factory_visits_route_index[factory][i] += 1

        # Update factory related stuff
        if insert_node.is_factory:
            # TODO: Arbitrary insertion (not append)
            self.factory_visits[node_id].append(vessel)
            self.factory_visits_route_index[node_id].append(idx)

        # print()
        # print(node_id)
        # print(route)
        # print("e and l:", list(zip(self.e[vessel], self.l[vessel])))
        # print(self.factory_visits)
        # print(self.factory_visits_route_index)
        # print()

    def check_insertion_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        # [x] check that the vessel load capacity is not violated
        # [x] check that the vessel's max n.o. products is not violated
        # [x] check that the time windows of visited nodes are not violated
        # [x] check that the route does, or has time to, end the route in a factory (with destination capacity)
        # [x] check that precedence/wait constraints are not violated
        # [ ] check max #vessels simultaneously at factory
        # [x] check factory destination max #vessels
        # [x] check production capacity (PPFC)
        return all([self.check_load_feasibility(insert_node, vessel, idx),
                    self.check_no_products_feasibility(insert_node, vessel, idx),
                    self.check_time_feasibility(insert_node.id, vessel, idx),
                    self.check_final_factory_destination_feasibility(insert_node.id, vessel, idx),
                    self.check_production_feasibility(insert_node_id=insert_node.id, vessel=vessel, idx=idx)])

    def propagate_e_and_l_from_insertion(self, idx: int, vessel: str) -> None:
        # Update latest time for preceding visits until no change
        for i in range(idx - 1, -1, -1):
            latest = self.get_latest(i, vessel)
            if self.l[vessel][i] > latest:
                self.l[vessel][i] = latest
            else:
                break

        # Update earliest time for succeeding visits until no change
        for i in range(idx + 1, len(self.routes[vessel])):
            earliest = self.get_earliest(i, vessel)
            if self.e[vessel][i] < earliest:
                self.e[vessel][i] = earliest
            else:
                break

    def get_earliest(self, idx: int, vessel: str, node_id: str = None, prev_node_id: str = None,
                     prev_e: int = None) -> int:
        # if node_id is given: Calculate for insertion of node_id at idx. Else, calculate for the current node at idx
        # use prev_node_id and prev_e argument if it is given
        route = self.routes[vessel]
        node = self.prbl.nodes[node_id] if node_id else self.prbl.nodes[route[idx]]

        # if node_id:
        #     print("Checking insertion of", node.id, "at index", idx, "for vessel", vessel)
        # else:
        #     print("Checking already inserted node", node.id, "at index", idx, "for vessel", vessel)

        if not prev_node_id and not prev_e:
            prev_node_id = route[idx - 1] if idx > 0 else None
            prev_e = self.e[vessel][idx - 1] if prev_node_id else self.prbl.start_times_for_vessels[vessel]

        prev_transport_time = self.prbl.transport_times[prev_node_id, node.id] if prev_node_id else 0
        prev_loading_unloading_time = self.prbl.loading_unloading_times[vessel, prev_node_id] if prev_node_id else 0
        earliest = max(node.tw_start, prev_e + prev_loading_unloading_time + prev_transport_time)

        # If the precedence extension is included, earliest visiting time must also incorporate minimum waiting time
        if self.prbl.precedence and prev_node_id and (
                (self.prbl.nodes[prev_node_id].zone, self.prbl.nodes[node.id].zone) in
                [("red", "green"), ("red", "yellow"), ("yellow", "green")]):
            earliest = max(earliest, prev_e + prev_loading_unloading_time + self.prbl.min_wait_if_sick_abs)
        return earliest

    def get_latest(self, idx: int, vessel: str, node_id: str = None, next_node_id: str = None,
                   next_l: int = None) -> int:
        # if node_id is given: Calculate for insertion of node_id at idx. Else, calculate for the current node at idx
        # use next_node_id and next_l argument if it is given
        route = self.routes[vessel]
        node = self.prbl.nodes[node_id] if node_id else self.prbl.nodes[route[idx]]
        if not next_node_id and not next_l:
            next_node_id = route[idx + 1] if idx + 1 < len(route) else None
            next_l = self.l[vessel][idx + 1] if next_node_id else len(self.prbl.time_periods) - 1

        next_transport_time = self.prbl.transport_times[node.id, next_node_id] if next_node_id else 0

        if idx == len(route) and node.is_factory:
            loading_unloading_time = 0  # zero loading time for last visited factory
        else:
            loading_unloading_time = self.prbl.loading_unloading_times[vessel, node.id]

        latest = min(node.tw_end, next_l - next_transport_time - loading_unloading_time)

        # if the last node is not a factory, make sure there is enough time to reach a factory destination
        if (idx == len(route) and node_id or idx == len(route) - 1) and not node.is_factory:
            time_to_factory = min(
                self.prbl.transport_times[node.id, f] for f in
                self.prbl.factory_nodes)
            latest = min(latest, len(self.prbl.time_periods) - time_to_factory - loading_unloading_time - 1)

        # If the precedence extension is included, latest visiting time must also incorporate minimum waiting time
        if self.prbl.precedence and next_node_id is not None and (
                (node.zone, self.prbl.nodes[next_node_id].zone) in
                [("red", "green"), ("red", "yellow"), ("yellow", "green")]):
            latest = min(latest, next_l - loading_unloading_time - self.prbl.min_wait_if_sick_abs)
        return latest

    # Perhaps depreciated when temp variable introduced
    # def get_latest_backward(self, vessel: str, insert_node_idx: int, insert_node_id: str) -> int:  # , next_node_id: str, next_l: int) -> bool:
    #     """Iteratively checks latest time for preceding visits until start factory reached"""
    #     route = self.routes[vessel]
    #     latest = self.get_latest(idx=insert_node_idx, vessel=vessel, node_id=insert_node_id)
    #     node = insert_node_id
    #     for i in range(insert_node_idx - 1, -1, -1):
    #         latest = self.get_latest(idx=i, vessel=vessel, next_node_id=node, next_l=latest)
    #         if self.l[vessel][i] == latest:  # propagation of check stops if l is unchanged
    #             return self.l[vessel][0]  # go directly to value we are looking for
    #         node = route[i]
    #     return latest

    def check_final_factory_destination_feasibility(self, insert_node_id: str = None, vessel: str = None,
                                                    idx: int = None) -> bool:
        # insert_node_id, vessel and idx given: check feasibility for terminating this vessel's route in given factory
        if insert_node_id and vessel and idx:
            if insert_node_id not in self.prbl.factory_nodes.keys():  # no changes to destination factory
                return True
            if ((len([v for v in self.prbl.vessels if self.routes[v][-1] == insert_node_id]) +
                 (1 if insert_node_id != (self.routes[vessel][-1]) else 0))  # plus 1 if dest factory is changed
                    > self.prbl.factory_max_vessels_destination[insert_node_id]):
                return False

        # else: check feasibility for all current route destinations
        else:
            for f_id in self.prbl.factory_nodes.keys():
                if (len([v for v in self.prbl.vessels if self.routes[v][-1] == f_id])
                        > self.prbl.factory_max_vessels_destination[f_id]):
                    return False
        return True

    def check_time_feasibility(self, insert_node_id: str, vessel: str, idx: int) -> bool:
        route = self.routes[vessel]
        idx = len(route) + idx + 1 if idx < 0 else idx  # transform negative indexes
        e = self.get_earliest(idx, vessel, insert_node_id)
        l = self.get_latest(idx, vessel, insert_node_id)
        if l < e:  # not enough space for insertion
            return False

        # check latest time for preceding visits until no change
        next_node_id = insert_node_id
        next_l = l
        for i in range(idx - 1, -1, -1):
            l = self.get_latest(i, vessel, next_node_id=next_node_id, next_l=next_l)
            if l < self.e[vessel][i]:  # insertion squeezes out preceding order
                return False
            if self.l[vessel][i] > l:  # propagation of check continues if l is higher than before
                next_node_id = route[i]
                next_l = l
            else:
                break

        # check earliest time for succeeding visits until no change
        prev_node_id = insert_node_id
        prev_e = e
        for i in range(idx, len(route)):
            e = self.get_earliest(i, vessel, prev_node_id=prev_node_id, prev_e=prev_e)
            if self.l[vessel][i] < e:  # insertion squeezes out succeeding order
                return False
            if self.e[vessel][i] < e:  # propagation of check continues if e is lower than before
                prev_node_id = route[i]
                prev_e = e
            else:
                break
        return True

    def check_factory_destination(self, insert_node_id: str, vessel: str, idx: int) -> bool:
        # TODO - but is this the same as check_final_factory_destination_feasibility?
        pass

    def update_factory_visit_times(self, factory: str):
        # TODO
        # update e
        visits = self.factory_visits[factory]
        quay_capacity = self.prbl.factory_max_vessels_loading[factory]
        t = 0
        self.get_overlapping_visits(factory)
        for vessel in visits:
            pass
            # t =
            # sim_count += 1

    def get_overlapping_visits(self, factory: str):
        # TODO
        pass
        # visits = self.factory_visits[factory]
        # n_visits = {vessel: 0}
        # overlap_clusters = []
        # overlap_times = []

    # def check_time_feasibility2(self, insert_node: Node, vessel: str, idx: int) -> bool:
    #     route = self.routes[vessel]
    #     prev_node_id = route[idx - 1]
    #     next_node_id = route[idx + 1]
    #     time_between = (self.prbl.loading_unloading_times[prev_node_id]
    #                     + self.prbl.transport_times[prev_node_id, insert_node.id]
    #                     + self.prbl.loading_unloading_times[insert_node.id]
    #                     + self.prbl.transport_times[insert_node.id, next_node_id])
    #     return self.e[vessel][idx - 1] + time_between - 1 <= self.l[vessel][idx + 1]
    #     # Lianes and Noreng is strange here

    def check_load_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        route = self.routes[vessel]
        voyage_start, voyage_end = self.get_voyage_start_end_idx(vessel, idx)
        voyage_demand = sum(d for node_id in route[voyage_start:voyage_end] for d in self.prbl.nodes[node_id].demand)
        return voyage_demand + sum(d for d in insert_node.demand) <= self.prbl.vessel_ton_capacities[vessel]

    def check_no_products_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        if self.prbl.vessel_nprod_capacities[vessel] >= len(self.prbl.products):
            return True
        route = self.routes[vessel]
        voyage_start, voyage_end = self.get_voyage_start_end_idx(vessel, idx)
        voyage_demanded_products = [any(self.prbl.nodes[node_id].demand[p]
                                        for node_id in route[voyage_start + 1:voyage_end])
                                    for p in range(len(self.prbl.products))]
        insert_node_demanded_products = [bool(d) for d in insert_node.demand]
        combined_demanded_products = np.logical_or(voyage_demanded_products, insert_node_demanded_products)
        return sum(combined_demanded_products) <= self.prbl.vessel_nprod_capacities[vessel]

    def get_insertion_utility(self, node: Node, vessel: str, idx: int) -> float:  # High utility -> good insertion
        net_sail_change = (self.prbl.transport_times[self.routes[vessel][idx - 1],
                                                     node.id])
        if idx < len(self.routes[vessel]) - 1:  # node to be inserted is not at end of route
            net_sail_change += (self.prbl.transport_times[node.id,
                                                          self.routes[vessel][
                                                              idx]]  # cost to next node, currently at idx
                                - self.prbl.transport_times[self.routes[vessel][idx - 1],
                                                            self.routes[vessel][idx]])
        delivery_gain = self.prbl.external_delivery_penalties[node.id] if not node.is_factory else 0

        return delivery_gain - net_sail_change * self.prbl.transport_unit_costs[vessel]

    def get_voyage_start_idxs_for_factory(self, factory_node_id: str) -> Dict[str, List[Tuple[int, int]]]:
        # Returns: {vessel: (factory visit index in route, latest loading start time)} for the factory given as input
        # if vessel visits input factory for loading

        voyage_start_idxs_for_vessels: Dict[str, List[Tuple[int, int]]] = {}
        for v in self.prbl.vessels:
            voyage_start_idxs_for_vessels[v] = []
            route = self.routes[v]

            # Adding index and latest loading time for vessels loading at input factory
            for i in range(len(route)):  # note that the last node may also be included here
                if self.prbl.nodes[route[i]].is_factory and self.prbl.nodes[route[i]].id == factory_node_id:
                    voyage_start_idxs_for_vessels[v].append(
                        tuple((i, self.l[v][i])))  # TODO: Should be l_temp here

            # Vessel does not load at input factory -> vessel is skipped
            if len(voyage_start_idxs_for_vessels[v]) == 0:
                voyage_start_idxs_for_vessels.pop(v, None)
        return voyage_start_idxs_for_vessels

    def get_vessel_factories(self, vessel: str) -> List[str]:
        return [node_id for node_id in self.routes[vessel] if self.prbl.nodes[node_id].is_factory]

    def check_production_feasibility(self, insert_node_id: str = None, vessel: str = None, idx: int = None) -> bool:
        # If checking whether an insertion gives a feasible solution: give parameters insertion_node_id, vessel, idx
        # If validating an existing solution: skip parameters
        if idx < 0 and vessel is not None and insert_node_id is not None:
            idx = len(self.routes[vessel]) + idx + 1

        # Finding which factories to check
        if insert_node_id is None:  # check all factories
            factories_to_check = self.prbl.factory_nodes.keys()  # validate solution
        else:
            factories_to_check = []
            for f in self.prbl.factory_nodes.keys():
                if self.get_voyage_start_factory(vessel=vessel, idx=idx) == f:
                    factories_to_check.append(f)
                elif self.is_factory_latest_changed_in_temp(f):
                    factories_to_check.append(f)

        # Feasibility is checked for relevant factories
        for factory_node_id in factories_to_check:
            # voyage_start_idxs on the form {vessel: (index in route, temp latest factory loading time)}
            # for vessels visiting factory with id factory_node_id
            voyage_start_idxs: Dict[str, List[Tuple[int, int]]] = self.get_voyage_start_idxs_for_factory(
                factory_node_id)

            # If we insert a _factory_ in the middle of a route we must add it to the list voyage_start_idx
            if ((insert_node_id and vessel and idx)
                    and insert_node_id == factory_node_id
                    and idx < len(self.routes[vessel])):  # Factory visited to load orders
                # Vessel exists in dict, as it already has another visit to the factory
                if vessel in voyage_start_idxs.keys():
                    voyage_start_idxs[vessel].append((idx, self.get_latest(idx, vessel, insert_node_id)))  # TODO: get latest temp
                    sorted(voyage_start_idxs[vessel], key=lambda x: x[0])  # sort according to first elem, visit index
                # The inserted visit is the vessel's only visit to the relevant factory
                else:
                    voyage_start_idxs[vessel] = [(idx, self.get_latest(idx, vessel, insert_node_id))]
            # If we insert an _order_ in the route, we find the start index of the relevant voyage
            insertion_voyage_start_idx = None
            if (insert_node_id is not None
                    and not self.prbl.nodes[insert_node_id].is_factory
                    and self.get_voyage_start_factory(vessel, idx) == factory_node_id):
                insertion_voyage_start_idx = self.get_voyage_start_end_idx(vessel, idx)[0]

            # Collect quantity to be delivered for each voyage
            products_for_voyage: List[List[int]] = []
            latest_loading_times: List[int] = []

            for v in voyage_start_idxs.keys():
                for i in range(len(voyage_start_idxs[v])):  # for each voyage whose loading factory is investigated
                    stop_index = self.get_voyage_end_idx(vessel=v, start_idx=voyage_start_idxs[v][i][0])
                    add_rows = [self.prbl.nodes[node_id].demand
                                for node_id in self.routes[v][voyage_start_idxs[v][i][0] + 1:stop_index]]
                    if i == insertion_voyage_start_idx and vessel == v:
                        add_rows.append(self.prbl.nodes[insert_node_id].demand)

                    # Only add the row if demand was found for the route
                    # Matrix products_for_voyage - rows: voyages (index i), columns: products
                    # List latest_loading_times - latest loading time (index i)
                    if len(add_rows) > 0:
                        prod_sums = []
                        for p in range(len(self.prbl.products)):
                            prod_sums.append(sum(row[p] for row in add_rows))
                        products_for_voyage.append(prod_sums)
                        latest_loading_times.append(voyage_start_idxs[v][i][1]
                                                    if voyage_start_idxs[v][i][1] is not math.inf
                                                    else max(self.prbl.time_periods))
                        # TODO: can probably change to: latest_loadting_times.append(voyage_start_idxs[v][i][1])
                        # TODO: when this value refers to the temp value

            # Order rows in matrix products_for_voyage by corresponding latest_loading_times asc
            products_for_voyage = [sublist for _, sublist in sorted(zip(latest_loading_times, products_for_voyage),
                                                                    key=lambda pair: pair[0])]
            latest_loading_times.sort()

            # Sum rows whose latest delivery time is equal (these are treated as one production order in the PP)
            def indices_with_same_value(lis, value):
                return [j for j, x in enumerate(lis) if x == value]
            same_delivery_times = [indices_with_same_value(latest_loading_times, value)
                                   for value in set(latest_loading_times)]

            # Looping backwards in order not to mess up the indices when popping
            for i in range(len(same_delivery_times) - 1, -1, -1):
                if len(same_delivery_times[i]) > 1:  # there are rows whose loading time is equal
                    products_for_voyage[i] = [sum(row[p]
                                                  for row in products_for_voyage[
                                                             same_delivery_times[i][0]:same_delivery_times[i][-1]])
                                              for p in range(len(self.prbl.products))]
                    for j in range(1, len(same_delivery_times[i])):
                        products_for_voyage.pop(j)
                        latest_loading_times.pop(j)
            # prod_sums is now a list of products for orders summed for voyages with same latest starting time

            # Make cumulative representation
            production_requirement_cum = np.cumsum(products_for_voyage, axis=0)

            # Find the minimum number of activities that must be undertaken before a given loading event
            activity_requirement_cum = np.copy(production_requirement_cum)
            production_lines = [l for (i, l) in
                                filter(lambda x: x[0] == factory_node_id, self.prbl.production_lines_for_factories)]

            for p in range(len(self.prbl.products)):  # for all columns in the matrix
                initial_inventory = self.prbl.factory_initial_inventories[(factory_node_id, self.prbl.products[p])]
                production_capacity_max = max(
                    [self.prbl.production_max_capacities[l, self.prbl.products[p]] for l in production_lines])
                for k in range(np.shape(production_requirement_cum)[0]):
                    activity_requirement_cum[k][p] = max(
                        [0,
                         np.ceil((production_requirement_cum[k, p] - initial_inventory) / production_capacity_max)])

            for k in range(len(activity_requirement_cum) - 1, 0, -1):
                production_time_periods = len(production_lines) * sum(
                    [self.prbl.production_stops[factory_node_id, t]
                     for t in range(latest_loading_times[k - 1],
                                    latest_loading_times[k])])
                for i in range(max(np.sum(activity_requirement_cum[k], axis=0) -
                                   np.sum(activity_requirement_cum[k - 1], axis=0) -
                                   production_time_periods,
                                   0)):  # number of activities in this interval exceeding production_time_periods
                    for p in range(np.shape(production_requirement_cum)[1]):
                        if activity_requirement_cum[k][p] > 0:
                            activity_requirement_cum[k - 1][
                                p] += 1  # pushing production activities to occur earlier
                            break

            # If pushing production activities earlier results in too much production taking place
            # before the first loading, then the production schedule is infeasible
            if len(latest_loading_times) > 0:  # if there are orders to be delivered at all
                latest = latest_loading_times[0]
                first_production_time_periods = len(production_lines) * sum(
                    [self.prbl.production_stops[factory_node_id, t]
                     for t in range(latest + 1)])
                if first_production_time_periods < np.sum(activity_requirement_cum[0], axis=None):
                    return False

            # Checking for inventory feasibility
            for k in range(np.shape(activity_requirement_cum)[0]):  # for all rows in the array
                production_capacity_min = min(
                    [self.prbl.production_min_capacities[l, p] for l in production_lines for p in
                     self.prbl.products])
                inventory = (np.sum(activity_requirement_cum[k], axis=0) * production_capacity_min +
                             np.sum(
                                 [self.prbl.factory_initial_inventories[factory_node_id, p] for p in
                                  self.prbl.products]))
                if k > 0:  # subtract previous loadings
                    inventory = inventory - np.sum(products_for_voyage[:k])
                if inventory > self.prbl.factory_inventory_capacities[factory_node_id]:
                    return False
        return True

    def is_factory_latest_changed_in_temp(self, factory_node_id: str) -> bool:
        for vessel in self.prbl.vessels:
            route = self.routes[vessel]
            for i in range(len(route)):
                if route[i] == factory_node_id and self.l[vessel][i] != self.l[vessel][i]:  # TODO: replace with != self.temp_l[vessel][i]
                    return True
        return True  # TODO: Change to False when above replacement is done


    def get_voyage_start_end_idx(self, vessel: str, idx: int) -> Tuple[int, int]:
        route = self.routes[vessel]
        voyage_start_idx = -1
        voyage_end_idx = math.inf
        for i in range(idx - 1, -1, -1):
            if self.prbl.nodes[route[i]].is_factory:
                voyage_start_idx = i
                break
        for i in range(idx, len(route)):
            if self.prbl.nodes[route[i]].is_factory:
                voyage_end_idx = i
                break
        # if no destination factory yet, end voyage at the end of route:
        voyage_end_idx = min(voyage_end_idx, len(route))
        assert voyage_start_idx != -1, "Illegal voyage, no initial factory"
        return voyage_start_idx, voyage_end_idx

    def get_voyage_end_idx(self, vessel: str, start_idx: int) -> int:
        route = self.routes[vessel]
        for i in range(start_idx + 1, len(route)):
            if self.prbl.nodes[route[i]].is_factory:
                return i
        return len(route)

    def get_voyage_start_factory(self, vessel: str, idx: int) -> str:
        route = self.routes[vessel]
        for i in range(idx - 1, 0, -1):
            if self.prbl.nodes[route[i]].is_factory:
                return self.routes[vessel][i]
        return self.routes[vessel][0]

    def init_factory_visits(self) -> Dict[str, List[str]]:
        starting_times = list(self.prbl.start_times_for_vessels.items())
        starting_times.sort(key=lambda item: item[1])
        factory_visits: Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}
        for vessel, _ in starting_times:
            start_location = self.prbl.vessel_initial_locations[vessel]
            factory_visits[start_location].append(vessel)
        return factory_visits

    # DEPRECIATED
    # def get_nth_visit_idx(self, n: int, vessel: str, node_id: str) -> int:
    #     m = 0
    #     for i, node_id2 in enumerate(self.routes[vessel]):
    #         if node_id == node_id2:
    #             m += 1
    #             if n == m:
    #                 return i

    # DEPRECIATED
    # def update_factory_visit_index(self):
    #     n_visits = {(factory, vessel): 0 for vessel in self.prbl.vessels for factory in self.prbl.factory_nodes}
    #     for factory in self.prbl.factory_nodes:
    #         for i, vessel in enumerate(self.factory_visits[factory]):
    #             n_visits[(factory, vessel)] += 1
    #             route_idx = self.get_nth_visit_idx(n_visits[(factory, vessel)], vessel, factory)
    #             self.factory_visits_route_index[factory][i] = route_idx

    def check_insertion_feasibility_insert_and_print(self, insert_node_id: str, vessel: str, idx: int):
        print(
            f"{bcolors.BOLD}Checking for insertion of node {insert_node_id} at index {idx} at route of vessel {vessel}{bcolors.ENDC}")
        print("Utility of insertion:", self.get_insertion_utility(self.prbl.nodes[insert_node_id], vessel, idx))
        feasibility = self.check_insertion_feasibility(self.prbl.nodes[insert_node_id], vessel, idx)
        if not feasibility and self.debug:
            print(f"{bcolors.FAIL}Infeasible insertion - node not inserted{bcolors.ENDC}")
            print("> Final factory destination feasibility:",
                  self.check_final_factory_destination_feasibility(insert_node_id, vessel, idx))
            print("> Production feasibility:",
                  sol.check_production_feasibility(insert_node_id=insert_node_id, vessel=vessel, idx=idx))
            print("> Load feasibility:", self.check_load_feasibility(self.prbl.nodes[insert_node_id], vessel, idx))
            print("> Number of products feasibility:", self.check_no_products_feasibility(insert_node_id, vessel, idx))
            print("> Time feasibility:", self.check_time_feasibility(insert_node_id, vessel, idx))
        else:
            print(f"{bcolors.OKGREEN}Insertion is feasible - inserting node{bcolors.ENDC}")
            self.insert_node(insert_node_id, vessel, idx)
            print(vessel, self.routes[vessel])
            print(list(zip(self.e[vessel], self.l[vessel])))
        print()
        return feasibility


# TESTING
problem = InternalProblemData('../../data/input_data/medium_testcase.xlsx', precedence=True)
for node in problem.nodes.values():
    print(node)
sol = Solution(problem, debug=True)
# sol.check_load_feasibility(problem.nodes['o_1'], 'v_1', 1)

# print("SMALL_TESTCASE_ONE_VESSEL INSERTIONS")
print()
print("Initial routes:", sol.routes)
print("Initial e and l:", {v: list(zip(sol.e[v], sol.l[v])) for v in problem.vessels})
print()
# problem.vessel_ton_capacities['v_1'] = 89
# problem.vessel_nprod_capacities['v_1'] = 3

# SMALL TESTCASE ONE VESSEL
# vessel='v_1'
# sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_1', vessel=vessel, idx=1)
# sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_2', vessel=vessel, idx=1)
# sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_3', vessel=vessel, idx=1)
# sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_1', vessel=vessel, idx=len(sol.routes[vessel]))

# MEDIUM TESTCASE
vessel = 'v_1'
print(f"INITIAL ROUTE, VESSEL {vessel}: {sol.routes[vessel]}")
sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_5', vessel=vessel, idx=1)
sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_4', vessel=vessel, idx=1)
sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_2', vessel=vessel, idx=len(sol.routes[vessel]))
vessel = 'v_2'
print(f"INITIAL ROUTE, VESSEL {vessel}: {sol.routes[vessel]}")
sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_3', vessel=vessel, idx=1)
sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_1', vessel=vessel, idx=1)
sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_6', vessel=vessel, idx=1)
sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_2', vessel=vessel, idx=len(sol.routes[vessel]))

# LARGE TESTCASE
# vessel = 'v_1'
# sol.check_insertion_feasibility_insert_and_print('o_6', vessel, 1)
# sol.check_insertion_feasibility_insert_and_print('o_4', vessel, 1)
# sol.check_insertion_feasibility_insert_and_print('f_1', vessel, len(sol.routes[vessel]))
# vessel = 'v_2'
# sol.check_insertion_feasibility_insert_and_print('o_9', vessel, 1)
# sol.check_insertion_feasibility_insert_and_print('o_11', vessel, 1)
# sol.check_insertion_feasibility_insert_and_print('f_2', vessel, len(sol.routes[vessel]))
# sol.check_insertion_feasibility_insert_and_print('o_3', vessel, 1)
# sol.check_insertion_feasibility_insert_and_print('o_5', vessel, 1)

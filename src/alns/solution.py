from __future__ import annotations
from typing import List, Dict, Tuple
import math
import numpy as np
import bisect
import copy
from src.read_problem_data import ProblemData
from src.util.print import bcolors

int_inf = 9999


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


class ProblemDataExtended(ProblemData):

    def __init__(self, file_path: str, precedence: bool = False) -> None:
        super().__init__(file_path)
        self.precedence = precedence
        self._init_nodes()
        self._init_quay_capacities()

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

    def _init_quay_capacities(self) -> None:
        quay_capacity = {factory: [cap for (f, t), cap in self.factory_max_vessels_loading.items() if f == factory]
                         for factory in self.factory_nodes}
        quay_cap_incr_times = {factory: [t for t in range(1, len(quay_capacity[factory]))
                                         if quay_capacity[factory][t - 1] < quay_capacity[factory][t]]
                               for factory in self.factory_nodes}
        quay_cap_decr_times = {factory: [t for t in range(1, len(quay_capacity[factory]))
                                         if quay_capacity[factory][t - 1] > quay_capacity[factory][t]]
                               for factory in self.factory_nodes}
        self.quay_capacity: Dict[str, List[int]] = quay_capacity
        self.quay_cap_incr_times: Dict[str, List[int]] = quay_cap_incr_times
        self.quay_cap_decr_times: Dict[str, List[int]] = quay_cap_decr_times


class Solution:

    def __init__(self, prbl: ProblemDataExtended, debug: bool = False) -> None:
        self.prbl = prbl
        self.debug = debug

        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.e: Dict[str, List[int]] = {v: [max(1, self.prbl.start_times_for_vessels[v])] for v in self.prbl.vessels}
        self.l: Dict[str, List[int]] = {v: [len(self.prbl.time_periods)] for v in self.prbl.vessels}
        self.factory_visits: Dict[str, List[str]] = self._init_factory_visits()
        self.factory_visits_route_index: Dict[str, List[int]] = {f: [0 for _ in self.factory_visits[f]]
                                                                 for f in self.prbl.factory_nodes}

        self.temp_routes: Dict[str, List[str]] = copy.deepcopy(self.routes)
        self.temp_e: Dict[str, List[int]] = copy.deepcopy(self.e)
        self.temp_l: Dict[str, List[int]] = copy.deepcopy(self.l)
        self.temp_factory_visits: Dict[str, List[str]] = copy.deepcopy(self.factory_visits)
        self.temp_factory_visits_route_index: Dict[str, List[int]] = copy.deepcopy(self.factory_visits_route_index)

        self.verbose = True

    def __repr__(self) -> str:
        return (f"Routes: {self.routes}")

    def copy(self) -> Solution:
        solution_copy = Solution(self.prbl)  # problem data is static

        solution_copy.routes = {vessel: route[:] for vessel, route in self.routes.items()}
        solution_copy.e = {vessel: e[:] for vessel, e in self.e.items()}
        solution_copy.l = {vessel: l[:] for vessel, l in self.l.items()}
        solution_copy.factory_visits = {factory: visits[:] for factory, visits in self.factory_visits.items()}
        solution_copy.factory_visits_route_index = {factory: visit_route_idxs[:]
                                                    for factory, visit_route_idxs in
                                                    self.factory_visits_route_index.items()}

        solution_copy.temp_routes = {vessel: route[:] for vessel, route in self.temp_routes.items()}
        solution_copy.temp_e = {vessel: e[:] for vessel, e in self.temp_e.items()}
        solution_copy.temp_l = {vessel: l[:] for vessel, l in self.temp_l.items()}
        solution_copy.temp_factory_visits = {factory: visits[:] for factory, visits in self.temp_factory_visits.items()}
        solution_copy.temp_factory_visits_route_index = {factory: visit_route_idxs[:]
                                                         for factory, visit_route_idxs in
                                                         self.temp_factory_visits_route_index.items()}

        solution_copy.verbose = self.verbose
        return solution_copy

    def insert_last_checked(self):
        self.routes = {vessel: route[:] for vessel, route in self.temp_routes.items()}
        self.e = {vessel: e[:] for vessel, e in self.temp_e.items()}
        self.l = {vessel: l[:] for vessel, l in self.temp_l.items()}
        self.factory_visits = {factory: visits[:] for factory, visits in self.temp_factory_visits.items()}
        self.factory_visits_route_index = {factory: visit_route_idxs[:]
                                           for factory, visit_route_idxs in
                                           self.temp_factory_visits_route_index.items()}

    def clear_last_checked(self):
        self.temp_routes = {vessel: route[:] for vessel, route in self.routes.items()}
        self.temp_e = {vessel: e[:] for vessel, e in self.e.items()}
        self.temp_l = {vessel: l[:] for vessel, l in self.l.items()}
        self.temp_factory_visits = {factory: visits[:] for factory, visits in self.factory_visits.items()}
        self.temp_factory_visits_route_index = {factory: visit_route_idxs[:]
                                                for factory, visit_route_idxs in self.factory_visits_route_index.items()}


    def check_insertion_feasibility(self, node_id: str, vessel: str, idx: int) -> bool:
        node = self.prbl.nodes[node_id]
        idx = len(self.routes[vessel]) if idx > len(self.routes[vessel]) else idx
        # Checks that do NOT assume node is inserted in temp:
        if not self.check_load_feasibility(node, vessel, idx):
            return False

        if not self.check_no_products_feasibility(node, vessel, idx):
            return False

        if not self.check_time_feasibility(node_id, vessel, idx):
            return False

        # Checks that do assume that node is inserted in temp:
        if not self.check_final_factory_destination_feasibility(vessel, idx):
            return False

        if not self.check_production_feasibility(vessel, idx):
            return False

        return True

    def get_earliest(self, idx: int, vessel: str) -> int:
        route = self.temp_routes[vessel]
        node = self.prbl.nodes[route[idx]]

        prev_node_id = route[idx - 1] if idx > 0 else None
        prev_e = self.temp_e[vessel][idx - 1] if prev_node_id else self.prbl.start_times_for_vessels[vessel]

        prev_transport_time = self.prbl.transport_times[prev_node_id, node.id] if prev_node_id else 0
        prev_loading_unloading_time = self.prbl.loading_unloading_times[vessel, prev_node_id] if prev_node_id else 0
        earliest = max(node.tw_start, prev_e + prev_loading_unloading_time + prev_transport_time)

        # If the precedence extension is included, earliest visiting time must also incorporate minimum waiting time
        if self.prbl.precedence and prev_node_id and (
                (self.prbl.nodes[prev_node_id].zone, self.prbl.nodes[node.id].zone) in
                [("red", "green"), ("red", "yellow"), ("yellow", "green")]):
            earliest = max(earliest, prev_e + prev_loading_unloading_time + self.prbl.min_wait_if_sick_abs)
        return earliest

    def get_latest(self, idx: int, vessel: str) -> int:
        route = self.temp_routes[vessel]
        node = self.prbl.nodes[route[idx]]
        next_node_id = route[idx + 1] if idx + 1 < len(route) else None
        next_l = self.temp_l[vessel][idx + 1] if next_node_id else len(self.prbl.time_periods) - 1

        next_transport_time = self.prbl.transport_times[node.id, next_node_id] if next_node_id else 0

        if idx == len(route) - 1 and node.is_factory:
            loading_unloading_time = 0  # zero loading time for last visited factory
        else:
            loading_unloading_time = self.prbl.loading_unloading_times[vessel, node.id]

        latest = min(node.tw_end, next_l - next_transport_time - loading_unloading_time)

        # if the last node is not a factory, make sure there is enough time to reach a factory destination
        if idx == len(route) - 1 and not node.is_factory:
            time_to_factory = min(self.prbl.transport_times[node.id, f] for f in self.prbl.factory_nodes)
            latest = min(latest, len(self.prbl.time_periods) - time_to_factory - loading_unloading_time - 1)

        # If the precedence extension is included, latest visiting time must also incorporate minimum waiting time
        if self.prbl.precedence and next_node_id is not None and (
                (node.zone, self.prbl.nodes[next_node_id].zone) in
                [("red", "green"), ("red", "yellow"), ("yellow", "green")]):
            latest = min(latest, next_l - loading_unloading_time - self.prbl.min_wait_if_sick_abs)
        return latest

    def get_factory_earliest(self, factory: str, idx: int,
                             prev_departure_times: List[int] = None) -> Tuple[int, List[int]]:
        """
        :param factory:
        :param idx: The index of the visit in self.factory_visits
        :param prev_departure_times: The previous visits' earliest departure times
        :return: The earliest visit time based on factory constraints
                 (route constraints are assumed to be already calculated)
        """
        quay_cap_incr_times = self.prbl.quay_cap_incr_times[factory] + [int_inf]
        visits = self.temp_factory_visits[factory]
        visit_indices = self.temp_factory_visits_route_index[factory]
        vessel = visits[idx]
        # destination factory -> zero loading time
        loading_times = [self.prbl.loading_unloading_times[visits[i], factory]
                         * (not self._is_destination_factory(visits[i], visit_indices[i]))
                         for i in range(len(visits))]
        curr_loading_time = loading_times[idx]
        prev_e = self.temp_e[visits[idx - 1]][visit_indices[idx - 1]] if idx > 0 else -1
        if not prev_departure_times:
            prev_departure_times = [self.temp_e[visits[i]][visit_indices[i]] + loading_times[i] - 1
                                    for i in range(0, idx)] + [int_inf]

        idx_e = self.temp_e[vessel][visit_indices[idx]]  # earliest from route
        t = max(prev_e, idx_e)

        # update event lists
        prev_departure_times.sort()
        prev_departure_times = self._ordered_list_min_threshold(prev_departure_times, t)
        quay_cap_incr_times = self._ordered_list_min_threshold(quay_cap_incr_times, t)

        # both quay capacity and the departures of other vessels limit insertion
        # -> iterate to find first possible t where quay_vessels < min quay_capacity over loading interval
        quay_vessels = len(prev_departure_times) - 1

        while curr_loading_time > 0 and quay_vessels >= self._get_min_quay_capacity(factory, t, curr_loading_time):
            if prev_departure_times[0] < quay_cap_incr_times[0]:
                t = prev_departure_times.pop(0) + 1
                quay_vessels -= 1
            else:
                t = quay_cap_incr_times.pop(0)

        if curr_loading_time > 0:
            bisect.insort(prev_departure_times, t + curr_loading_time - 1)
        return t, prev_departure_times

    def get_factory_latest(self, factory: str, idx: int,
                           next_arrival_times: List[int] = None) -> Tuple[int, List[int]]:
        """
        :param factory:
        :param idx: The index of the visit in self.factory_visits
        :param next_arrival_times: the succeeding visits' latest arrival times
        :return: The latest visit time based on factory constraints
        """
        quay_cap_decr_times = [-int_inf] + self.prbl.quay_cap_decr_times[factory]
        visits = self.temp_factory_visits[factory]
        visit_indices = self.temp_factory_visits_route_index[factory]
        vessel = visits[idx]
        # destination factory -> zero loading time
        curr_loading_time = (self.prbl.loading_unloading_times[vessel, factory]
                             * (not self._is_destination_factory(vessel, visit_indices[idx])))
        next_l = self.temp_l[visits[idx + 1]][visit_indices[idx + 1]] if idx + 1 < len(visits) else int_inf
        if not next_arrival_times:
            next_arrival_times = [-int_inf] + [self.temp_l[visits[i]][visit_indices[i]]
                                               for i in range(idx + 1, len(visits))]

        idx_l = self.temp_l[vessel][visit_indices[idx]]  # latest from route
        t = min(next_l, idx_l)

        # update event lists
        next_arrival_times.sort()
        next_arrival_times = self._ordered_list_max_threshold(next_arrival_times, t + curr_loading_time + 1)
        quay_cap_decr_times = self._ordered_list_max_threshold(quay_cap_decr_times, t + curr_loading_time + 1)

        # both quay capacity and the departures of other vessels limit insertion
        # -> iterate to find first possible t where quay_vessels < min quay_capacity over loading interval
        quay_vessels = len(next_arrival_times) - 1

        while curr_loading_time > 0 and quay_vessels >= self._get_min_quay_capacity(factory, t, curr_loading_time):
            if next_arrival_times[-1] >= quay_cap_decr_times[-1]:
                t = next_arrival_times.pop() - curr_loading_time
                quay_vessels -= 1
            else:
                t = quay_cap_decr_times.pop() - curr_loading_time

        if curr_loading_time > 0:
            bisect.insort(next_arrival_times, t)
        return t, next_arrival_times

    def check_earliest_forward(self, vessel: str, idx: int, force_propagation: bool = False) -> bool:
        """Iteratively checks earliest time for succeeding visits until no change"""
        route = self.temp_routes[vessel]
        for i in range(idx, len(route)):
            e = self.get_earliest(i, vessel)
            updated_e = e > self.temp_e[vessel][i]
            self.temp_e[vessel][i] = e if updated_e else self.temp_e[vessel][
                i]  # update temp_e if stronger bound is found

            if self.temp_l[vessel][i] < e:  # insertion squeezes out succeeding node
                if self.verbose:
                    print(f"Check failed at: check_earliest_forward for {vessel}. "
                          f"Forward from route index {idx} at {i}: ({e}, {self.temp_l[vessel][i]})")
                return False

            node = self.prbl.nodes[route[i]]
            if node.is_factory:  # check routes visiting the same factory before than this visit
                factory_visit_idx = self._get_factory_visit_idx(node.id, vessel, i)
                if not self.check_factory_visits_earliest_forward(node.id, factory_visit_idx + 1):
                    return False  # the changed e for the factory resulted in a node on another route being squeezed out

            if not updated_e and not force_propagation:  # propagation of check stops if e is unchanged
                break

        return True

    def check_latest_backward(self, vessel: str, idx: int, force_propagation: bool = False) -> bool:
        """Iteratively checks latest time for preceding visits until no change"""
        route = self.temp_routes[vessel]
        for i in range(idx - 1, -1, -1):
            l = self.get_latest(i, vessel)
            updated_l = l < self.temp_l[vessel][i]
            self.temp_l[vessel][i] = l if updated_l else self.temp_l[vessel][
                i]  # update temp_l if stronger bound is found

            if l < self.temp_e[vessel][i]:  # insertion squeezes out preceding node
                if self.verbose:
                    print(f"Check failed at: check_latest_backward for {vessel}. "
                          f"Backward from route index {idx} at {i}: ({self.temp_e[vessel][i]}, {l})")
                return False

            node = self.prbl.nodes[route[i]]
            if node.is_factory:  # check routes visiting the same factory after than this visit
                factory_visit_idx = self._get_factory_visit_idx(node.id, vessel, i)
                if not self.check_factory_visits_latest_backward(node.id, factory_visit_idx):
                    return False  # the changed l for the factory resulted in a node on another route being squeezed out

            if not updated_l and not force_propagation:  # propagation of check stops if l is unchanged
                break

        return True

    def check_factory_visits_earliest_forward(self, factory: str, idx: int, prev_dep_times: List[int] = None,
                                              force_propagation: bool = False) -> bool:
        """
        Iteratively checks a factory visits' earliest arrival times for succeeding vessels until violation or no overlap
        :param factory:
        :param idx: the factory visit index to check from
        :param prev_dep_times:
        :return: True if there is no constraint violation
        """
        for i in range(idx, len(self.factory_visits[factory])):
            vessel = self.temp_factory_visits[factory][i]
            route_index = self.temp_factory_visits_route_index[factory][i]

            e, prev_dep_times = self.get_factory_earliest(factory, i, prev_departure_times=prev_dep_times)
            updated_e = e > self.temp_e[vessel][route_index]
            if updated_e:
                self.temp_e[vessel][route_index] = e

            if self.temp_l[vessel][route_index] < e:
                if self.verbose:
                    print(f"Check failed at: check_factory_visits_earliest_forward for {factory}. "
                          f"Forward from factory visit {idx} at {i}: : ({e}, {self.temp_l[vessel][route_index]})")
                return False  # not enough time space for insertion

            if updated_e:
                self.temp_e[vessel][route_index] = e

                if not self.check_earliest_forward(vessel, route_index + 1):
                    return False

            if max(prev_dep_times[:-1], default=-int_inf) < self.temp_e[vessel][route_index] and not force_propagation:
                break  # stop propagation if no overlap between visit i and previous departures
        return True

    def check_factory_visits_latest_backward(self, factory: str, idx: int, next_arr_times: List[int] = None,
                                             force_propagation: bool = False) -> bool:
        """
        Iteratively checks a factory visits' latest arrival times for preceding vessels until violation or no overlap
        :param factory:
        :param idx: the factory visit index to check from
        :param next_arr_times:
        :return: True if there is no constraint violation
        """
        for i in range(idx - 1, -1, -1):
            vessel = self.temp_factory_visits[factory][i]
            route_index = self.temp_factory_visits_route_index[factory][i]

            l, next_arr_times = self.get_factory_latest(factory, i, next_arrival_times=next_arr_times)
            updated_l = l < self.temp_l[vessel][route_index]

            if l < self.temp_e[vessel][route_index]:
                if self.verbose:
                    print(f"Check failed at: check_factory_visits_latest_backward for {factory}. "
                          f"Backward from factory visit {idx} at {i}: ({self.temp_e[vessel][route_index]}, {l})")
                return False  # not enough time space for insertion

            if updated_l:
                self.temp_l[vessel][route_index] = l

                if not self.check_latest_backward(vessel, route_index):
                    return False

            if self.temp_l[vessel][route_index] < min(next_arr_times[-1:], default=int_inf) and not force_propagation:
                break  # stop propagation if no overlap between visit i and next arrivals
        return True

    def check_time_feasibility(self, insert_node_id: str, vessel: str, idx: int) -> bool:
        route = self.temp_routes[vessel]
        insert_node = self.prbl.nodes[insert_node_id]
        idx = len(route) + idx + 1 if idx < 0 else idx  # transform negative indexes

        # Increase route indexes for factory visits succeeding the insert
        for factory in self.prbl.factory_nodes:
            for i, v in enumerate(self.temp_factory_visits[factory]):
                if v == vessel and idx <= self.temp_factory_visits_route_index[factory][i]:
                    self.temp_factory_visits_route_index[factory][i] += 1
        self.temp_routes[vessel].insert(idx, insert_node_id)
        self.temp_e[vessel].insert(idx, insert_node.tw_start)  # initial value
        self.temp_l[vessel].insert(idx, insert_node.tw_end)  # initial value

        e = self.get_earliest(idx, vessel)
        l = self.get_latest(idx, vessel)

        if l < e:  # not enough time space for insertion
            if self.verbose:
                print(f"Check failed at: check_time_feasibility for {vessel}: ({e}, {l})")
            return False

        self.temp_e[vessel][idx] = e
        self.temp_l[vessel][idx] = l

        if insert_node.is_factory:
            factory_visit_idx = self._get_factory_visit_insert_idx(insert_node_id, e)
            self.temp_factory_visits[insert_node_id].insert(factory_visit_idx, vessel)
            self.temp_factory_visits_route_index[insert_node_id].insert(factory_visit_idx, idx)

        if not self.check_latest_backward(vessel, idx):
            return False

        if not self.check_earliest_forward(vessel, idx + 1):
            return False

        if insert_node.is_factory:
            e, prev_dep_times = self.get_factory_earliest(insert_node_id, factory_visit_idx)
            l, next_arr_times = self.get_factory_latest(insert_node_id, factory_visit_idx)

            if l < e:
                return False

            self.temp_e[vessel][idx] = max(e, self.temp_e[vessel][idx])
            self.temp_l[vessel][idx] = min(l, self.temp_l[vessel][idx])

            if not self.check_factory_visits_earliest_forward(insert_node_id, factory_visit_idx + 1, prev_dep_times):
                return False

            if not self.check_factory_visits_latest_backward(insert_node_id, factory_visit_idx, next_arr_times):
                return False

        # if node is inserted after a destination factory, we must update e for this factory's visits for
        if idx == len(route) - 1 and self.prbl.nodes[route[-2]].is_factory:
            return self.check_factory_visits_earliest_forward(route[-2], 1)

        return True

    def check_final_factory_destination_feasibility(self, vessel: str, idx: int):
        # Check assumes that the insertion is already present in temp variables
        node_id = self.temp_routes[vessel][idx]
        node = self.prbl.nodes[node_id]

        if node.is_factory:
            if self.verbose:
                print(f"check_final_factory_destination_feasibility failed for {vessel}, {node} inserted at {idx}")
            return (len([v for v in self.prbl.vessels if self.temp_routes[v][-1] == node_id])
                    <= self.prbl.factory_max_vessels_destination[node_id])
        else:  # destination factories are unchanged or 'removed'
            return True

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

    def check_production_feasibility(self, vessel: str, idx: int):
        # Check assumes that the insertion is already present in temp variables

        # Find which factories to check
        factories_to_check = []
        for f in self.prbl.factory_nodes.keys():
            if (not self.prbl.nodes[self.temp_routes[vessel][idx]].is_factory and
                    self.get_temp_voyage_start_factory(vessel=vessel, idx=idx) == f):  # added order picked up at f
                factories_to_check.append(f)
            elif self.is_factory_latest_changed_in_temp(f):  # factory may have to be ready for vessel loading earlier
                factories_to_check.append(f)

        # Feasibility is checked for relevant factories
        for factory_node_id in factories_to_check:
            # voyage_start_idxs: {vessel: (index in route, temp latest factory loading time)}
            # for vessels visiting factory with id factory_node_id and visit is not at the end of the route
            voyage_start_idxs: Dict[str, List[Tuple[int, int]]] = self.get_temp_voyage_start_idxs_for_factory(
                factory_node_id)

            # Collect quantity to be delivered for each voyage
            products_for_voyage: List[List[int]] = []
            latest_loading_times: List[int] = []

            for v in voyage_start_idxs.keys():
                for i in range(len(voyage_start_idxs[v])):
                    stop_index = self.get_temp_voyage_end_idx(vessel=v, start_idx=voyage_start_idxs[v][i][0])
                    add_rows = [self.prbl.nodes[node_id].demand
                                for node_id in self.temp_routes[v][voyage_start_idxs[v][i][0] + 1:stop_index]]

                    # Only add the row if demand was found for the route
                    # Matrix products_for_voyage - rows: voyages (index i), columns: products
                    # List latest_loading_times - latest loading time (index i)
                    if len(add_rows) > 0:
                        prod_sums = []
                        for p in range(len(self.prbl.products)):
                            prod_sums.append(sum(row[p] for row in add_rows))
                        products_for_voyage.append(prod_sums)
                        latest_loading_times.append(voyage_start_idxs[v][i][1])

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
                    if self.verbose:
                        print(f"check_production_feasibility failed on production for {factory_node_id}")
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
                    if self.verbose:
                        print(f"check_production_feasibility failed on inventory for {factory_node_id}")
                    return False
        return True

    def remove_node(self, vessel: str, idx: int):
        self.routes[vessel].pop(idx)
        self.e[vessel].pop(idx)
        self.l[vessel].pop(idx)
        self.temp_routes[vessel].pop(idx)
        self.temp_e[vessel].pop(idx)
        self.temp_l[vessel].pop(idx)

    def recompute_solution_variables(self):
        # recompute factory visit route indexes
        self.factory_visits_route_index = self.recompute_factory_visits_route_idx()
        self.temp_factory_visits_route_index = copy.deepcopy(self.factory_visits_route_index)

        # "open up" temp_e and temp_l to original time window
        for vessel, route in self.routes.items():
            for idx, node_id in enumerate(route):
                self.temp_e[vessel][idx] = self.prbl.nodes[node_id].tw_start
                self.temp_l[vessel][idx] = self.prbl.nodes[node_id].tw_end

        # recompute new e and l for routes
        for vessel, route in self.routes.items():
            self.check_earliest_forward(vessel, 0, force_propagation=True)
            self.check_latest_backward(vessel, len(route) - 1, force_propagation=True)

        # recompute new e and l for factory visits
        for factory, factory_visits in self.factory_visits.items():
            self.check_factory_visits_earliest_forward(factory, 0, force_propagation=True)
            self.check_factory_visits_latest_backward(factory, len(factory_visits) - 1, force_propagation=True)

        # move updates from temp to main variables
        self.insert_last_checked()

    def recompute_factory_visits_route_idx(self) -> Dict[str, List[int]]:
        # infer factory visit indexes from factory visits and routes
        factory_visits_route_idx = {factory: [] for factory in self.prbl.factory_nodes}
        for factory, factory_visits in self.factory_visits.items():
            vessel_prev_factory_idx = {vessel: -1 for vessel in self.prbl.vessels}
            for vessel in factory_visits:
                route_idx = self.routes[vessel].index(factory, vessel_prev_factory_idx[vessel] + 1)
                vessel_prev_factory_idx[vessel] = route_idx
                factory_visits_route_idx[factory].append(route_idx)
        return factory_visits_route_idx

    def get_solution_cost(self) -> int:
        transport_cost = 0
        unmet_orders = list(self.prbl.order_nodes.keys())

        for vessel in self.prbl.vessels:
            route = self.routes[vessel]
            for i in range(1, len(route)):
                transport_cost += self.prbl.transport_times[route[i - 1], route[i]] * self.prbl.transport_unit_costs[
                    vessel]
                if not self.prbl.nodes[route[i]].is_factory:
                    unmet_orders.remove(route[i])

        unmet_order_cost = 0
        for order_node in unmet_orders:
            unmet_order_cost += self.prbl.external_delivery_penalties[order_node]

        return transport_cost + unmet_order_cost


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

    def get_temp_voyage_start_idxs_for_factory(self, factory_node_id: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        :param factory_node_id:
        :return: {vessel: (factory visit index in route, latest loading start time)} for the input factory
                            if vessel visits input factory for loading
        """
        voyage_start_idxs_for_vessels: Dict[str, List[Tuple[int, int]]] = {}
        for v in self.prbl.vessels:
            voyage_start_idxs_for_vessels[v] = []
            route = self.temp_routes[v]

            # Adding index and latest loading time for vessels loading at input factory
            for i in range(len(route) - 1):  # last element in route is not included, as it cannot be a voyage start
                if self.prbl.nodes[route[i]].id == factory_node_id:
                    voyage_start_idxs_for_vessels[v].append(
                        tuple((i, self.temp_l[v][i])))

            # Vessel does not load at input factory -> vessel is skipped
            if len(voyage_start_idxs_for_vessels[v]) == 0:
                voyage_start_idxs_for_vessels.pop(v, None)
        return voyage_start_idxs_for_vessels

    def is_factory_latest_changed_in_temp(self, factory_node_id: str) -> bool:
        if (self.temp_factory_visits[factory_node_id] != self.factory_visits[factory_node_id] or
                self.temp_factory_visits_route_index[factory_node_id] != self.factory_visits_route_index[
                    factory_node_id] or
                not all(self.temp_l[self.temp_factory_visits[factory_node_id][i]][
                            self.temp_factory_visits_route_index[factory_node_id][i]] ==
                        self.l[self.factory_visits[factory_node_id][i]][
                            self.factory_visits_route_index[factory_node_id][i]]
                        for i in range(len(self.factory_visits[factory_node_id])))):
            return True
        return False

    def get_voyage_start_end_idx(self, vessel: str, idx: int) -> Tuple[int, int]:
        route = self.temp_routes[vessel]
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

    def get_temp_voyage_end_idx(self, vessel: str, start_idx: int) -> int:
        route = self.temp_routes[vessel]
        for i in range(start_idx + 1, len(route)):
            if self.prbl.nodes[route[i]].is_factory:
                return i
        return len(route)

    def get_temp_voyage_start_factory(self, vessel: str, idx: int) -> str:
        route = self.temp_routes[vessel]
        for i in range(idx - 1, 0, -1):
            if self.prbl.nodes[route[i]].is_factory:
                return self.temp_routes[vessel][i]
        return self.temp_routes[vessel][0]

    def _init_factory_visits(self) -> Dict[str, List[str]]:
        vessel_starting_times = list(self.prbl.start_times_for_vessels.items())
        vessel_starting_times.sort(key=lambda item: item[1])
        factory_visits: Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}

        for vessel, _ in vessel_starting_times:
            initial_factory = self.prbl.vessel_initial_locations[vessel]
            factory_visits[initial_factory].append(vessel)
        return factory_visits

    def _is_destination_factory(self, vessel: str, route_index: int) -> bool:
        return route_index == len(self.temp_routes[vessel]) - 1

    def _get_min_quay_capacity(self, factory: str, t: int, loading_time: int):
        """Helper function that defines min_quay_capacity for t < 0, t >= no_time_periods and loading_time=0"""
        no_time_periods = len(self.prbl.quay_capacity[factory])
        # print(list(range(t, min(no_time_periods, t + loading_time))))
        if t + loading_time <= 0:
            return self.prbl.quay_capacity[factory][0]
        elif t < no_time_periods and loading_time:
            return min(self.prbl.quay_capacity[factory][max(0, tau)]
                       for tau in range(t, min(no_time_periods, t + loading_time)))
        elif t < no_time_periods and not loading_time:
            return self.prbl.quay_capacity[factory][t]
        else:  # t >= no_time_periods
            return self.prbl.quay_capacity[factory][-1]

    def _get_factory_visit_insert_idx(self, factory: str, earliest: int):
        visits = self.temp_factory_visits[factory]
        visit_indices = self.temp_factory_visits_route_index[factory]
        # insertion where all previous factory visits have e < earliest
        i = bisect.bisect([self.temp_e[v][visit_indices[i]] for i, v in enumerate(visits)], earliest)
        return i

    def _get_factory_visit_idx(self, factory: str, vessel: str, route_idx: int) -> int:
        # route_idx = len(self.routes[vessel]) + route_idx - 1 if route_idx < 0 else route_idx
        for i, (v, idx) in enumerate(
                zip(self.temp_factory_visits[factory], self.temp_factory_visits_route_index[factory])):
            if v == vessel and idx == route_idx:
                return i
        raise IndexError(f"Route index {route_idx} does not exist for {vessel} visiting {factory}")

    @staticmethod
    def _ordered_list_min_threshold(ordered_list: List[int], min_threshold: int) -> List[int]:
        """Prunes the ordered_list to only contain values >= min_threshold"""
        i = bisect.bisect(ordered_list, min_threshold)
        return ordered_list[i:]

    @staticmethod
    def _ordered_list_max_threshold(ordered_list: List[int], max_threshold: int) -> List[int]:
        """Prunes the ordered_list to only contain values <= max_threshold"""
        i = bisect.bisect_left(ordered_list, max_threshold)
        return ordered_list[:i]

    def get_orders_not_served(self) -> List[str]:
        served_orders = set(o for v in self.prbl.vessels for o in self.routes[v])
        return list(set(self.prbl.order_nodes) - served_orders)

    def print_routes(self, highlight: List[Tuple[str, int]] = None):
        highlight = [] if not highlight else highlight
        for vessel, route in self.routes.items():
            s = ''
            for i, (node, e, l) in enumerate(zip(self.routes[vessel], self.e[vessel], self.l[vessel])):
                if (vessel, i) in highlight:
                    s += f'{bcolors.OKGREEN}{node} ({e},{l}){bcolors.RESET_ALL}, '
                elif i == len(self.routes[vessel]) - 1 and self.prbl.nodes[node].is_factory:
                    s += f'{bcolors.GREY}{node} ({e},{l}){bcolors.RESET_ALL}'
                else:
                    s += f'{node} ({e},{l}), '
            print(f'{vessel}: {s}')

    def print_factory_visits(self, highlight: List[Tuple[str, int]] = None):
        highlight = [] if not highlight else highlight
        for factory, visits in self.factory_visits.items():
            s = ''
            for vessel, route_idx in zip(visits, self.factory_visits_route_index[factory]):
                if (vessel, route_idx) in highlight:
                    s += f'{bcolors.OKGREEN}{vessel} ({self.e[vessel][route_idx]}, {self.l[vessel][route_idx]})' \
                         f'{bcolors.RESET_ALL}, '
                elif self._is_destination_factory(vessel, route_idx):
                    s += f'{bcolors.GREY}{vessel} ({self.e[vessel][route_idx]}, {self.l[vessel][route_idx]})' \
                         f'{bcolors.RESET_ALL}, '
                else:
                    s += f'{vessel} ({self.e[vessel][route_idx]},{self.l[vessel][route_idx]}), '
            print(f'{factory}: {s}')

    def check_insertion_feasibility_insert_and_print(self, insert_node_id: str, vessel: str, idx: int):
        insert_node = self.prbl.nodes[insert_node_id]
        print(
            f"{bcolors.BOLD}Checking for insertion of node {insert_node_id} at index {idx} at route of vessel {vessel}{bcolors.ENDC}")
        print("Utility of insertion:", self.get_insertion_utility(self.prbl.nodes[insert_node_id], vessel, idx))
        feasibility = self.check_insertion_feasibility(insert_node_id, vessel, idx)
        if not feasibility and self.debug:
            print(f"{bcolors.FAIL}Infeasible insertion - node not inserted{bcolors.ENDC}")
            print("> Final factory destination feasibility:",
                  self.check_final_factory_destination_feasibility(vessel, idx))
            print("> Production feasibility:",
                  self.check_production_feasibility(vessel=vessel, idx=idx))
            print("> Load feasibility:", self.check_load_feasibility(insert_node, vessel, idx))
            print("> Number of products feasibility:", self.check_no_products_feasibility(insert_node, vessel, idx))
            print("> Time feasibility:", self.check_time_feasibility(insert_node_id, vessel, idx))
            self.clear_last_checked()
        else:
            print(f"{bcolors.OKGREEN}Insertion is feasible - inserting node{bcolors.ENDC}")
            self.insert_last_checked()
            print(vessel, self.routes[vessel])
            print(list(zip(self.e[vessel], self.l[vessel])))
        print()
        return feasibility


if __name__ == '__main__':
    problem = ProblemDataExtended('../../data/input_data/large_testcase.xlsx', precedence=True)
    sol = Solution(problem)

    insertions = [  # large testcase
        ('o_1', 'v_1', 1),
        ('f_1', 'v_1', 2),
        ('o_4', 'v_1', 2),
        ('o_2', 'v_1', 3),
        ('f_1', 'v_2', 1),
        ('o_1', 'v_2', 2),
        ('f_1', 'v_2', 3),
        ('o_9', 'v_3', 1),
        ('f_1', 'v_3', 2),
        ('o_6', 'v_3', 2),
        ('o_7', 'v_3', 2),
        ('o_8', 'v_3', 2),
    ]

    for node, vessel, idx in insertions:
        print(f'Inserting {node} into {vessel} at {idx}.')
        if sol.check_insertion_feasibility(node, vessel, idx):
            sol.insert_last_checked()
        else:
            sol.clear_last_checked()
        sol.print_routes(highlight=[(vessel, idx)])
        print()
        sol.print_factory_visits(highlight=[(vessel, idx)])
        print("\n\n")

    # SMALL TESTCASE ONE VESSEL
    # vessel='v_1'
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_1', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_2', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_3', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_1', vessel=vessel, idx=len(sol.routes[vessel]))

    # MEDIUM TESTCASE
    # vessel = 'v_1'
    # print(f"INITIAL ROUTE, VESSEL {vessel}: {sol.routes[vessel]}")
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_5', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_4', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_2', vessel=vessel, idx=len(sol.routes[vessel]))
    # vessel = 'v_2'
    # print(f"INITIAL ROUTE, VESSEL {vessel}: {sol.routes[vessel]}")
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_3', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_1', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='o_6', vessel=vessel, idx=1)
    # sol.check_insertion_feasibility_insert_and_print(insert_node_id='f_2', vessel=vessel, idx=len(sol.routes[vessel]))

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
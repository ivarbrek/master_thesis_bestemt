from typing import List, Dict, Tuple, Union
from src.read_problem_data import ProblemData
import math
import numpy as np
import bisect
int_inf = 9999


class Node:
    id: str
    tw_start: int = -1
    tw_end: int = math.inf
    demand: List[int] = []

    def __init__(self, name, is_factory=False):
        self.id: str = name
        self.is_factory: bool = is_factory

    def __repr__(self):
        return f"{self.id}: ({self.tw_start}, {self.tw_end}), {self.demand}"


class InternalProblemData(ProblemData):

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
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

    def __init__(self, prbl: InternalProblemData) -> None:
        self.prbl = prbl
        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.e: Dict[str, List[int]] = {v: [max(1, self.prbl.start_times_for_vessels[v])] for v in self.prbl.vessels}
        self.l: Dict[str, List[int]] = {v: [len(self.prbl.time_periods)] for v in self.prbl.vessels}
        self.factory_visits: Dict[str, List[str]] = self._init_factory_visits()
        self.factory_visits_route_index: Dict[str, List[int]] = {f: [0 for _ in self.factory_visits[f]]
                                                                 for f in self.prbl.factory_nodes}

        self.temp_routes: Dict[str, List[str]] = self.routes.copy()
        self.temp_e: Dict[str, List[int]] = self.e.copy()
        self.temp_l: Dict[str, List[int]] = self.l.copy()
        self.temp_factory_visits: Dict[str, List[str]] = self.factory_visits.copy()
        self.temp_factory_visits_route_index: Dict[str, List[int]] = self.factory_visits_route_index.copy()

        self._init_factory_visits_e_and_l()
        self.verbose = True

    def temp_to_solution(self):
        self.routes = self.temp_routes
        self.e = self.temp_e
        self.l = self.temp_l
        self.factory_visits = self.temp_factory_visits
        self.factory_visits_route_index = self.temp_factory_visits_route_index

    def solution_to_temp(self):
        self.temp_routes = self.routes.copy()
        self.temp_e = self.e.copy()
        self.temp_l = self.l.copy()
        self.temp_factory_visits = self.factory_visits.copy()
        self.temp_factory_visits_route_index = self.factory_visits_route_index.copy()

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

        # TODO: If inserted "removes" a destination factory, extra time must be added

        # increase the index of all succeeding factory nodes in factory_visits_route_index
        for factory in self.prbl.factory_nodes:
            for i, v in enumerate(self.factory_visits[factory]):
                if v == vessel and idx <= self.factory_visits_route_index[factory][i]:
                    self.factory_visits_route_index[factory][i] += 1

        # Update factory related stuff
        if insert_node.is_factory:
            i = self._get_factory_visit_insert_idx(node_id, vessel, earliest)
            self.factory_visits[node_id].insert(i, vessel)
            self.factory_visits_route_index[node_id].insert(i, idx)

            # TODO: Update factory e and l

        print("insert:", node_id, "on", vessel, "at idx", idx, "  ->", route)
        print(list(zip(self.e[vessel], self.l[vessel])))
        print(self.factory_visits)
        print(self.factory_visits_route_index)
        print()

    def check_insertion_feasibility(self, insert_node: str, vessel: str, idx: int) -> bool:
        # [x] check that the vessel load capacity is not violated
        # [x] check that the vessel's max n.o. products is not violated
        # [x] check that the time windows of visited nodes are not violated
        # [/] check that the route does, or has time to end the route in a factory (with destination capacity)
        # [ ] check that precedence/wait constraints are not violated
        # [ ] check max #vessels simultaneously at factory
        # [ ] check factory destination max #vessels
        # [ ] check production capacity (PPFC)
        node = self.prbl.nodes[insert_node]
        all_checks = [self.check_load_feasibility(node, vessel, idx),
                      self.check_no_products_feasibility(node, vessel, idx),
                      self.check_time_feasibility(insert_node, vessel, idx)]
        checks_names = ['vessel_load',
                        'vessel_no_products',
                        'time']
        if not all_checks:
            print("Failed:", [name for name, check in zip(checks_names, all_checks) if not check])
        return all(all_checks)

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

    def get_earliest(self, idx: int, vessel: str, node_id: str = None, prev_node_id: str = None, prev_e: int = None) -> int:
        # if node_id is given: Calculate for insertion of node_id at idx. Else, calculate for the current node at idx
        # use prev_node_id and prev_e argument if it is given
        route = self.routes[vessel]
        node = self.prbl.nodes[node_id] if node_id else self.prbl.nodes[route[idx]]

        if not prev_node_id and not prev_e:
            prev_node_id = route[idx - 1] if idx > 0 else None
            prev_e = self.e[vessel][idx - 1] if prev_node_id else self.prbl.start_times_for_vessels[vessel]

        prev_transport_time = self.prbl.transport_times[prev_node_id, node.id] if prev_node_id else 0
        prev_loading_unloading_time = self.prbl.loading_unloading_times[vessel, prev_node_id] if prev_node_id else 0
        earliest = max(node.tw_start, prev_e + prev_loading_unloading_time + prev_transport_time)
        return earliest

    def get_latest(self, idx: int, vessel: str, node_id: str = None, next_node_id: str = None, next_l: int = None) -> int:
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
            time_to_factory = min(self.prbl.transport_times[node.id, f] for f in self.prbl.factory_nodes)  # TODO: If capacity
            latest = min(latest, len(self.prbl.time_periods) - time_to_factory - loading_unloading_time - 1)
        return latest

    def get_factory_earliest(self, factory: str, idx: int, insert_vessel: str = None, insert_e: int = None,
                             route_idx: int = None, prev_e: int = None, previous_departure_times: List[int] = None,
                             destination_removal_check: bool = False) -> Tuple[int, List[int]]:
        """
        :param factory:
        :param idx: The index of the visit in self.factory_visits
        :param insert_vessel: The vessel to be inserted at idx (optional)
        :param insert_e: The earliest visit time for the inserted node, based on route constraints (optional)
        :param route_idx: The inserted factory's route index (optional)
        :param prev_e: The earliest time for the factory's previous visit, based on factory constraints (optional)
        :param previous_departure_times: The previous visits' earliest departure times
        :return: The earliest visit time based on factory constraints
                 (route constraints are assumed to be already calculated)
        """
        quay_cap_incr_times = self.prbl.quay_cap_incr_times[factory] + [int_inf]
        visits = self.factory_visits[factory]
        visit_indices = self.factory_visits_route_index[factory]
        vessel = insert_vessel if insert_vessel else visits[idx]
        # destination factory -> zero loading time
        loading_times = [self.prbl.loading_unloading_times[visits[i], factory] *
                         (not self._is_destination_factory(visits[i], visit_indices[i]))
                         for i in range(len(visits))]
        curr_is_destination = (route_idx == len(self.routes[vessel]) if route_idx
                               else self._is_destination_factory(vessel, visit_indices[idx]))
        curr_loading_time = self.prbl.loading_unloading_times[vessel, factory] * (not curr_is_destination or
                                                                                  destination_removal_check)
        if not prev_e:
            prev_e = self.e[visits[idx - 1]][visit_indices[idx - 1]] if idx > 0 else -1
        if not previous_departure_times:
            previous_departure_times = [self.e[visits[i]][visit_indices[i]] + loading_times[i] - 1
                                        for i in range(0, idx)] + [int_inf]
            previous_departure_times.sort()

        idx_e = insert_e if insert_e else self.e[vessel][visit_indices[idx]]  # earliest from route
        t = max(prev_e, idx_e)

        # update event lists
        previous_departure_times.sort()
        previous_departure_times = self._ordered_list_min_threshold(previous_departure_times, t)
        quay_cap_incr_times = self._ordered_list_min_threshold(quay_cap_incr_times, t)

        # both quay capacity and the departures of other vessels limit insertion
        # -> iterate to find first possible t where quay_vessels < min quay_capacity over loading interval
        quay_vessels = len(previous_departure_times) - 1

        while curr_loading_time > 0 and quay_vessels >= self._get_min_quay_capacity(factory, t, curr_loading_time):
            if previous_departure_times[0] < quay_cap_incr_times[0]:
                t = previous_departure_times.pop(0) + 1
                quay_vessels -= 1
            else:
                t = quay_cap_incr_times.pop(0)

        if curr_loading_time > 0:
            bisect.insort(previous_departure_times, t + curr_loading_time - 1)
        return t, previous_departure_times

    def get_factory_latest(self, factory: str, idx: int, insert_vessel: str = None,
                           insert_l: int = None, route_idx: int = None, next_l: int = None,
                           next_arrival_times: List[int] = None) -> Tuple[int, List[int]]:
        """
        :param factory:
        :param idx: The index of the visit in self.factory_visits
        :param insert_vessel: The vessel to be inserted at idx (optional)
        :param insert_l: The latest visit time for the inserted node, based on route constraints (optional)
        :param route_idx: The inserted factory's route index (optional)
        :param next_l: The latest time for the factory's next visit, based on factory constraints (optional)
        :param next_arrival_times: the succeeding visits' latest arrival times
        :return: The latest visit time based on factory constraints
        """
        quay_cap_decr_times = [-int_inf] + self.prbl.quay_cap_decr_times[factory]
        visits = self.factory_visits[factory]
        visit_indices = self.factory_visits_route_index[factory]
        vessel = insert_vessel if insert_vessel else visits[idx]
        # destination factory -> zero loading time
        curr_is_destination = (route_idx == len(self.routes[vessel]) if route_idx
                               else self._is_destination_factory(vessel, visit_indices[idx]))
        curr_loading_time = self.prbl.loading_unloading_times[vessel, factory] * (not curr_is_destination)
        if not next_l:
            next_l = self.l[visits[idx + 1]][visit_indices[idx + 1]] if idx + 1 < len(visits) else int_inf
        if not next_arrival_times:
            next_arrival_times = [-int_inf] + [self.l[visits[i]][visit_indices[i]] for i in range(idx, len(visits))]
            next_arrival_times.sort()

        idx_l = insert_l if insert_l else self.l[vessel][visit_indices[idx]]  # latest from route
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

    def check_earliest_forward(self, vessel: str, idx: int, prev_node_id: str, prev_e: int) -> bool:
        """Iteratively checks earliest time for succeeding visits until no change"""
        route = self.routes[vessel]
        for i in range(idx, len(route)):
            prev_e = self.get_earliest(i, vessel, prev_node_id=prev_node_id, prev_e=prev_e)
            self.temp_e[vessel][i] = max(prev_e, self.temp_e[vessel][i])  # update temp_e if stronger bound is found

            if self.l[vessel][i] < prev_e:  # insertion squeezes out succeeding node
                if self.verbose:
                    print(f"Check failed at: check_earliest_forward for {vessel}. "
                          f"Forward from route index {idx} at {i}: ({prev_e}, {self.l[vessel][i]})")
                return False

            if self.e[vessel][i] == prev_e:  # propagation of check stops if e is unchanged
                break

            node = self.prbl.nodes[route[i]]
            if node.is_factory:  # check routes visiting the same factory before than this visit
                factory_visit_idx = self._get_factory_visit_idx(node.id, vessel, i)
                if not self.check_factory_visits_earliest_forward(node.id, factory_visit_idx, prev_e):
                    return False  # the changed e for the factory resulted in a node on another route being squeezed out

            prev_node_id = route[i]
        return True

    def check_latest_backward(self, vessel: str, idx: int, next_node_id: str, next_l: int) -> bool:
        """Iteratively checks latest time for preceding visits until no change"""
        route = self.routes[vessel]
        for i in range(idx - 1, -1, -1):
            next_l = self.get_latest(i, vessel, next_node_id=next_node_id, next_l=next_l)
            self.temp_l[vessel][i] = min(next_l, self.temp_l[vessel][i])  # update temp_l if stronger bound is found

            if next_l < self.e[vessel][i]:  # insertion squeezes out preceding node
                if self.verbose:
                    print(f"Check failed at: check_latest_backward for {vessel}. "
                          f"Backward from route index {idx} at {i}: ({self.e[vessel][i]}, {next_l})")
                return False

            if self.l[vessel][i] == next_l:  # propagation of check stops if l is unchanged
                break

            node = self.prbl.nodes[route[i]]
            if node.is_factory:  # check routes visiting the same factory after than this visit
                factory_visit_idx = self._get_factory_visit_idx(node.id, vessel, i)
                if not self.check_factory_visits_latest_backward(node.id, factory_visit_idx, next_l):
                    return False  # the changed l for the factory resulted in a node on another route being squeezed out

            next_node_id = route[i]
        return True

    def check_factory_visits_earliest_forward(self, factory: str, idx: int, e: int,
                                              prev_dep_times: List[int] = None) -> bool:
        """
        Iteratively checks a factory visits' earliest arrival times for succeeding vessels until violation or no overlap
        :param factory:
        :param idx: the factory visit index to check from
        :param e: e for the visit at idx
        :param prev_dep_times:
        :return: True if there is no constraint violation
        """
        for i in range(idx, len(self.factory_visits[factory])):
            vessel = self.factory_visits[factory][i]
            route_index = self.factory_visits_route_index[factory][i]
            node_id = self.routes[vessel][route_index]

            e, prev_dep_times = self.get_factory_earliest(factory, i, prev_e=e, previous_departure_times=prev_dep_times)
            self.temp_e[vessel][route_index] = max(e, self.temp_e[vessel][route_index])

            if self.l[vessel][route_index] < e:
                if self.verbose:
                    print(f"Check failed at: check_factory_visits_earliest_forward for {vessel}. "
                          f"Forward from factory visit {idx} at {i}: : ({e}, {self.l[vessel][i]})")
                return False  # not enough time space for insertion

            if max(prev_dep_times[:-1], default=-int_inf) < self.e[vessel][route_index]:
                break  # stop propagation if no overlap between visit i and previous departures

            if self.e[vessel][route_index] < e and not self.check_earliest_forward(vessel, route_index, node_id, e):
                return False
        return True

    def check_factory_visits_latest_backward(self, factory: str, idx: int, l: int, next_arr_times: List[int] = None) -> bool:
        """
        Iteratively checks a factory visits' latest arrival times for preceding vessels until violation or no overlap
        :param factory:
        :param idx: the factory visit index to check from
        :param l: l for the visit at idx
        :param next_arr_times:
        :return: True if there is no constraint violation
        """
        for i in range(idx - 1, -1, -1):
            vessel = self.factory_visits[factory][i]
            route_index = self.factory_visits_route_index[factory][i]
            node_id = self.routes[vessel][route_index]
            i_latest_departure = self.l[vessel][route_index] + self.prbl.loading_unloading_times[vessel, factory]

            l, next_arr_times = self.get_factory_latest(factory, i, next_l=l,
                                                        next_arrival_times=next_arr_times)
            self.temp_l[vessel][route_index] = min(l, self.temp_l[vessel][route_index])

            if l < self.e[vessel][route_index]:
                if self.verbose:
                    print(f"Check failed at: check_factory_visits_latest_backward for {vessel}. "
                          f"Backward from factory visit {idx} at {i}: ({self.e[vessel][i]}, {l})")
                return False  # not enough time space for insertion

            if i_latest_departure < min(next_arr_times[-1:], default=int_inf):
                break  # stop propagation if no overlap between visit i and next arrivals

            if self.l[vessel][route_index] > l and not self.check_latest_backward(vessel, route_index, node_id, l):
                return False

        return True

    def check_time_feasibility(self, insert_node_id: str, vessel: str, idx: int) -> bool:
        route = self.routes[vessel]
        insert_node = self.prbl.nodes[insert_node_id]
        idx = len(route) + idx + 1 if idx < 0 else idx  # transform negative indexes
        e = self.get_earliest(idx, vessel, insert_node_id)
        l = self.get_latest(idx, vessel, insert_node_id)

        if l < e:  # not enough time space for insertion
            print(f"Check failed at: check_time_feasibility for {vessel}: ({e}, {l})")
            return False

        if not self.check_latest_backward(vessel, idx, insert_node_id, l):
            return False

        if not self.check_earliest_forward(vessel, idx, insert_node_id, e):
            return False

        # TODO
        #  if the node is inserted after what was previously a factory destination node, the earliest arrivals of
        #  succeeding factory visits for this factory destination node must be updated
        # last_route_node = self.prbl.nodes[route[-1]]
        # if idx == len(route) and last_route_node.is_factory:
        #     factory_visit_idx = self._get_factory_visit_idx(last_route_node.id, vessel, len(route) - 1)
        #     self.check_factory_visits_earliest_forward(last_route_node.id, factory_visit_idx, self.e[vessel][idx - 2],
        #                                                destination_removal_check=True)

        self.temp_routes[vessel].insert(idx, insert_node_id)
        self.temp_e[vessel].insert(idx, e)
        self.temp_l[vessel].insert(idx, l)

        if insert_node.is_factory:
            factory_visit_idx = self._get_factory_visit_insert_idx(insert_node_id, vessel, e)
            e, prev_dep_times = self.get_factory_earliest(insert_node_id, factory_visit_idx, vessel, e, idx)
            l, next_arr_times = self.get_factory_latest(insert_node_id, factory_visit_idx, vessel, l, idx)
            if l < e:
                return False

            if not self.check_factory_visits_earliest_forward(insert_node_id, factory_visit_idx, e, prev_dep_times):
                return False

            if not self.check_factory_visits_latest_backward(insert_node_id, factory_visit_idx, l, next_arr_times):
                return False

            self.temp_factory_visits[insert_node_id].insert(factory_visit_idx, vessel)
            self.temp_factory_visits_route_index[insert_node_id].insert(factory_visit_idx, idx)
            self.temp_e[vessel][idx] = max(e, self.temp_e[vessel][idx])
            self.temp_l[vessel][idx] = min(l, self.temp_l[vessel][idx])

        return True

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

    def get_insertion_utility(self, node: Node, route: str, idx: int) -> float:
        # calculate the change in objective value if node is inserted
        pass

    def get_vessel_factories(self, vessel: str) -> List[str]:
        return [node_id for node_id in self.routes[vessel] if self.prbl.nodes[node_id].is_factory]

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

    def _init_factory_visits(self) -> Dict[str, List[str]]:
        vessel_starting_times = list(self.prbl.start_times_for_vessels.items())
        vessel_starting_times.sort(key=lambda item: item[1])
        factory_visits:  Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}

        for vessel, _ in vessel_starting_times:
            initial_factory = self.prbl.vessel_initial_locations[vessel]
            factory_visits[initial_factory].append(vessel)
        return factory_visits

    def _init_factory_visits_e_and_l(self) -> None:
        for factory in self.prbl.factory_nodes:
            if len(self.factory_visits[factory]):
                last_idx = len(self.factory_visits[factory]) - 1
                first_visit = self.factory_visits[factory][0]
                last_visit = self.factory_visits[factory][-1]

                e, prev_dep_times = self.get_factory_earliest(factory, 0, self.factory_visits[factory][0])
                l, next_arr_times = self.get_factory_latest(factory, last_idx, self.factory_visits[factory][-1])
                self.temp_e[first_visit][0] = e
                self.temp_l[last_visit][0] = l
                if (not self.check_factory_visits_earliest_forward(factory, 0, e, prev_dep_times)  # TODO
                        or not self.check_factory_visits_latest_backward(factory, last_idx, l, next_arr_times)):
                    raise AssertionError("Infeasible initial factory visits")
        self.temp_to_solution()

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

    def _get_factory_visit_insert_idx(self, factory: str, vessel: str, earliest: int):
        visits = self.factory_visits[factory]
        visit_indices = self.factory_visits_route_index[factory]
        # insertion where all previous factory visits have e < earliest
        i = bisect.bisect([self.e[v][visit_indices[i]] for i, v in enumerate(visits)], earliest)
        return i

    def _get_factory_visit_idx(self, factory: str, vessel: str, route_idx: int) -> int:
        # route_idx = len(self.routes[vessel]) - route_idx if route_idx < 0 else route_idx
        for i, (v, idx) in enumerate(zip(self.factory_visits[factory], self.factory_visits_route_index[factory])):
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

    def print_routes(self):
        for vessel, route in self.routes.items():
            s = ''
            for node, e, l in zip(self.routes[vessel], self.e[vessel], self.l[vessel]):
                s += f'{node} ({e},{l}), '
            print(f'{vessel}: {s}')


# TESTING
# problem = InternalProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
# problem = InternalProblemData('../../data/input_data/medium_testcase.xlsx')
problem = InternalProblemData('../../data/input_data/large_testcase.xlsx')
for node in problem.nodes.values():
    print(node)
sol = Solution(problem)
print(sol.routes)

# insertions = [('o_1', 'v_1', 1),  # medium testcase
#               ('o_4', 'v_1', 2),
#               ('f_1', 'v_1', 2),
#               ('o_2', 'v_2', 1),
#               ('f_2', 'v_2', 2),
#               # ('o_6', 'v_3', 1),
#               # ('f_1', 'v_3', 1),
#               ('f_2', 'v_3', 1)]

insertions = [  # large testcase
    ('o_1', 'v_1', 1),
    ('o_4', 'v_1', 2),
    ('f_1', 'v_1', 2),
    ('o_2', 'v_2', 1),
    ('f_1', 'v_2', 2),
    ('o_6', 'v_3', 1),
    ('f_1', 'v_2', 1),
    ('o_1', 'v_2', 1),
    ('f_2', 'v_3', 2),
]

for node, vessel, idx in insertions:
    print(f'Inserting {node} into {vessel} at {idx}.')
    sol.check_insertion_feasibility(node, vessel, idx)
    sol.temp_to_solution()
    sol.print_routes()
    print()


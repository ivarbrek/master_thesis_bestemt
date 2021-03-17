from typing import List, Dict, Tuple
from src.read_problem_data import ProblemData
import math
import numpy as np
import bisect
import copy
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

        self.temp_routes: Dict[str, List[str]] = copy.deepcopy(self.routes)
        self.temp_e: Dict[str, List[int]] = copy.deepcopy(self.e)
        self.temp_l: Dict[str, List[int]] = copy.deepcopy(self.l)
        self.temp_factory_visits: Dict[str, List[str]] = copy.deepcopy(self.factory_visits)
        self.temp_factory_visits_route_index: Dict[str, List[int]] = copy.deepcopy(self.factory_visits_route_index)

        # self._init_factory_visits_e_and_l()
        self.verbose = True

    def temp_to_solution(self):
        self.routes = copy.deepcopy(self.temp_routes)
        self.e = copy.deepcopy(self.temp_e)
        self.l = copy.deepcopy(self.temp_l.copy())
        self.factory_visits = copy.deepcopy(self.temp_factory_visits.copy())
        self.factory_visits_route_index = copy.deepcopy(self.temp_factory_visits_route_index.copy())

    def solution_to_temp(self):
        self.temp_routes = copy.deepcopy(self.routes)
        self.temp_e = copy.deepcopy(self.e)
        self.temp_l = copy.deepcopy(self.l)
        self.temp_factory_visits = copy.deepcopy(self.factory_visits)
        self.temp_factory_visits_route_index = copy.deepcopy(self.factory_visits_route_index)


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

    def get_earliest(self, idx: int, vessel: str) -> int:
        route = self.temp_routes[vessel]
        node = self.prbl.nodes[route[idx]]

        prev_node_id = route[idx - 1] if idx > 0 else None
        prev_e = self.temp_e[vessel][idx - 1] if prev_node_id else self.prbl.start_times_for_vessels[vessel]

        prev_transport_time = self.prbl.transport_times[prev_node_id, node.id] if prev_node_id else 0
        prev_loading_unloading_time = self.prbl.loading_unloading_times[vessel, prev_node_id] if prev_node_id else 0
        earliest = max(node.tw_start, prev_e + prev_loading_unloading_time + prev_transport_time)
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
            time_to_factory = min(self.prbl.transport_times[node.id, f] for f in self.prbl.factory_nodes)  # TODO: If capacity
            latest = min(latest, len(self.prbl.time_periods) - time_to_factory - loading_unloading_time - 1)
        return latest

    def get_factory_earliest(self, factory: str, idx: int,
                             prev_departure_times: List[int] = None) -> Tuple[int, List[int]]:
        """
        :param factory:
        :param idx: The index of the visit in self.factory_visits
        :param insert_vessel: The vessel to be inserted at idx (optional)
        :param insert_e: The earliest visit time for the inserted node, based on route constraints (optional)
        :param route_idx: The inserted factory's route index (optional)
        :param prev_e: The earliest time for the factory's previous visit, based on factory constraints (optional)
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
        :param insert_vessel: The vessel to be inserted at idx (optional)
        :param insert_l: The latest visit time for the inserted node, based on route constraints (optional)
        :param route_idx: The inserted factory's route index (optional)
        :param next_l: The latest time for the factory's next visit, based on factory constraints (optional)
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

    def check_earliest_forward(self, vessel: str, idx: int) -> bool:
        """Iteratively checks earliest time for succeeding visits until no change"""
        route = self.temp_routes[vessel]
        for i in range(idx, len(route)):
            e = self.get_earliest(i, vessel)
            updated_e = e > self.temp_e[vessel][i]
            self.temp_e[vessel][i] = e if updated_e else self.temp_e[vessel][i]  # update temp_e if stronger bound is found

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

            if not updated_e:  # propagation of check stops if e is unchanged
                break

        return True

    def check_latest_backward(self, vessel: str, idx: int) -> bool:
        """Iteratively checks latest time for preceding visits until no change"""
        route = self.temp_routes[vessel]
        for i in range(idx - 1, -1, -1):
            l = self.get_latest(i, vessel)
            updated_l = l < self.temp_l[vessel][i]
            self.temp_l[vessel][i] = l if updated_l else self.temp_l[vessel][i]  # update temp_l if stronger bound is found

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

            if not updated_l:  # propagation of check stops if l is unchanged
                break

        return True

    def check_factory_visits_earliest_forward(self, factory: str, idx: int, prev_dep_times: List[int] = None) -> bool:
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
                    print(f"Check failed at: check_factory_visits_earliest_forward for {factory }. "
                          f"Forward from factory visit {idx} at {i}: : ({e}, {self.temp_l[vessel][i]})")
                return False  # not enough time space for insertion

            if updated_e:
                self.temp_e[vessel][route_index] = e

                if not self.check_earliest_forward(vessel, route_index + 1):
                    return False

            if max(prev_dep_times[:-1], default=-int_inf) < self.temp_e[vessel][route_index]:
                break  # stop propagation if no overlap between visit i and previous departures
        return True

    def check_factory_visits_latest_backward(self, factory: str, idx: int, next_arr_times: List[int] = None) -> bool:
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
                          f"Backward from factory visit {idx} at {i}: ({self.e[vessel][i]}, {l})")
                return False  # not enough time space for insertion

            if updated_l:
                self.temp_l[vessel][route_index] = l

                if not self.check_latest_backward(vessel, route_index):
                    return False

            if self.temp_l[vessel][route_index] < min(next_arr_times[-1:], default=int_inf):
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
            self.check_factory_visits_earliest_forward(route[-2], 1)

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

    def _init_factory_visits(self) -> Dict[str, List[str]]:
        vessel_starting_times = list(self.prbl.start_times_for_vessels.items())
        vessel_starting_times.sort(key=lambda item: item[1])
        factory_visits:  Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}

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
        for i, (v, idx) in enumerate(zip(self.temp_factory_visits[factory], self.temp_factory_visits_route_index[factory])):
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

    def print_routes(self, highlight: List[Tuple[str, int]] = None):
        highlight = [] if not highlight else highlight
        for vessel, route in self.routes.items():
            s = ''
            for i, (node, e, l) in enumerate(zip(self.routes[vessel], self.e[vessel], self.l[vessel])):
                if (vessel, i) in highlight:
                    s += f'{bcolors.OKGREEN}{node} ({e},{l}){bcolors.RESET_ALL}, '
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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    GREY = '\033[37m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    BRIGHT = '\033[1m'
    RESET_ALL = '\033[0m'

# TESTING
# problem = InternalProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
# problem = InternalProblemData('../../data/input_data/medium_testcase.xlsx')
problem = InternalProblemData('../../data/input_data/large_testcase.xlsx')
# for node in problem.nodes.values():
#     print(node)
sol = Solution(problem)


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
    if node == 'o_1':
        a = 1
    if sol.check_insertion_feasibility(node, vessel, idx):
        sol.temp_to_solution()
    else:
        sol.solution_to_temp()
    # print(sol.factory_visits)
    # print(sol.factory_visits_route_index)
    sol.print_routes(highlight=[(vessel, idx)])
    print()
    sol.print_factory_visits(highlight=[(vessel, idx)])
    print("\n\n")



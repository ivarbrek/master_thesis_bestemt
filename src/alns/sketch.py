from typing import List, Dict, Tuple
from src.read_problem_data import ProblemData
import math
import numpy as np


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


class Solution:

    def __init__(self, prbl: InternalProblemData) -> None:
        self.prbl = prbl
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

        print(node_id)
        print(route)
        print(list(zip(self.e[vessel], self.l[vessel])))
        # print(self.factory_visits)
        # print(self.factory_visits_route_index)
        print()

    def check_insertion_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        # [x] check that the vessel load capacity is not violated
        # [x] check that the vessel's max n.o. products is not violated
        # [x] check that the time windows of visited nodes are not violated
        # [/] check that the route does, or has time to end the route in a factory (with destination capacity)
        # [ ] check that precedence/wait constraints are not violated
        # [ ] check max #vessels simultaneously at factory
        # [ ] check factory destination max #vessels
        # [ ] check production capacity (PPFC)
        return all([self.check_load_feasibility(insert_node, vessel, idx),
                    self.check_no_products_feasibility(insert_node, vessel, idx)])

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
        # TODO
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

    def init_factory_visits(self) -> Dict[str, List[str]]:
        starting_times = list(self.prbl.start_times_for_vessels.items())
        starting_times.sort(key=lambda item: item[1])
        factory_visits:  Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}
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


# TESTING
problem = InternalProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
for node in problem.nodes.values():
    print(node)
sol = Solution(problem)
print(sol.routes)
print(list(zip(sol.e['v_1'], sol.l['v_1'])))
# problem.vessel_ton_capacities['v_1'] = 89
# problem.vessel_nprod_capacities['v_1'] = 3
print(sol.check_time_feasibility('o_1', 'v_1', 1))
sol.insert_node('o_1', 'v_1', 1)
print(sol.check_time_feasibility('o_2', 'v_1', 1))
sol.insert_node('o_2', 'v_1', 1)
print(sol.check_time_feasibility('o_3', 'v_1', 1))
sol.insert_node('o_3', 'v_1', 1)
print(sol.check_time_feasibility('o_2', 'v_1', 1))
sol.insert_node('o_2', 'v_1', 1)
print(sol.check_time_feasibility('o_3', 'v_1', 1))
sol.insert_node('o_3', 'v_1', 1)
print(sol.check_time_feasibility('f_1', 'v_1', -1))
sol.insert_node('f_1', 'v_1', -1)
for i in range(1, 5):
    print(i, sol.check_time_feasibility('o_1', 'v_1', i))
sol.insert_node('o_1', 'v_1', 2)
print(sol.routes)
# print(sol.check_no_products_feasibility(problem.nodes['o_1'], 'v_1', 1))
# print(sol.check_load_feasibility(problem.nodes['o_1'], 'v_1', 1))
# print(sol.get_nth_visit_idx(2, 'v_1', 'f_2'))
# print(sol.factory_visits_route_index)



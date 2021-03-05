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
            self.factory_nodes[i] = Node(i, is_factory=True)
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
        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.e: Dict[str, List[int]] = {v: [self.prbl.start_times_for_vessels[v]] for v in self.prbl.vessels}
        self.l: Dict[str, List[int]] = {v: [math.inf] for v in self.prbl.vessels}
        # TODO: Update e and l due to factory visits

    def insert_node(self, node_id: str, vessel: str, idx: int) -> None:
        insert_node = self.prbl.nodes[node_id]
        route = self.routes[vessel]
        if idx == -1:
            route.append(node_id)
        else:
            route.insert(idx, node_id)
        # prev_node = route[idx - 1]
        # next_node = route[idx + 1]
        # if insert_node.is_factory:
        #     # TODO:
        #     pass
        # else:
        #     self.e_order[node_id] = max(node.tw_start, self.get_e(prev_node, vessel, ))
        # Update e and l
        # Update factory related stuff: Factory visits, e and l for factories

    def check_insertion_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        # [x] check that the vessel load capacity is not violated
        # [x] check that the vessel's max n.o. products is not violated
        # [ ] copy route (and nodes in it), and calculate e(i) and l(i)
        # [ ] check that the time windows of visited nodes are not violated
        # [ ] check that the route does, or has time to end the route in a factory (with destination capacity)
        # [ ] check that precedence/wait constraints are not violated
        # [ ] check max #vessels simultaneously at factory
        # [ ] check factory destination max #vessels
        # [ ] check production capacity (PPFC)
        return all([self.check_load_feasibility(insert_node, vessel, idx),
                    self.check_no_products_feasibility(insert_node, vessel, idx)])

    def check_time_feasibility(self, insert_node: Node, vessel: str, idx: int) -> bool:
        route = self.routes[vessel]
        prev_node_id = route[idx - 1]
        next_node_id = route[idx + 1]
        time_between = (self.prbl.loading_unloading_times[prev_node_id]
                        + self.prbl.transport_times[prev_node_id, insert_node.id]
                        + self.prbl.loading_unloading_times[insert_node.id]
                        + self.prbl.transport_times[insert_node.id, next_node_id])
        return self.e[vessel][idx - 1] + time_between - 1 <= self.l[vessel][idx + 1]
        # TODO: Lianes and Noreng is strange here

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
        voyage_end_idx = -1
        for i in range(idx - 1, -1, -1):
            if self.prbl.nodes[route[i]].is_factory:
                voyage_start_idx = i
                break
        for i in range(idx, len(route) + 1):
            if self.prbl.nodes[route[i]].is_factory:
                voyage_end_idx = i
                break
        assert voyage_start_idx != -1 and voyage_end_idx != -1, "Illegal voyage, not enclosed by factories"
        return voyage_start_idx, voyage_end_idx

    def init_factory_visits(self) -> Dict[str, List[str]]:
        starting_times = list(self.prbl.start_times_for_vessels.items())
        starting_times.sort(key=lambda item: item[1])
        factory_visits:  Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}
        for vessel, _ in starting_times:
            start_location = self.prbl.vessel_initial_locations[vessel]
            factory_visits[start_location].append(vessel)
        return factory_visits


# TESTING
problem = InternalProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
for node in problem.nodes.values():
    print(node)
sol = Solution(problem)
print(sol.routes)
problem.vessel_ton_capacities['v_1'] = 89
problem.vessel_nprod_capacities['v_1'] = 2
sol.insert_node('o_3', 'v_1', 1)
sol.insert_node('o_2', 'v_1', 1)
# sol.insert_node('o_2', 'v_1', 1)
sol.insert_node('f_2', 'v_1', -1)
print(sol.routes)
print(sol.check_no_products_feasibility(problem.nodes['o_1'], 'v_1', 1))
print(sol.check_load_feasibility(problem.nodes['o_1'], 'v_1', 1))






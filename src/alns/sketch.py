from typing import List, Dict
from src.read_problem_data import ProblemData
import math


class Node:
    id: str
    tw_start: int = -1
    tw_end: int = math.inf
    demand: List[int] = []

    def __init__(self, name):
        self.id = name

    def __repr__(self):
        return f"{self.id}: ({self.tw_start}, {self.tw_end}), {self.demand}"


class InternalProblemData(ProblemData):

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self._init_nodes()

    def _init_nodes(self):
        f_nodes = self.factory_nodes[:]
        o_nodes = self.order_nodes[:]
        self.factory_nodes = {}
        self.order_nodes = {}
        for i in f_nodes:
            self.factory_nodes[i] = Node(i)
        for i in o_nodes:
            node = Node(i)
            node.tw_start = self.tw_start[i]
            node.tw_end = self.tw_end[i]
            node.demand = [self.demands[i, i, p] for p in self.products]
            self.order_nodes[i] = node
        self.nodes: Dict = {**self.factory_nodes, **self.order_nodes}


class Solution:

    def __init__(self, prbl: InternalProblemData) -> None:
        self.prbl = prbl
        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.factory_visits: Dict[str, List[str]] = self.init_factory_visits()
        self.e: Dict[str, int] = {i: -1 for i in self.prbl.nodes}  # earliest visits
        self.l: Dict[str, int] = {i: math.inf for i in self.prbl.nodes}  # latest visits
        self.factory_e: Dict[str, int] = {i: -1 for i in self.prbl.nodes}    # earliest visit due to quay capacity
        self.factory_l: Dict[str, int] = {i: math.inf for i in self.prbl.nodes}  # latest due to quay capacity

    def init_factory_visits(self):
        # TODO
        pass

    def insert_node(self, node_id: str, route: str, i: int) -> None:
        self.routes[route].insert(i, node_id)

    def check_insertion_feasibility(self, node: Node, route: str, i: int) -> bool:
        # check that the vessel load capacity is not violated
        # check that the vessel's max n.o. products is not violated
        # copy route (and nodes in it), and calculate e(i) and l(i)
        # check that the time windows of visited nodes are not violated
        # check that precedence/wait constraints are not violated
        # check that the route does, or has time to end the route in a factory (with destination capacity)
        # check max #vessels simultaneously at factory
        # check factory destination max #vessels
        # check production capacity (PPFC)
        pass

    def check_load_feasibility(self, node: Node, route: str, i: int) -> bool:
        # load_capacity = self.prbl.vessel_ton_capacities[]

        return True

    def get_insertion_utility(self, node: Node, route: str, i: int) -> float:
        # calculate the change in objective value if node is inserted
        pass


# TESTING
problem = InternalProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
for node in problem.nodes.values():
    print(node)
sol = Solution(problem)
print(sol.routes)
sol.insert_node('o_1', 'v_1', 1)
sol.check_load_feasibility(problem.nodes['o_1'], 'v_1', 1)
print(sol.routes)






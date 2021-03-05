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
        self.id = name
        self.is_factory = is_factory

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
            self.factory_nodes[i] = Node(i, is_factory=True)
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
        self.factory_visits: Dict[str, List[str]] = self.init_factory_visits()
        self.routes: Dict[str, List[str]] = {v: [self.prbl.vessel_initial_locations[v]] for v in self.prbl.vessels}
        self.e: Dict[str, List[int]] = {v: [self.prbl.start_times_for_vessels[v]] for v in self.prbl.vessels}
        self.l: Dict[str, List[int]] = {v: [math.inf] for v in self.prbl.vessels}
        # TODO: Update e and l due to factory visits

    def init_factory_visits(self) -> Dict[str, List[str]]:
        starting_times = list(self.prbl.start_times_for_vessels.items())
        starting_times.sort(key=lambda item: item[1])
        factory_visits: Dict[str, List[str]] = {i: [] for i in self.prbl.factory_nodes}
        for vessel, _ in starting_times:
            start_location = self.prbl.vessel_initial_locations[vessel]
            factory_visits[start_location].append(vessel)
        return factory_visits

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

    def get_voyage_start_idxs_for_vessels(self, factory_node_id: str) -> Dict[
        str, List[Tuple[int, int]]]:  # Tuple (index in route, latest loading start time)
        voyage_start_idxs_for_vessels: Dict[str, List[Tuple[int, int]]] = {}
        for v in self.prbl.vessels:
            voyage_start_idxs_for_vessels[v] = []
            route = self.routes[v]
            for i in range(len(route) - 1):  # Last factory visit is not a voyage start
                if self.prbl.nodes[route[i]].is_factory and self.prbl.nodes[route[i]].id == factory_node_id:
                    voyage_start_idxs_for_vessels[v].append(tuple((i, self.l[v][i])))
        return voyage_start_idxs_for_vessels

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

    def check_load_feasibility(self, node: Node, vessel: str, idx: int) -> bool:
        route = self.routes[vessel]
        trip_start_idx = max([i for i, node_id in enumerate(route) if i < idx and self.prbl.nodes[node_id].is_factory])
        trip_end_idx = min([i for i, node_id in enumerate(route) if i > idx and self.prbl.nodes[node_id].is_factory])
        trip_demand = sum(d for node_id in route[trip_start_idx:trip_end_idx] for d in self.prbl.nodes[node_id].demand)
        return trip_demand + sum(d for d in node.demand) <= self.prbl.vessel_ton_capacities[vessel]

    def get_insertion_utility(self, node: Node, route: str, i: int) -> float:
        # calculate the change in objective value if node is inserted
        pass

    def check_production_feasibility(self, factory_node_id: str) -> bool:
        # What to find:

        # Find the voyages originating from a factory node; tuple(index in route, latest loading start time)
        voyage_start_idxs: Dict[str, List[Tuple[int, int]]] = self.get_voyage_start_idxs_for_vessels(factory_node_id)

        # Collect quantity to be delivered for each voyage
        products_for_voyage = np.empty([0, len(self.prbl.get_products())], int)
        latest_loading_times: List[int] = []
        for vessel in voyage_start_idxs.keys():
            for i in range(len(voyage_start_idxs[vessel])):
                latest_loading_times.append(voyage_start_idxs[vessel][i][1])
                if i == len(voyage_start_idxs[vessel]) - 1:
                    stop_index = len(self.routes[vessel]) - 1  # Voyage stop is the last factory to be visited
                else:
                    stop_index = voyage_start_idxs[vessel][i + 1][0]
                products_for_voyage = np.append(products_for_voyage,
                                                np.array([np.sum([[d for d in self.prbl.nodes[node_id].demand]
                                                                  for node_id in self.routes[vessel][
                                                                                 voyage_start_idxs[vessel][i][
                                                                                     0] + 1:stop_index]], axis=0)]),
                                                axis=0)

        # Matrix products_for_voyage - rows: voyages (index i), columns: products
        # List latest_loading_times - latest loading time (index i)

        # Order rows in matrix products_for_voyage by corresponding latest_loading_times asc
        idx_sorted = np.argsort(latest_loading_times)
        latest_loading_times = np.sort(latest_loading_times)
        products_for_voyage = products_for_voyage[idx_sorted]

        # Sum rows whose latest delivery time is equal (these are treated as one production order in the PP)
        same_delivery_times = [np.where(latest_loading_times == element)[0].tolist()
                               for element in np.unique(latest_loading_times)]
        for identical_times in same_delivery_times:
            if len(identical_times) > 1:  # There are rows whose loading time is equal
                products_for_voyage[identical_times[0]] = np.sum(products_for_voyage[identical_times], axis=0)
                products_for_voyage = np.delete(products_for_voyage, identical_times.pop(), axis=0)
                latest_loading_times = np.delete(latest_loading_times, identical_times)

        # Make cumulative representation
        production_requirement_cum = np.cumsum(products_for_voyage, axis=0)

        # Find the minimum number of activities that must be undertaken before a given loading event
        activity_requirement_cum = production_requirement_cum
        production_lines = [l for (i, l) in
                            filter(lambda x: x[0] == factory_node_id, self.prbl.production_lines_for_factories)]

        for p in range(np.shape(production_requirement_cum)[1]):  # for all columns
            initial_inventory = self.prbl.factory_initial_inventories[(factory_node_id, self.prbl.products[p])]
            production_capacity_max = max(
                [self.prbl.production_max_capacities[l, self.prbl.products[p]] for l in production_lines])
            for k in range(np.shape(production_requirement_cum)[0]):
                activity_requirement_cum[k][p] = max(
                    [0, np.ceil((production_requirement_cum[k, p] - initial_inventory) / production_capacity_max)])

        for k in range(len(activity_requirement_cum) - 1, 0, -1):
            production_time_periods = [len(production_lines) * sum([self.prbl.production_stops[factory_node_id, t]
                                                                    for t in range(latest_loading_times[k - 1],
                                                                                   latest_loading_times[k])])]
            while (np.sum(activity_requirement_cum[k], axis=0) - np.sum(activity_requirement_cum[k - 1], axis=0)
                   > production_time_periods):
                for p in range(np.shape(production_requirement_cum)[1]):  # TODO: Do this in a better way (more efficient, and prioritize the most beneficial p)
                    if activity_requirement_cum[k][p] > 0:
                        activity_requirement_cum[k - 1][p] += 1  # Pushing production activities to occur earlier
                    break

        # If pushing production activities earlier results in too much production taking place before the first loading,
        # then the production schedule is infeasible
        first_production_time_periods = [len(production_lines) * sum([self.prbl.production_stops[factory_node_id, t]
                                                                      for t in range(latest_loading_times[0] + 1)])]
        if first_production_time_periods < np.sum(activity_requirement_cum[0], axis=0):
            return False

        return True


# TESTING
problem = InternalProblemData('../../data/input_data/medium_testcase.xlsx')
for node in problem.nodes.values():
    print(node)
sol = Solution(problem)
# sol.check_load_feasibility(problem.nodes['o_1'], 'v_1', 1)

# Test routes
sol.insert_node('o_2', 'v_2', 1)
sol.insert_node('f_2', 'v_2', 2)

sol.insert_node('o_1', 'v_1', 1)
sol.insert_node('f_1', 'v_1', 2)
sol.insert_node('o_3', 'v_1', 3)
sol.insert_node('o_4', 'v_1', 4)
sol.insert_node('f_1', 'v_1', 5)

sol.l['v_1'] = [3, 5, 7, 9, 10, 12]
sol.l['v_2'] = [0, 10, 14]

print(sol.routes)

# Testing PPFC with f_1 production
print("Voyage starts, tuple(index, latest loading time):", sol.get_voyage_start_idxs_for_vessels('f_1'))
print(sol.check_production_feasibility('f_1'))

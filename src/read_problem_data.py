import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class ProblemData:

    def __init__(self, file_path: str, soft_tw=False) -> None:
        self.soft_tw = soft_tw

        self.vessel_capacities_df = pd.read_excel(file_path, sheet_name='vessel_capacity', index_col=0, skiprows=[0])
        self.inventory_capacities_df = pd.read_excel(file_path, sheet_name='inventory_capacity', index_col=0,
                                                     skiprows=[0])
        self.time_windows_for_orders_df = pd.read_excel(file_path, sheet_name='time_windows_for_order', index_col=0,
                                                        skiprows=[0])
        self.vessel_availability_df = pd.read_excel(file_path, sheet_name='vessel_availability', index_col=0,
                                                    skiprows=[0])
        self.nodes_for_vessels_df = pd.read_excel(file_path, sheet_name='nodes_for_vessel', index_col=0, skiprows=[0])
        self.initial_inventories_df = pd.read_excel(file_path, sheet_name='initial_inventory', index_col=0,
                                                    skiprows=[0])
        self.inventory_unit_costs_rewards_df = pd.read_excel(file_path, sheet_name='inventory_cost', index_col=0,
                                                             skiprows=[0])
        self.transport_unit_costs_df = pd.read_excel(file_path, sheet_name='transport_cost', index_col=0, skiprows=[0])
        self.transport_times_df = pd.read_excel(file_path, sheet_name='transport_time', index_col=[0, 1], skiprows=[0])
        self.transport_times_exact_df = pd.read_excel(file_path, sheet_name='transport_time_exact', index_col=[0, 1],
                                                      skiprows=[0])
        self.loading_unloading_times_df = pd.read_excel(file_path, sheet_name='loading_unloading_time', index_col=0,
                                                        skiprows=[0])
        self.demands_df = pd.read_excel(file_path, sheet_name='demand', index_col=0, skiprows=[0])
        self.production_max_capacities_df = pd.read_excel(file_path, sheet_name='production_max_capacity', index_col=0,
                                                          skiprows=[0])
        # self.production_min_capacities_df = pd.read_excel(file_path, sheet_name='production_min_capacity', index_col=0, skiprows=[0])  # depreciated
        self.production_lines_for_factories_df = pd.read_excel(file_path, sheet_name='production_lines_for_factory',
                                                               index_col=0, skiprows=[0])
        self.production_line_min_times_df = pd.read_excel(file_path, sheet_name='production_line_min_time',
                                                          index_col=0, skiprows=[0])
        self.product_groups_df = pd.read_excel(file_path, sheet_name='product_group', index_col=0, skiprows=[0])
        self.key_values_df = pd.read_excel(file_path, sheet_name='key_values', index_col=0, skiprows=[0])
        self.factory_max_vessels_loading_df = pd.read_excel(file_path, sheet_name='factory_max_vessel_loading',
                                                            index_col=0, skiprows=[0])
        self.factory_max_vessel_destination_df = pd.read_excel(file_path, sheet_name='factory_max_vessel_destination',
                                                               index_col=0, skiprows=[0])
        self.order_zones_df = pd.read_excel(file_path, sheet_name='order_zones', index_col=0, skiprows=[0])
        # self.inventory_targets_df = pd.read_excel(file_path, sheet_name='inventory_target', index_col=0, skiprows=[0])
        self.production_stop_df = pd.read_excel(file_path, sheet_name='production_stop', index_col=0, skiprows=[0])
        self.production_start_costs_df = pd.read_excel(file_path, sheet_name='production_start_cost', index_col=0,
                                                       skiprows=[0])

        # Validate sets
        self._validate_set_consistency()
        self._validate_feasible_problem()

        # Attributes
        self.nodes = self.get_nodes()
        self.factory_nodes = self.get_factory_nodes()
        self.order_nodes = self.get_order_nodes()
        self.orders_for_zones = self.get_zone_orders_dict()
        self.zones = self.get_zones()
        self.nodes_for_vessels = self.get_nodes_for_vessels_dict()
        self.products = self.get_products()
        self.vessels = self.get_vessels()
        self.time_periods = self.get_time_periods()
        self.start_times_for_vessels = self.get_start_times_for_vessels_dict()
        self.vessel_initial_locations = self.get_vessel_first_location()
        self.time_windows_for_orders = self.get_time_windows_for_orders_dict()
        self.tw_start = {i: self.get_time_window_start(i) for i in self.order_nodes}
        self.tw_end = {i: self.get_time_window_end(i) for i in self.order_nodes}
        self.vessel_ton_capacities = self.get_vessel_ton_capacities_dict()
        self.vessel_nprod_capacities = self.get_vessel_nprod_capacities_dict()
        self.factory_inventory_capacities = self.get_inventory_capacities_dict()
        self.factory_initial_inventories = self.get_initial_inventories_dict()
        self.inventory_unit_costs = self.get_inventory_unit_costs_dict()
        self.loading_unloading_times = self.get_loading_unloading_times_dict()
        self.transport_unit_costs = self.get_transport_costs_dict()
        self.transport_times = self.get_transport_times_per_vessel_dict()
        self.transport_times_exact = self.get_transport_times_per_vessel_dict(exact=True)
        self.arcs_for_vessels = self.generate_arcs_for_vessels()
        self.demands = self.get_demands_dict()
        self.production_stops = self.get_production_stops_dict()
        self.production_start_costs = self.get_production_start_costs_dict()
        self.production_max_capacities = self.get_production_max_capacities_dict()
        self.production_lines = self.get_production_lines()
        self.production_lines_for_factories = self.get_production_lines_for_factories_list()
        self.production_line_min_times = self.get_production_line_min_times_dict()
        self.product_groups = self.get_product_groups_dict()
        self.factory_max_vessels_destination = self.get_factory_max_vessels_destination_dict()
        self.factory_max_vessels_loading = self.get_factory_max_vessels_loading_dict()
        # self.inventory_targets = self.get_inventory_targets()  # depreciated
        # self.inventory_unit_rewards = self.get_inventory_unit_rewards_dict()  # depreciated
        # self.external_delivery_penalty = self.get_key_value("external_delivery_penalty")  # depreciated
        self.min_wait_if_sick = self.get_min_wait_if_sick_dict()
        self.min_wait_if_sick_abs = self.get_min_wait_if_sick()
        self.external_delivery_penalties = self.get_external_delivery_penalties_dict()
        self.no_time_periods = len(self.time_periods)

    def _validate_set_consistency(self) -> None:
        # Products
        assert set(self.get_products()) == set(self.initial_inventories_df.columns)
        assert set(self.get_products()) == set(self.demands_df.columns)
        assert set(self.get_products()) == set(self.production_max_capacities_df.index)
        assert set(self.get_products()) == set(self.production_line_min_times_df.index)
        assert set(self.get_products()) == set(self.product_groups_df.index)
        # assert set(self.get_products()) == set(self.inventory_targets_df.index)  # depreciated
        assert set(self.get_products()) == set(self.production_start_costs_df.index)

        # Production lines
        assert set(self.get_production_lines()) == set(self.production_max_capacities_df.columns)
        assert set(self.get_production_lines()) == set(self.production_lines_for_factories_df.index)
        assert set(self.get_production_lines()) == set(self.production_line_min_times_df.columns)

        # Vessels
        assert set(self.get_vessels()) == set(self.nodes_for_vessels_df.index)
        assert set(self.get_vessels()) == set(self.vessel_availability_df.index)
        assert set(self.get_vessels()) == set(self.vessel_capacities_df.index)
        assert set(self.get_vessels()) == set(self.transport_unit_costs_df.index)
        assert set(self.get_vessels()) == set(self.loading_unloading_times_df.columns)

        # Factories
        assert set(self.get_factory_nodes()) == set(self.inventory_capacities_df.index)
        assert set(self.get_factory_nodes()) == set(self.initial_inventories_df.index)
        assert set(self.get_factory_nodes()) == set(self.inventory_unit_costs_rewards_df.index)
        assert set(self.get_factory_nodes()) == set(self.production_stop_df.columns)
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessels_loading_df.columns)
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessel_destination_df.index)
        # assert set(self.get_factory_nodes()) == set(self.inventory_targets_df.columns)  # depreciated
        assert set(self.get_factory_nodes()) == set(self.production_start_costs_df.columns)

        # Order nodes
        assert set(self.get_order_nodes()) == set(self.time_windows_for_orders_df.index)
        assert set(self.get_order_nodes()) == set(self.demands_df.index)

        # All nodes
        assert set(self.get_nodes()) == set(self.nodes_for_vessels_df.columns)
        assert set(self.get_nodes()) == set(self.transport_times_df.index.levels[0])
        assert set(self.get_nodes()) == set(self.transport_times_df.columns)
        assert set(self.get_nodes()) == set(self.loading_unloading_times_df.index)

        # Time periods
        assert set(self.get_time_periods()).issubset(set(self.get_time_periods()))
        assert set(self.get_time_periods()) == set(self.factory_max_vessels_loading_df.index)
        assert set(self.get_time_periods()) == set(self.production_stop_df.index)

    def _validate_feasible_problem(self) -> None:
        assert (sum(int(self.factory_max_vessel_destination_df.loc[factory, 'vessel_number'])
                    for factory in self.factory_max_vessel_destination_df.index)
                >= len(self.get_vessels()))

    def _validate_transport_time_triangle_inequality(self, distance_matrix: Dict[Tuple[str, str], int]) -> None:
        nodes = self.transport_times_df.index
        for origin in nodes:
            for intermediate in nodes:
                for destination in nodes:
                    assert (distance_matrix[origin, destination] <=
                            distance_matrix[origin, intermediate] +
                            distance_matrix[intermediate, destination])

    def get_vessels(self) -> List[str]:
        return list(self.vessel_capacities_df.index)

    def get_vessel_ton_capacities_dict(self) -> Dict[str, int]:
        return {vessel: self.vessel_capacities_df.loc[vessel, 'capacity [t]'] for vessel in
                self.vessel_capacities_df.index}

    def get_vessel_nprod_capacities_dict(self) -> Dict[str, int]:
        return {vessel: self.vessel_capacities_df.loc[vessel, 'capacity [nProd]'] for vessel in
                self.vessel_capacities_df.index}

    def get_order_nodes(self) -> List[str]:
        return list(self.time_windows_for_orders_df.index)

    def is_order_node(self, node_id: str) -> bool:
        return node_id.startswith("o")

    def is_factory_node(self, node_id: str) -> bool:
        return node_id.startswith("f")

    def get_zone_orders_dict(self):
        zone_orders_dict = defaultdict(list)
        for order in self.order_zones_df.index:
            zone = self.order_zones_df.loc[order, 'zone']
            zone_orders_dict[zone].append(order)
        return zone_orders_dict

    def get_zones(self) -> List[str]:
        return list(set([self.order_zones_df.loc[order, 'zone'] for order in self.order_zones_df.index]))

    def get_factory_nodes(self, v: str = None) -> List[str]:
        if v:
            return [i for i in self.initial_inventories_df.index if i in self.get_nodes_for_vessels_dict2()[v]]
        else:
            return list(self.initial_inventories_df.index)

    def get_nodes(self) -> List[str]:
        return self.get_order_nodes() + self.get_factory_nodes()

    def get_time_periods(self) -> List[int]:
        return list(self.factory_max_vessels_loading_df.index)

    def get_time_windows_for_orders_dict(self) -> Dict[Tuple[str, int], int]:
        return {(order_node, int(time_period)): int(self.get_time_window_start(order_node)
                                                    <= time_period
                                                    <= self.get_time_window_end(order_node))
                for order_node in self.time_windows_for_orders_df.index
                for time_period in self.get_time_periods()}

    def get_time_window_start(self, order_node):
        return self.time_windows_for_orders_df.loc[order_node, 'tw_start']

    def get_time_window_end(self, order_node):
        return self.time_windows_for_orders_df.loc[order_node, 'tw_end']

    def get_start_times_for_vessels_dict(self) -> Dict[str, int]:
        return {vessel: int(self.vessel_availability_df.loc[vessel, 'time_period'])
                for vessel in self.vessel_availability_df.index}

    def get_vessel_first_location(self) -> Dict[str, str]:
        return {vessel: (self.vessel_availability_df.loc[vessel, 'location'])
                for vessel in self.vessel_availability_df.index}

    def get_nodes_for_vessels_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): self.nodes_for_vessels_df.loc[vessel, node]
                for vessel in self.nodes_for_vessels_df.index
                for node in self.nodes_for_vessels_df.columns}

    def get_nodes_for_vessels_dict2(self) -> Dict[str, List[str]]:
        return {vessel: [node for node in self.nodes_for_vessels_df.columns
                         if self.nodes_for_vessels_df.loc[vessel, node] == 1]
                for vessel in self.nodes_for_vessels_df.index}

    def get_products(self) -> List[str]:
        return list(self.initial_inventories_df.columns)

    def get_production_lines(self) -> List[str]:
        return list(self.production_max_capacities_df.columns)

    def get_production_line_min_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(production_line, product): self.production_line_min_times_df.loc[product, production_line]
                for product in self.production_line_min_times_df.index
                for production_line in self.production_line_min_times_df.columns}

    def get_initial_inventories_dict(self) -> Dict[Tuple[str, str], int]:
        return {(factory_node, product): self.initial_inventories_df.loc[factory_node, product]
                for factory_node in self.initial_inventories_df.index
                for product in self.initial_inventories_df.columns}

    def get_inventory_capacities_dict(self) -> Dict[Tuple[str, str], int]:
        return {factory_node: self.inventory_capacities_df.loc[factory_node, 'capacity'] for factory_node in
                self.inventory_capacities_df.index}

    def get_inventory_unit_costs_dict(self) -> Dict[str, int]:
        return {factory_node: self.inventory_unit_costs_rewards_df.loc[factory_node, 'unit_cost'] for factory_node in
                self.inventory_unit_costs_rewards_df.index}

    def get_inventory_unit_rewards_dict(self) -> Dict[str, int]:
        return {factory_node: self.inventory_unit_costs_rewards_df.loc[factory_node, 'unit_reward'] for factory_node in
                self.inventory_unit_costs_rewards_df.index}

    def get_transport_costs_dict(self) -> Dict[str, int]:
        return {vessel: self.transport_unit_costs_df.loc[vessel, 'unit_transport_cost'] for vessel in
                self.transport_unit_costs_df.index}

    def get_transport_times_per_vessel_dict(self, exact: bool = False) -> Dict[Tuple[str, str, str], int]:
        # Triangle inequality holds for all distances in original distance matrix
        if exact:
            transport_times_df = self.transport_times_exact_df
        else:
            transport_times_df = self.transport_times_df

        d = {**{(vessel, orig, dest): transport_times_df.loc[(orig, vessel), dest]
                for orig in transport_times_df.index.levels[0]
                for dest in transport_times_df.columns
                for vessel in transport_times_df.index.levels[1]},
             **{(vessel, 'd_0', node): 1 for node in transport_times_df.index.levels[0]
                for vessel in transport_times_df.index.levels[1]},
             **{(vessel, node, 'd_-1'): 1 for node in transport_times_df.index.levels[0]
                for vessel in transport_times_df.index.levels[1]}}
        return d

    def get_loading_unloading_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): int(self.loading_unloading_times_df.loc[node, vessel])
                for node in self.loading_unloading_times_df.index
                for vessel in self.loading_unloading_times_df.columns}

    def get_demands_dict(self) -> Dict[Tuple[str, str, str], int]:
        order_node_dict = {(order_node, order_node, product): int(self.demands_df.loc[order_node, product])
                           for order_node in self.demands_df.index
                           for product in self.demands_df.columns}
        factory_node_dict = {(factory_node, order_node, product): -int(self.demands_df.loc[order_node, product])
                             for factory_node in self.get_factory_nodes()
                             for order_node in self.get_order_nodes()
                             for product in self.demands_df.columns}
        return {**order_node_dict, **factory_node_dict}

    def get_production_start_costs_dict(self) -> Dict[Tuple[str, str], int]:
        return {(factory_node, product): int(self.production_start_costs_df.loc[product, factory_node])
                for product in self.production_start_costs_df.index
                for factory_node in self.production_start_costs_df.columns}

    def get_production_stops_dict(self) -> Dict[Tuple[str, int], int]:
        return {(factory_node, time_period): int(self.production_stop_df.loc[time_period, factory_node])
                for time_period in self.production_stop_df.index
                for factory_node in self.production_stop_df.columns}

    def get_production_max_capacities_dict(self) -> Dict[Tuple[str, str], int]:
        return {(production_line, product): int(self.production_max_capacities_df.loc[product, production_line])
                for product in self.production_max_capacities_df.index
                for production_line in self.production_max_capacities_df.columns}

    def get_production_lines_for_factories_list(self) -> List[Tuple[str, str]]:
        return [(self.production_lines_for_factories_df.at[production_line, 'factory'], production_line)
                for production_line in self.production_lines_for_factories_df.index]

    def get_key_value(self, key):
        return self.key_values_df.loc[key, 'value']

    def get_min_wait_if_sick(self) -> int:
        return self.get_key_value("min_wait_if_sick")

    def get_min_wait_if_sick_dict(self) -> Dict:
        orders_for_zones = self.get_zone_orders_dict()
        min_wait = int(self.get_key_value('min_wait_if_sick'))
        vessels = self.get_vessels()

        # Find wait time periods from red/yellow to green
        min_wait_times_if_sick = {(v, i, j): min_wait - self.transport_times[v, i, j]
                                  for i in orders_for_zones['red'] + orders_for_zones['yellow']
                                  for j in orders_for_zones['green']
                                  for v in vessels
                                  if min_wait - self.transport_times[v, i, j] > 0
                                  and (i, j) in self.arcs_for_vessels[v]}

        # Add wait time periods from red to yellow
        min_wait_times_if_sick.update({(v, i, j): max(0, min_wait - self.transport_times[v, i, j])
                                       for i in orders_for_zones['red']
                                       for j in orders_for_zones['yellow']
                                       for v in vessels
                                       if min_wait - self.transport_times[v, i, j] > 0
                                       and (i, j) in self.arcs_for_vessels[v]})
        return min_wait_times_if_sick

    def _is_legal_arc(self, i, j, v):
        return (i, j) in self.arcs_for_vessels[v]

    def get_max_time_window_violation(self) -> int:
        return 0 if not self.soft_tw else int(self.get_key_value('max_tw_violation'))

    # def get_tw_violation_cost(self) -> float:
    #     return float(self.get_key_value('tw_violation_unit_cost'))

    def get_product_groups_dict(self) -> Dict[str, List[str]]:
        # Create dict on form (product : product group)
        product_group_dict = {product: self.product_groups_df.loc[product, 'product_group']
                              for product in self.product_groups_df.index}
        # Return dict on form (product_group : [product1, product2, ...]
        return {product_group: [product
                                for product in self.product_groups_df.index
                                if product_group_dict[product] == product_group]
                for product_group in set(product_group_dict.values())}

    def get_factory_max_vessels_loading_dict(self) -> Dict[Tuple[str, int], int]:
        return {(factory, time_period): int(self.factory_max_vessels_loading_df.loc[time_period, factory])
                for time_period in self.factory_max_vessels_loading_df.index
                for factory in self.factory_max_vessels_loading_df.columns}

    def get_external_delivery_penalties_dict(self) -> Dict[str, int]:
        max_transport_cost = 2 * (max(self.get_transport_costs_dict().values()) *
                                  max(self.transport_times_exact.values()))
        max_production_start_cost = (max(sum(self.demands[order, order, product] > 0 for product in self.products)
                                         for order in self.order_nodes) *  # max no. products in same order
                                     max(self.production_start_costs[factory, product]
                                         for factory in self.factory_nodes
                                         for product in self.products))  # max production start cost
        max_inventory_cost = (max(self.demands[order, order, product]
                                  for product in self.products
                                  for order in self.order_nodes) *   # max order quantity
                              max(self.inventory_unit_costs[factory] for factory in self.factory_nodes) *
                              # max inventory unit cost
                              len(self.time_periods))  # no. time periods
        return {order: max_transport_cost + max_production_start_cost + max_inventory_cost for order in self.order_nodes}

    def get_factory_max_vessels_destination_dict(self) -> Dict[str, int]:
        return {factory: int(self.factory_max_vessel_destination_df.loc[factory])
                for factory in self.factory_max_vessel_destination_df.index}

    # def get_inventory_targets(self) -> Dict[Tuple[str, str], int]:
    #     return {(factory, product): int(self.inventory_targets_df.loc[product, factory])
    #             for product in self.inventory_targets_df.index
    #             for factory in self.inventory_targets_df.columns}

    def generate_arcs_for_vessels(self) -> Dict[str, List[Tuple[str, str]]]:
        dummy_start_arc = {v: [('d_0', i)] for v, i in self.get_vessel_first_location().items()}
        dummy_end_arcs = {v: [(i, 'd_-1') for i in self.get_factory_nodes(v)] for v in self.get_vessels()}
        all_other_arcs = {v: [(i, j)
                              for i in self.get_nodes_for_vessels_dict2()[v]
                              for j in self.get_nodes_for_vessels_dict2()[v]
                              if i != j
                              and not self._has_tw_conflict(v, i, j)]
                          for v in self.get_vessels()}
        arcs = {v: dummy_start_arc[v] + dummy_end_arcs[v] + all_other_arcs[v] for v in self.get_vessels()}
        return arcs

    def _has_tw_conflict(self, v, i, j):
        # i or j is a factory node
        if i in self.get_factory_nodes() or j in self.get_factory_nodes():
            return False
        # i or j's time window ends before vessel becomes available
        elif (self.get_time_window_end(j) < self.get_start_times_for_vessels_dict()[v]
              or self.get_time_window_end(i) < self.get_start_times_for_vessels_dict()[v]):
            return True
        else:
            extra_violation = 2 * self.get_max_time_window_violation() if self.soft_tw else 0
            return (self.get_time_window_start(i)
                    + self.loading_unloading_times[v, i]
                    + self.transport_times[v, i, j]
                    >
                    self.get_time_window_end(j)
                    + extra_violation)


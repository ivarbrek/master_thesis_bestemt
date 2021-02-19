import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class ProblemData:

    def __init__(self, file_path: str, soft_tw=False) -> None:
        self.soft_tw = soft_tw

        self.vessel_capacities = pd.read_excel(file_path, sheet_name='vessel_capacity', index_col=0, skiprows=[0])
        self.inventory_capacities = pd.read_excel(file_path, sheet_name='inventory_capacity', index_col=0, skiprows=[0])
        self.time_windows_for_orders = pd.read_excel(file_path, sheet_name='time_windows_for_order', index_col=0,
                                                     skiprows=[0])
        self.vessel_availability = pd.read_excel(file_path, sheet_name='vessel_availability', index_col=0,
                                                 skiprows=[0])
        self.nodes_for_vessels = pd.read_excel(file_path, sheet_name='nodes_for_vessel', index_col=0, skiprows=[0])
        self.initial_inventories = pd.read_excel(file_path, sheet_name='initial_inventory', index_col=0, skiprows=[0])
        self.inventory_unit_costs_rewards = pd.read_excel(file_path, sheet_name='inventory_cost', index_col=0, skiprows=[0])
        self.transport_unit_costs = pd.read_excel(file_path, sheet_name='transport_cost', index_col=0, skiprows=[0])
        self.transport_times = pd.read_excel(file_path, sheet_name='transport_time', index_col=0, skiprows=[0])
        self.loading_unloading_times = pd.read_excel(file_path, sheet_name='loading_unloading_time', index_col=0, skiprows=[0])
        self.demands = pd.read_excel(file_path, sheet_name='demand', index_col=0, skiprows=[0])
        self.production_max_capacities = pd.read_excel(file_path, sheet_name='production_max_capacity', index_col=0,
                                                       skiprows=[0])
        self.production_min_capacities = pd.read_excel(file_path, sheet_name='production_min_capacity', index_col=0,
                                                       skiprows=[0])
        self.production_lines_for_factories = pd.read_excel(file_path, sheet_name='production_lines_for_factory',
                                                            index_col=0, skiprows=[0])
        self.production_line_min_times = pd.read_excel(file_path, sheet_name='production_line_min_time',
                                                       index_col=0, skiprows=[0])
        self.product_groups = pd.read_excel(file_path, sheet_name='product_group', index_col=0, skiprows=[0])
        self.key_values = pd.read_excel(file_path, sheet_name='key_values', index_col=0, skiprows=[0])
        self.factory_max_vessels_loading = pd.read_excel(file_path, sheet_name='factory_max_vessel_loading', index_col=0, skiprows=[0])
        self.factory_max_vessel_destination = pd.read_excel(file_path, sheet_name='factory_max_vessel_destination', index_col=0, skiprows=[0])
        self.order_zones = pd.read_excel(file_path, sheet_name='order_zones', index_col=0, skiprows=[0])
        self.inventory_targets = pd.read_excel(file_path, sheet_name='inventory_target', index_col=0, skiprows=[0])
        self.production_stop = pd.read_excel(file_path, sheet_name='production_stop', index_col=0, skiprows=[0])
        self.production_start_costs = pd.read_excel(file_path, sheet_name='production_start_cost', index_col=0, skiprows=[0])

        # Validate sets
        self._validate_set_consistency()
        self._validate_feasible_problem()

    def _validate_set_consistency(self) -> None:
        # Products
        assert set(self.get_products()) == set(self.initial_inventories.columns)
        assert set(self.get_products()) == set(self.demands.columns)
        assert set(self.get_products()) == set(self.production_max_capacities.index)
        assert set(self.get_products()) == set(self.production_min_capacities.index)
        assert set(self.get_products()) == set(self.production_line_min_times.index)
        assert set(self.get_products()) == set(self.product_groups.index)
        assert set(self.get_products()) == set(self.inventory_targets.index)
        assert set(self.get_products()) == set(self.production_start_costs.index)

        # Production lines
        assert set(self.get_production_lines()) == set(self.production_max_capacities.columns)
        assert set(self.get_production_lines()) == set(self.production_min_capacities.columns)
        assert set(self.get_production_lines()) == set(self.production_lines_for_factories.index)
        assert set(self.get_production_lines()) == set(self.production_line_min_times.columns)

        # Vessels
        assert set(self.get_vessels()) == set(self.nodes_for_vessels.index)
        assert set(self.get_vessels()) == set(self.vessel_availability.index)
        assert set(self.get_vessels()) == set(self.vessel_capacities.index)
        assert set(self.get_vessels()) == set(self.transport_unit_costs.index)
        assert set(self.get_vessels()) == set(self.loading_unloading_times.columns)

        # Factories
        assert set(self.get_factory_nodes()) == set(self.inventory_capacities.index)
        assert set(self.get_factory_nodes()) == set(self.initial_inventories.index)
        assert set(self.get_factory_nodes()) == set(self.inventory_unit_costs_rewards.index)
        assert set(self.get_factory_nodes()) == set(self.production_stop.columns)
        assert set(self.get_factory_nodes()) == set(
            self.production_lines_for_factories['factory'])  # At least 1 production line per factory
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessels_loading.columns)
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessel_destination.index)
        assert set(self.get_factory_nodes()) == set(self.inventory_targets.columns)
        assert set(self.get_factory_nodes()) == set(self.production_start_costs.columns)

        # Order nodes
        assert set(self.get_order_nodes()) == set(self.time_windows_for_orders.index)
        assert set(self.get_order_nodes()) == set(self.demands.index)

        # All nodes
        assert set(self.get_nodes()) == set(self.nodes_for_vessels.columns)
        assert set(self.get_nodes()) == set(self.transport_times.index)
        assert set(self.get_nodes()) == set(self.transport_times.columns)
        assert set(self.get_nodes()) == set(self.loading_unloading_times.index)

        # Time periods
        assert set(self.get_time_periods()) == set(self.time_windows_for_orders.columns)
        assert set(self.get_time_periods()) == set(self.factory_max_vessels_loading.index)
        assert set(self.get_time_periods()) == set(self.production_stop.index)

    def _validate_feasible_problem(self) -> None:
        assert (sum(int(self.factory_max_vessel_destination.loc[factory, 'vessel_number'])
                    for factory in self.factory_max_vessel_destination.index)
                >= len(self.get_vessels()))

    def get_vessels(self) -> List[str]:
        return list(self.vessel_capacities.index)

    def get_vessel_ton_capacities_dict(self) -> Dict[str, int]:
        return {vessel: self.vessel_capacities.loc[vessel, 'capacity [t]'] for vessel in self.vessel_capacities.index}

    def get_vessel_nprod_capacities_dict(self) -> Dict[str, int]:
        return {vessel: self.vessel_capacities.loc[vessel, 'capacity [nProd]'] for vessel in
                self.vessel_capacities.index}

    def get_order_nodes(self) -> List[str]:
        return list(self.time_windows_for_orders.index)

    def get_zone_orders_dict(self):
        zone_orders_dict = defaultdict(list)
        for order in self.order_zones.index:
            zone = self.order_zones.loc[order, 'zone']
            zone_orders_dict[zone].append(order)
        return zone_orders_dict

    def get_factory_nodes(self, v: str = None) -> List[str]:
        if v:
            return [i for i in self.initial_inventories.index if i in self.get_nodes_for_vessels_dict2()[v]]
        else:
            return list(self.initial_inventories.index)

    def get_nodes(self) -> List[str]:
        return self.get_order_nodes() + self.get_factory_nodes()

    def get_time_periods(self) -> List[int]:
        return list(int(time_period) for time_period in self.time_windows_for_orders.columns)

    def get_time_windows_for_orders_dict(self) -> Dict[Tuple[str, int], int]:
        return {(order_node, int(time_period)): self.time_windows_for_orders.loc[order_node, time_period]
                for order_node in self.time_windows_for_orders.index
                for time_period in self.time_windows_for_orders.columns}

    def get_time_window_start(self, order_node):
        return min(t for (i, t), val in self.get_time_windows_for_orders_dict().items() if i == order_node and val == 1)

    def get_time_window_end(self, order_node):
        return max(t for (i, t), val in self.get_time_windows_for_orders_dict().items() if i == order_node and val == 1)

    def get_time_periods_for_vessels_dict(self) -> Dict[str, int]:
        return {vessel: int(self.vessel_availability.loc[vessel, 'time_period'])
                for vessel in self.vessel_availability.index}

    def get_vessel_first_location(self) -> Dict[str, str]:
        return {vessel: (self.vessel_availability.loc[vessel, 'location'])
                for vessel in self.vessel_availability.index}

    def get_nodes_for_vessels_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): self.nodes_for_vessels.loc[vessel, node]
                for vessel in self.nodes_for_vessels.index
                for node in self.nodes_for_vessels.columns}

    def get_nodes_for_vessels_dict2(self) -> Dict[str, List[str]]:
        return {vessel: [node for node in self.nodes_for_vessels.columns
                         if self.nodes_for_vessels.loc[vessel, node] == 1]
                for vessel in self.nodes_for_vessels.index}

    def get_products(self) -> List[str]:
        return list(self.initial_inventories.columns)

    def get_production_lines(self) -> List[str]:
        return list(self.production_max_capacities.columns)

    def get_production_line_min_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(production_line, product): self.production_line_min_times.loc[product, production_line]
                for product in self.production_line_min_times.index
                for production_line in self.production_line_min_times.columns}

    def get_initial_inventories_dict(self) -> Dict[Tuple[str, str], int]:
        return {(factory_node, product): self.initial_inventories.loc[factory_node, product]
                for factory_node in self.initial_inventories.index
                for product in self.initial_inventories.columns}

    def get_inventory_capacities_dict(self) -> Dict[Tuple[str, str], int]:
        return {factory_node: self.inventory_capacities.loc[factory_node, 'capacity [t]'] for factory_node in
                self.inventory_capacities.index}

    def get_inventory_unit_costs_dict(self) -> Dict[str, int]:
        return {factory_node: self.inventory_unit_costs_rewards.loc[factory_node, 'unit_cost'] for factory_node in
                self.inventory_unit_costs_rewards.index}

    def get_inventory_unit_rewards_dict(self) -> Dict[str, int]:
        return {factory_node: self.inventory_unit_costs_rewards.loc[factory_node, 'unit_reward'] for factory_node in
                self.inventory_unit_costs_rewards.index}

    def get_transport_costs_dict(self) -> Dict[str, int]:
        return {vessel: self.transport_unit_costs.loc[vessel, 'unit_transport_cost'] for vessel in
                self.transport_unit_costs.index}

    def get_transport_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {**{(node1, node2): self.transport_times.loc[node1, node2]
                   for node1 in self.transport_times.index
                   for node2 in self.transport_times.columns},
                **{('d_0', node): 1 for node in self.transport_times.index},
                **{(node, 'd_-1'): 1 for node in self.transport_times.index}}

    def get_loading_unloading_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): int(self.loading_unloading_times.loc[node, vessel])
                for node in self.loading_unloading_times.index
                for vessel in self.loading_unloading_times.columns}

    def get_demands_dict(self) -> Dict[Tuple[str, str], int]:
        order_node_dict = {(order_node, order_node, product): int(self.demands.loc[order_node, product])
                           for order_node in self.demands.index
                           for product in self.demands.columns}
        factory_node_dict = {(factory_node, order_node, product): -int(self.demands.loc[order_node, product])
                             for factory_node in self.get_factory_nodes()
                             for order_node in self.get_order_nodes()
                             for product in self.demands.columns}
        return {**order_node_dict, **factory_node_dict}

    def get_production_start_costs_dict(self) -> Dict[Tuple[str, str], int]:
        return {(factory_node, product): int(self.production_start_costs.loc[product, factory_node])
                for product in self.production_start_costs.index
                for factory_node in self.production_start_costs.columns}

    def get_production_stops_dict(self) -> Dict[Tuple[str, int], int]:
        return {(factory_node, time_period): int(self.production_stop.loc[time_period, factory_node])
                for time_period in self.production_stop.index
                for factory_node in self.production_stop.columns}

    def get_production_max_capacities_dict(self) -> Dict[Tuple[str, str], int]:
        return {(production_line, product): int(self.production_max_capacities.loc[product, production_line])
                for product in self.production_max_capacities.index
                for production_line in self.production_max_capacities.columns}

    def get_production_min_capacities_dict(self) -> Dict[Tuple[str, str], int]:
        return {(production_line, product): int(self.production_min_capacities.loc[product, production_line])
                for product in self.production_min_capacities.index
                for production_line in self.production_min_capacities.columns}

    def get_production_lines_for_factories_list(self) -> List[Tuple[str, str]]:
        return [(self.production_lines_for_factories.at[production_line, 'factory'], production_line) for
                production_line
                in self.production_lines_for_factories.index]

    def get_key_value(self, key):
        return self.key_values.loc[key, 'value']

    def get_min_wait_if_sick(self) -> Dict:
        orders_for_zones = self.get_zone_orders_dict()
        transport_times = self.get_transport_times_dict()
        min_wait = int(self.get_key_value('min_wait_if_sick'))

        # Find wait time periods from red/yellow to green
        min_wait_times_if_sick = {(i, j): min_wait - transport_times[i, j]
                                  for i in orders_for_zones['red'] + orders_for_zones['yellow']
                                  for j in orders_for_zones['green']
                                  if min_wait - transport_times[i, j] > 0}

        # Add wait time periods from red to yellow
        min_wait_times_if_sick.update({(i, j): max(0, min_wait - transport_times[i, j])
                                       for i in orders_for_zones['red']
                                       for j in orders_for_zones['yellow']
                                       if min_wait - transport_times[i, j] > 0})
        return min_wait_times_if_sick

    def get_max_time_window_violation(self) -> int:
        return int(self.get_key_value('max_tw_violation'))

    def get_tw_violation_cost(self) -> float:
        return float(self.get_key_value('tw_violation_unit_cost'))

    def get_product_groups_dict(self) -> Dict[str, List[str]]:
        # Create dict on form (product : product group)
        product_group_dict = {product: self.product_groups.loc[product, 'product_group']
                              for product in self.product_groups.index}
        # Return dict on form (product_group : [product1, product2, ...]
        return {product_group: [product
                                for product in self.product_groups.index
                                if product_group_dict[product] == product_group]
                for product_group in set(product_group_dict.values())}

    def get_factory_max_vessels_loading_dict(self) -> Dict[Tuple[str, int], int]:
        return {(factory, time_period): int(self.factory_max_vessels_loading.loc[time_period, factory])
                for time_period in self.factory_max_vessels_loading.index
                for factory in self.factory_max_vessels_loading.columns}

    def get_factory_max_vessels_destination_dict(self) -> Dict[str, int]:
        return {factory: int(self.factory_max_vessel_destination.loc[factory])
                for factory in self.factory_max_vessel_destination.index}

    def get_inventory_targets(self) -> Dict[Tuple[str, str], int]:
        return {(factory, product): int(self.inventory_targets.loc[product, factory])
                for product in self.inventory_targets.index
                for factory in self.inventory_targets.columns}

    def get_arcs_for_vessel_dict(self) -> Dict[str, List[Tuple[str, str]]]:
        dummy_start_arc = {v: [('d_0', i)] for v, i in self.get_vessel_first_location().items()}
        dummy_end_arcs = {v: [(i, 'd_-1') for i in self.get_factory_nodes(v)] for v in self.get_vessels()}
        all_other_arcs = {v: [(i, j)
                              for i in self.get_nodes_for_vessels_dict2()[v]
                              for j in self.get_nodes_for_vessels_dict2()[v]
                              if i != j
                              and not self._has_tw_conflict(v, i, j)]
                          for v in self.get_vessels()}
        return {v: dummy_start_arc[v] + dummy_end_arcs[v] + all_other_arcs[v] for v in self.get_vessels()}

    def _has_tw_conflict(self, v, i, j):
        # i or j is a factory node
        if i in self.get_factory_nodes() or j in self.get_factory_nodes():
            return False
        # i or j's time window ends before vessel becomes available
        elif (self.get_time_window_end(j) < self.get_time_periods_for_vessels_dict()[v]
              or self.get_time_window_end(i) < self.get_time_periods_for_vessels_dict()[v]):
            return True
        else:
            extra_violation = 2 * self.get_max_time_window_violation() if self.soft_tw else 0
            return (self.get_time_window_start(i)
                    + self.get_loading_unloading_times_dict()[v, i]
                    + self.get_transport_times_dict()[i, j]
                    >
                    self.get_time_window_end(j)
                    + extra_violation)



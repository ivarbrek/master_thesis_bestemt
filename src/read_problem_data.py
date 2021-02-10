import pandas as pd
from typing import List, Dict, Tuple


class ProblemData:

    def __init__(self, file_path: str) -> None:
        self.vessel_capacities = pd.read_excel(file_path, sheet_name='vessel_capacity', index_col=0, skiprows=[0])
        self.inventory_capacities = pd.read_excel(file_path, sheet_name='inventory_capacity', index_col=0, skiprows=[0])
        self.time_windows_for_orders = pd.read_excel(file_path, sheet_name='time_windows_for_order', index_col=0,
                                                     skiprows=[0])
        self.vessel_availability = pd.read_excel(file_path, sheet_name='vessel_availability', index_col=0,
                                                 skiprows=[0])
        self.vessel_initial_loads = pd.read_excel(file_path, sheet_name='initial_vessel_load', index_col=0,
                                                  skiprows=[0])
        self.nodes_for_vessels = pd.read_excel(file_path, sheet_name='nodes_for_vessel', index_col=0, skiprows=[0])
        self.initial_inventories = pd.read_excel(file_path, sheet_name='initial_inventory', index_col=0, skiprows=[0])
        self.inventory_unit_costs_rewards = pd.read_excel(file_path, sheet_name='inventory_cost', index_col=0, skiprows=[0])
        self.transport_unit_costs = pd.read_excel(file_path, sheet_name='transport_cost', index_col=0, skiprows=[0])
        self.transport_times = pd.read_excel(file_path, sheet_name='transport_time', index_col=0, skiprows=[0])
        self.unloading_times = pd.read_excel(file_path, sheet_name='unloading_time', index_col=0, skiprows=[0])
        self.loading_times = pd.read_excel(file_path, sheet_name='loading_time', index_col=0, skiprows=[0])
        self.demands = pd.read_excel(file_path, sheet_name='demand', index_col=0, skiprows=[0])
        self.production_unit_costs = pd.read_excel(file_path, sheet_name='production_cost', index_col=0, skiprows=[0])
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
        self.inventory_targets = pd.read_excel(file_path, sheet_name='inventory_target', index_col=0, skiprows=[0])

        # Validate sets
        self._validate_set_consistency()
        self._validate_feasible_problem()

    def _validate_set_consistency(self) -> None:
        # Products
        assert set(self.get_products()) == set(self.initial_inventories.columns)
        assert set(self.get_products()) == set(self.vessel_initial_loads.index)
        assert set(self.get_products()) == set(self.demands.columns)
        assert set(self.get_products()) == set(self.production_unit_costs.index)
        assert set(self.get_products()) == set(self.production_max_capacities.index)
        assert set(self.get_products()) == set(self.production_min_capacities.index)
        assert set(self.get_products()) == set(self.production_line_min_times.index)
        assert set(self.get_products()) == set(self.product_groups.index)
        assert set(self.get_products()) == set(self.inventory_targets.index)

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
        assert set(self.get_vessels()) == set(self.unloading_times.columns)
        assert set(self.get_vessels()) == set(self.loading_times.columns)
        assert set(self.get_vessels()) == set(self.vessel_initial_loads.columns)

        # Factories
        assert set(self.get_factory_nodes()) == set(self.inventory_capacities.index)
        assert set(self.get_factory_nodes()) == set(self.initial_inventories.index)
        assert set(self.get_factory_nodes()) == set(self.inventory_unit_costs_rewards.index)
        assert set(self.get_factory_nodes()) == set(self.production_unit_costs.columns)
        assert set(self.get_factory_nodes()) == set(
            self.production_lines_for_factories['factory'])  # At least 1 production line per factory
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessels_loading.columns)
        assert set(self.get_factory_nodes()) == set(self.factory_max_vessel_destination.index)
        assert set(self.get_factory_nodes()) == set(self.inventory_targets.columns)

        # Order nodes
        assert set(self.get_order_nodes()) == set(self.time_windows_for_orders.index)
        assert set(self.get_order_nodes()) == set(self.demands.index)

        # All nodes
        assert set(self.get_nodes()) == set(self.nodes_for_vessels.columns)
        assert set(self.get_nodes()) == set(self.transport_times.index)
        assert set(self.get_nodes()) == set(self.transport_times.columns)
        assert set(self.get_nodes()) == set(self.unloading_times.index)
        assert set(self.get_nodes()) == set(self.loading_times.index)

        # Time periods
        assert set(self.get_time_periods()) == set(self.time_windows_for_orders.columns)
        assert set(self.get_time_periods()) == set(self.factory_max_vessels_loading.index)

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

    def get_factory_nodes(self) -> List[str]:
        return list(self.initial_inventories.index)

    def get_nodes(self) -> List[str]:
        return self.get_order_nodes() + self.get_factory_nodes()

    def get_time_periods(self) -> List[int]:
        return list(int(time_period) for time_period in self.time_windows_for_orders.columns)

    def get_time_windows_for_orders_dict(self) -> Dict[Tuple[str, int], int]:
        return {(order_node, int(time_period)): self.time_windows_for_orders.loc[order_node, time_period]
                for order_node in self.time_windows_for_orders.index
                for time_period in self.time_windows_for_orders.columns}

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

    def get_vessel_initial_loads_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, product): self.vessel_initial_loads.loc[product, vessel]
                for product in self.vessel_initial_loads.index
                for vessel in self.vessel_initial_loads.columns}

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
                **{('d_0', node): 0 for node in self.transport_times.index}}

    def get_unloading_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): self.unloading_times.loc[node, vessel]
                for node in self.unloading_times.index
                for vessel in self.unloading_times.columns}

    def get_loading_times_dict(self) -> Dict[Tuple[str, str], int]:
        return {(vessel, node): int(self.loading_times.loc[node, vessel])
                for node in self.loading_times.index
                for vessel in self.loading_times.columns}

    def get_demands_dict(self) -> Dict[Tuple[str, str], int]:
        order_node_dict = {(order_node, order_node, product): int(self.demands.loc[order_node, product])
                           for order_node in self.demands.index
                           for product in self.demands.columns}
        factory_node_dict = {(factory_node, order_node, product): -int(self.demands.loc[order_node, product])
                             for factory_node in self.get_factory_nodes()
                             for order_node in self.get_order_nodes()
                             for product in self.demands.columns}
        return {**order_node_dict, **factory_node_dict}

    def get_production_unit_costs_dict(self) -> Dict[Tuple[str, str], int]:
        return {(factory_node, product): int(self.production_unit_costs.loc[product, factory_node])
                for product in self.production_unit_costs.index
                for factory_node in self.production_unit_costs.columns}

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

    def get_key_values(self) -> Dict[str, int]:
        return {key: int(self.key_values.loc[key, 'value']) for key in self.key_values.index}

    def get_product_groups(self) -> Dict[str, str]:
        return {product: self.product_groups.loc[product, 'product_group'] for product in self.product_groups.index}

    def get_product_shifting_costs(self) -> Dict[Tuple[str, str], int]:
        product_groups = self.get_product_groups()
        return {(product1, product2): (self.get_key_values()['shifting_cost'] *
                                       (0 if (product_groups[product1] == product_groups[product2]) else 1))
                for product1 in self.get_products()
                for product2 in self.get_products()}

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

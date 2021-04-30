from __future__ import annotations
import math
from typing import List, Dict, Tuple
import pandas as pd
from src.generate_data import orders_generator
import random
import numpy as np


def _hours_to_time_periods(hours: float, time_period_length: float) -> int:
    if hours % time_period_length < 0.25:  # TODO: Is this OK? Rounding down if less than 0.25 over time period threshold
        time = math.floor(hours / time_period_length)
    else:
        time = math.ceil(hours / time_period_length)
    return time


class NodeLocationMapping:

    node_location_mapping: Dict[str, str] = {}

    def __init__(self, order_locations: List[str], factory_locations: List[str]):
        """
        :param order_locations: location id for aquaculture farm
        :param factory_locations: location id for aquaculture farm
        """
        for i in range(len(order_locations)):
            self.node_location_mapping["o_" + str(i)] = order_locations[i]
        for i in range(len(factory_locations)):
            self.node_location_mapping["f_" + str(i)] = factory_locations[i]

    def get_location_from_node(self, node_id: str) -> str:
        return self.node_location_mapping[node_id]

    def get_nodes(self) -> List[str]:
        return list(self.node_location_mapping.keys())

    def get_locations(self) -> List[str]:
        return list(self.node_location_mapping.values())

    def get_order_nodes(self) -> List[str]:
        return [node for node in self.node_location_mapping.keys() if node.startswith("o")]

    def get_factory_nodes(self) -> List[str]:
        return [node for node in self.node_location_mapping.keys() if node.startswith("f")]

    def get_factory_items(self) -> List[Tuple]:
        return [(node, location) for node, location in self.node_location_mapping.items() if node.startswith("f")]

    @staticmethod
    def node_is_factory(node_id: str) -> bool:
        return node_id.startswith("f")

    @staticmethod
    def node_is_order(node_id: str) -> bool:
        return node_id.startswith("o")


class TestDataGenerator:
    excel_writer: pd.ExcelWriter
    nlm: NodeLocationMapping
    prod_lines_for_factory: Dict[str, List[str]]
    time_windows_df: pd.DataFrame

    def __init__(self):
        self.vessels_df = pd.read_excel("../../data/vessels.xlsx", sheet_name="vessels", index_col=[0])
        self.distances_df = pd.read_excel("../../data/distance_matrix_2.xlsx", sheet_name="distances", index_col=[0])
        self.factories_df = pd.read_excel("../../data/factories.xlsx", sheet_name="factories", index_col=[0],
                                          skiprows=[0])
        self.factories_df.index = self.factories_df.index.astype('str')

    def write_test_instance_to_file(self, out_filepath: str,
                                    vessels: List[str],
                                    factory_locations: List[str],
                                    time_periods: int,
                                    time_period_length: int,
                                    no_orders: int,
                                    no_products: int,
                                    no_product_groups: int,
                                    factory_level: float,
                                    ext_depot_level: float,
                                    quay_activity_level: float,
                                    orders_from_company: str = None):
        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w',
                                           options={'strings_to_formulas': False})

        # Set orders and factories
        orders = orders_generator.sample_orders_df(no_orders, company=orders_from_company, no_products=no_products)
        order_locations = list(orders.index)
        self.nlm = NodeLocationMapping(order_locations, factory_locations)
        self.factories_df = self.factories_df.loc[factory_locations]
        # Write sheets to file
        self.write_demands_to_file(orders)
        self.write_transport_times_to_file(relevant_vessels=vessels, relevant_nodes=self.nlm.get_nodes(),
                                           time_period_length=time_period_length)
        self.write_loading_times_to_file(relevant_vessels=vessels, relevant_nodes=self.nlm.get_nodes(),
                                         time_period_length=time_period_length)
        self.write_time_windows_to_file(time_periods, tw_length=20, earliest_tw_start=5)

        self.write_factory_max_vessel_destination_to_file(vessels)
        self.write_factory_max_vessel_loading_to_file(quay_activity_level, time_periods, time_period_length)
        self.write_initial_inventory_to_file(factory_level, ext_depot_level, orders)
        self.write_inventory_capacity_to_file()
        self.write_inventory_cost_to_file()
        self.write_production_start_cost_to_file(no_products)
        self.write_product_group_to_file(no_product_groups, no_products)
        self.write_production_lines_for_factory_to_file()
        self.write_production_max_capacity_to_file(no_products)
        self.write_prod_line_min_time_to_file(time_period_length, no_products)
        self.excel_writer.close()

    def generate_random_time_windows(self, time_periods: int, tw_length: int, earliest_tw_start: int) -> pd.DataFrame:
        data = []
        for order in self.nlm.get_order_nodes():
            tw_start = random.randint(earliest_tw_start, time_periods - tw_length)  # pick random tw start
            tw_end = tw_start + tw_length
            data.append((order, tw_start, tw_end))
        df = pd.DataFrame(data)
        df.columns = ['order', 'tw_start', 'tw_end']
        df = df.set_index('order')
        self.time_windows_df = df
        return df

    def generate_initial_inventory(self, factory_level: float, ext_depot_level: float, orders: pd.DataFrame):
        """
        :param factory_level: percentage of total demand stored on factories (excl. external depots)
        :param ext_depot_level: percentage of inventory capacity stored on external depots
        :param orders: DataFrame with orders
        :return: DataFrame with initial inventories
        """
        no_products = len(orders.columns)
        df = orders.join(self.time_windows_df)
        df = df.sort_values('tw_start')
        df = df.drop(columns=['tw_start', 'tw_end'])
        # Note: In this function we distinguish factories form external depots
        factories = [node for node in self.nlm.get_factory_nodes() if not self._is_external_depot_node(node)]
        ext_depots = [node for node in self.nlm.get_factory_nodes() if self._is_external_depot_node(node)]
        inv_capacities = {node: self.factories_df.loc[self.nlm.get_location_from_node(node), 'inventory_capacity']
                          for node in factories + ext_depots}
        total_demand = df.sum().sum()

        # Initialize inventory with small values
        init_inventory = {node: np.random.randint(0, 0.1 * total_demand // no_products, no_products)
                          for node in factories}

        # Add a proportion of earliest orders until desired factory_level is reached
        for idx, row in df.iterrows():
            chosen_factory = random.choice(factories)
            init_inventory[chosen_factory] += (row.to_numpy() * 0.8).astype('int64')
            if sum(inventory.sum() for factory, inventory in init_inventory.items()) > factory_level * total_demand:
                break

        # Check that capacity is not violated
        for factory in factories:
            assert init_inventory[factory].sum() <= inv_capacities[factory], "Init. inventory > capacity "

        # Add inventory for external depots
        for depot in ext_depots:
            capacity = self.factories_df.loc[self.nlm.get_location_from_node(depot), 'inventory_capacity']
            # Linear decrease in initial inventory for products,
            # where p_0 gets ext_depot_level * capacity and p_n gets 0.
            init_inventory[depot] = [int((no_products - i)/no_products * ext_depot_level * inv_capacities[depot])
                                     for i in range(no_products)]

        df = pd.DataFrame(init_inventory)
        df.index = [f'p_{i}' for i in range(no_products)]
        return df.transpose()

    def get_factory_max_vessel_destination(self, vessels: List[str]) -> pd.DataFrame:
        no_vessels = len(vessels)
        no_factories = len(self.factories_df)
        # no_vessels // no_factories + 1 for factories, 0 for external depots
        data = [(node, (no_vessels // no_factories + 1) * (not self._is_external_depot_node(node)))
                for node in self.nlm.get_factory_nodes()]
        df = pd.DataFrame(data)
        df.columns = ['factory', 'vessel_number']
        df = df.set_index('factory')
        return df

    def get_factory_max_vessel_loading(self, quay_activity_level: float, time_periods: int,
                                        time_period_length: int) -> pd.DataFrame:
        # Note: In this function we distinguish factories form external depots
        factories = [node for node in self.nlm.get_factory_nodes() if not self._is_external_depot_node(node)]

        # rm = raw material
        rm_loading_time = 12   # 12 hours
        rm_loading_periods = rm_loading_time // time_period_length
        rm_loading_insert_count = int(time_periods * quay_activity_level / rm_loading_periods)

        max_vessels_loading = {factory: [self.factories_df.loc[location, 'max_vessel_loading']] * time_periods
                               for factory, location in self.nlm.get_factory_items()
                               if not self._is_external_depot_node(factory)}
        for factory in factories:
            for _ in range(rm_loading_insert_count):
                possible_inserts = [t for t, max_vessels in enumerate(max_vessels_loading[factory])
                                    if (t <= time_periods - rm_loading_periods and
                                        all(max_vessels_loading[factory][tau] > 0
                                            for tau in range(t, t + rm_loading_periods)))]
                insert_t = random.choice(possible_inserts)
                for t in range(insert_t, insert_t + rm_loading_periods):
                    max_vessels_loading[factory][t] -= 1

        return pd.DataFrame(max_vessels_loading)

    def get_all_loading_times(self, relevant_vessels: List[str], relevant_nodes: List[str],
                              time_period_length: int) -> pd.DataFrame:
        loading_times_df = pd.DataFrame(relevant_nodes, columns=["node"]).set_index("node", drop=True)
        for v in relevant_vessels:
            d = {}
            for node in loading_times_df.index:
                try:
                    if self.nlm.node_is_factory(node):
                        hours = 100 / self.vessels_df.loc[v, "loading_rate"]  # TODO: Fix quantity read (in elif also)
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    else:
                        hours = 100 / self.vessels_df.loc[v, "unloading_rate"]
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    loading_times_df[v] = loading_times_df.index.to_series().map(d)
                except KeyError:
                    print(f"Failed to find rate for vessel {v} and node {node}")
        return loading_times_df

    def get_vessel_transport_times(self, relevant_vessels: List[str],
                                   relevant_nodes: List[str],
                                   time_period_length: int) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        relevant_locations = list(int(self.nlm.get_location_from_node(node)) for node in relevant_nodes)
        distances_df = self.distances_df.loc[list(set(relevant_locations)), list(set(relevant_locations))]
        idx = pd.MultiIndex.from_product([relevant_nodes, relevant_vessels], names=['node', 'vessel'])
        transport_times_df = pd.DataFrame(0, index=idx, columns=relevant_nodes)
        for v in relevant_vessels:
            speed = vessels_df.loc[v, 'knop_avg']
            for orig in relevant_nodes:
                for dest in relevant_nodes:
                    hours = int(math.floor(distances_df.loc[int(self.nlm.get_location_from_node(orig)),
                                                            int(self.nlm.get_location_from_node(dest))]) * speed)
                    transport_times_df.loc[(orig, v), dest] = _hours_to_time_periods(hours, time_period_length)
        # TODO: Must implement reading of 3D table in read_problem_data
        return transport_times_df


    def get_inventory_capacity(self) -> pd.DataFrame:
        data = [(factory_node, self.factories_df.loc[location, 'inventory_capacity'])
                for factory_node, location in self.nlm.get_factory_items()]
        df = pd.DataFrame(data)
        df.columns = ['factory', 'capacity']
        df = df.set_index('factory')
        return df

    def get_inventory_cost(self) -> pd.DataFrame:
        data = [(factory_node, self.factories_df.loc[location, 'inventory_cost'])
                for factory_node, location in self.nlm.get_factory_items()]
        df = pd.DataFrame(data)
        df.columns = ['factory', 'unit_cost']
        df = df.set_index('factory')
        return df

    def get_production_start_cost(self, no_products) -> pd.DataFrame:
        data = {'product': [f'p_{i}' for i in range(no_products)]}
        for factory_node, factory_location in self.nlm.get_factory_items():
            start_cost = self.factories_df.loc[factory_location, 'production_start_cost']
            data.update({factory_node: [start_cost] * no_products})
        df = pd.DataFrame(data)
        df = df.set_index('product')
        return df

    def get_product_groups(self, no_product_groups: int, no_products: int) -> pd.DataFrame:
        """Draws a random product group for each product"""
        data = [(f'p_{i}', f'pg_{random.randint(0, no_product_groups - 1)}') for i in range(no_products)]
        df = pd.DataFrame(data)
        df.columns = ['product', 'product_group']
        df = df.set_index('product')
        return df

    def get_production_lines_for_factory(self) -> pd.DataFrame:
        prod_lines_for_factory = {}
        line_factory_records = []
        i = 0
        for factory_node, factory_location in self.nlm.get_factory_items():
            i_end = i + self.factories_df.loc[factory_location, 'production_lines']
            line_factory_records += [(f'l_{i}', factory_node) for i in range(i, i_end)]
            prod_lines_for_factory[factory_node] = [f'l_{i}' for i in range(i, i_end)]
            i = i_end

        self.prod_lines_for_factory = prod_lines_for_factory  # This is for later use by other functions
        df = pd.DataFrame(line_factory_records)
        df.columns = ['production_line', 'factory']
        df = df.set_index('production_line')
        return df

    def get_production_line_min_time(self, time_period_length: int, no_products: int) -> pd.DataFrame:
        data = {'product': [f'p_{i}' for i in range(no_products)]}
        for factory, prod_lines in self.prod_lines_for_factory.items():
            if prod_lines:
                min_size = self.factories_df.loc[self.nlm.get_location_from_node(factory), 'production_series_min_size']
                prod_cap = self.factories_df.loc[self.nlm.get_location_from_node(factory), 'production_capacity']
                min_hours = min_size / prod_cap
                min_time_periods = math.ceil(min_hours / time_period_length)
                data.update({line: [min_time_periods] * no_products for line in prod_lines})
        df = pd.DataFrame(data)
        df = df.set_index('product')
        return df

    def get_production_max_capacity(self, no_products: int) -> pd.DataFrame:
        data = {'product': [f'p_{i}' for i in range(no_products)]}
        for factory, prod_lines in self.prod_lines_for_factory.items():
            if prod_lines:
                prod_cap = self.factories_df.loc[self.nlm.get_location_from_node(factory), 'production_capacity']
                data.update({line: [prod_cap] * no_products for line in prod_lines})
        df = pd.DataFrame(data)
        df = df.set_index('product')
        return df

    def get_vessel_name(self, vessel_id: str) -> str:
        return self.vessels_df.loc[vessel_id, "vessel_name"]

    def write_transport_times_to_file(self, relevant_vessels: List[str], relevant_nodes: List[str],
                                      time_period_length: int) -> None:
        transport_times_df = self.get_vessel_transport_times(relevant_vessels, relevant_nodes, time_period_length)
        transport_times_df.to_excel(self.excel_writer, sheet_name="transport_time", startrow=1)
        # startrow=1 because of skiprows in read

    def write_loading_times_to_file(self, relevant_vessels: List[str], relevant_nodes: List[str],
                                    time_period_length: int) -> None:
        loading_times_df = self.get_all_loading_times(relevant_vessels, relevant_nodes, time_period_length)
        loading_times_df.to_excel(self.excel_writer, sheet_name="loading_unloading_time", startrow=1)
        # startrow=1 because of skiprows in read

    def write_demands_to_file(self, orders: pd.DataFrame) -> None:
        orders.index = [f'o_{i}' for i in range(len(orders))]
        orders.index.name = 'order'
        orders.to_excel(self.excel_writer, sheet_name="demand", startrow=1)

    def write_time_windows_to_file(self, time_periods: int, tw_length: int, earliest_tw_start: int) -> None:
        tw_df = self.generate_random_time_windows(time_periods, tw_length, earliest_tw_start)
        tw_df.to_excel(self.excel_writer, sheet_name="time_windows_for_order", startrow=1)

    def write_initial_inventory_to_file(self, factory_level: float, ext_depot_level: float,
                                        orders: pd.DataFrame) -> None:
        df = self.generate_initial_inventory(factory_level, ext_depot_level, orders)
        df.to_excel(self.excel_writer, sheet_name="initial_inventory", startrow=1)

    def write_inventory_capacity_to_file(self):
        df = self.get_inventory_capacity()
        df.to_excel(self.excel_writer, sheet_name="inventory_capacity", startrow=1)

    def write_inventory_cost_to_file(self):
        df = self.get_inventory_cost()
        df.to_excel(self.excel_writer, sheet_name="inventory_cost", startrow=1)

    def write_product_group_to_file(self, no_product_groups: int, no_products: int) -> None:
        df = self.get_product_groups(no_product_groups, no_products)
        df.to_excel(self.excel_writer, sheet_name="product_group", startrow=1)

    def write_production_start_cost_to_file(self, no_products: int) -> None:
        df = self.get_production_start_cost(no_products)
        df.to_excel(self.excel_writer, sheet_name="production_start_cost", startrow=1)

    def write_production_max_capacity_to_file(self, no_products: int) -> None:
        df = self.get_production_max_capacity(no_products)
        df.to_excel(self.excel_writer, sheet_name="production_max_capacity", startrow=1)

    def write_production_lines_for_factory_to_file(self) -> None:
        df = self.get_production_lines_for_factory()
        df.to_excel(self.excel_writer, sheet_name="production_lines_for_factory", startrow=1)

    def write_prod_line_min_time_to_file(self, time_period_length: int, no_products: int) -> None:
        df = self.get_production_line_min_time(time_period_length, no_products)
        df.to_excel(self.excel_writer, sheet_name="prod_line_min_time", startrow=1)

    def write_factory_max_vessel_destination_to_file(self, vessels: List[str]) -> None:
        df = self.get_factory_max_vessel_destination(vessels)
        df.to_excel(self.excel_writer, sheet_name="factory_max_vessel_destination", startrow=1)

    def write_factory_max_vessel_loading_to_file(self, quay_activity_level: float, time_periods: int,
                                                 time_period_length: int) -> None:
        df = self.get_factory_max_vessel_loading(quay_activity_level, time_periods, time_period_length)
        df.to_excel(self.excel_writer, sheet_name="factory_max_vessel_loading", startrow=1)

    def _is_external_depot_node(self, factory_node):
        return self.factories_df.loc[self.nlm.get_location_from_node(factory_node), 'production_lines'] == 0


if __name__ == '__main__':
    tdg = TestDataGenerator()
    tdg.write_test_instance_to_file(out_filepath="../../data/testoutputfile.xlsx",
                                    vessels=["v_1", "v_2", "v_3", "v_4"],
                                    factory_locations=["2022", "482", '2015'],
                                    no_orders=20,
                                    no_products=8,
                                    no_product_groups=3,
                                    factory_level=0.3,
                                    ext_depot_level=0.4,
                                    quay_activity_level=0.1,
                                    orders_from_company=None,
                                    time_periods=100,
                                    time_period_length=2)

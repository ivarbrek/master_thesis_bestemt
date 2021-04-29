from __future__ import annotations
import math
from typing import List, Dict
import pandas as pd
from src.generate_data import orders_generator
import random


def _hours_to_time_periods(hours: float, time_period_length: float) -> int:
    if hours % time_period_length < 0.25:  # TODO: Is this OK? Rounding down if less than 0.25 over time period threshold
        time_floor = math.floor(hours / time_period_length)
        time = time_floor if time_floor > 0 else 1
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

    @staticmethod
    def node_is_factory(node_id: str) -> bool:
        return node_id.startswith("f")

    @staticmethod
    def node_is_order(node_id: str) -> bool:
        return node_id.startswith("o")


class TestDataGenerator:
    excel_writer: pd.ExcelWriter
    nlm: NodeLocationMapping

    def __init__(self):
        self.vessels_df = pd.read_excel("../../data/vessels.xlsx", sheet_name="vessels", index_col=[0])
        self.distances_df = pd.read_excel("../../data/distance_matrix.xlsx", sheet_name="distances", index_col=[0])

    def write_test_instance_to_file(self, out_filepath: str,
                                    vessels: List[str],
                                    factory_locations: List[str],
                                    time_periods: int,
                                    time_period_length: int,
                                    no_orders: int,
                                    no_products: int,
                                    orders_from_company: str = None):
        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='a',
                                           options={'strings_to_formulas': False})

        orders = orders_generator.sample_orders_df(no_orders, company=orders_from_company, no_products=no_products)
        order_locations = list(orders.index)
        self.nlm = NodeLocationMapping(order_locations, factory_locations)

        # Write sheets to file
        self.write_demands_to_file(orders)
        self.write_transport_times_to_file(relevant_vessels=vessels, relevant_nodes=self.nlm.get_nodes(),
                                           time_period_length=time_period_length)
        self.write_transport_costs_to_file(relevant_vessels=vessels)
        self.write_loading_times_to_file(relevant_vessels=vessels, relevant_nodes=self.nlm.get_nodes(),
                                         time_period_length=time_period_length)
        self.write_time_windows_to_file(time_periods, tw_length=20, earliest_tw_start=5)
        self.excel_writer.close()

    def generate_random_time_windows(self, time_periods: int, tw_length: int, earliest_tw_start: int) -> pd.DataFrame:
        data = []
        for order in self.nlm.get_order_nodes():
            tw_start = random.randint(earliest_tw_start, time_periods - tw_length)  # pick random tw start
            tw_end = tw_start + tw_length
            data.append((order, tw_start, tw_end))
        df = pd.DataFrame(data)
        df.columns = ['orders', 'tw_start', 'tw_end']
        df = df.set_index('orders')
        return df

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
            speed = vessels_df.loc[v, 'knop_avg'] * 1.85  # 1 knot = 1.85 km/h
            for orig in relevant_nodes:
                for dest in relevant_nodes:
                    hours = (int(math.floor(distances_df.loc[int(self.nlm.get_location_from_node(orig)),
                                                             int(self.nlm.get_location_from_node(dest))]) / 1000) / speed)
                    transport_times_df.loc[(orig, v), dest] = 0 if orig == dest else _hours_to_time_periods(hours, time_period_length)
        return transport_times_df

    def get_vessel_transport_costs(self, relevant_vessels: List[str]) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        transport_costs_df: pd.DataFrame = pd.DataFrame(relevant_vessels, columns=["vessel"]).set_index("vessel", drop=True)
        d = {}
        for v in relevant_vessels:
            d[v] = vessels_df.loc[v, 'unit_transport_cost']
        transport_costs_df['unit_transport_cost'] = transport_costs_df.index.to_series().map(d)
        return transport_costs_df

    def get_vessel_name(self, vessel_id: str) -> str:
        return self.vessels_df.loc[vessel_id, "vessel_name"]

    def write_time_windows_to_file(self, time_periods: int, tw_length: int, earliest_tw_start: int) -> None:
        tw_df = self.generate_random_time_windows(time_periods, tw_length, earliest_tw_start)
        tw_df.to_excel(self.excel_writer, sheet_name="time_windows_for_order", startrow=1)

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
        orders.to_excel(self.excel_writer, sheet_name="demand", startrow=1)

    def write_transport_costs_to_file(self, relevant_vessels: List[str]) -> None:
        transport_costs_df = self.get_vessel_transport_costs(relevant_vessels=relevant_vessels)
        transport_costs_df.to_excel(self.excel_writer, sheet_name="transport_cost", startrow=1)


if __name__ == '__main__':
    tdg = TestDataGenerator()
    tdg.write_test_instance_to_file(out_filepath="../../data/testoutputfile.xlsx",
                                    vessels=["v_1", "v_2", "v_3"],
                                    factory_locations=["2022", "482"],
                                    no_orders=12,
                                    no_products=8,
                                    orders_from_company=None,
                                    time_periods=100,
                                    time_period_length=2)

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Set, Tuple
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
    location_node_mapping: Dict[str, List[str]] = {}

    def __init__(self, order_locations: List[str], factory_locations: List[str]):
        """
        :param order_locations: location id for aquaculture farm
        :param factory_locations: location id for aquaculture farm
        """

        for i in range(len(order_locations)):
            self.node_location_mapping["o_" + str(i)] = order_locations[i]

            if order_locations[i] in self.location_node_mapping.keys():
                self.location_node_mapping[order_locations[i]].append("o_" + str(i))
            else:
                self.location_node_mapping[order_locations[i]] = ["o_" + str(i)]

        for i in range(len(factory_locations)):
            self.node_location_mapping["f_" + str(i)] = factory_locations[i]

            if factory_locations[i] in self.location_node_mapping.keys():
                self.location_node_mapping[factory_locations[i]].append("f_" + str(i))
            else:
                self.location_node_mapping[factory_locations[i]] = ["f_" + str(i)]

    def get_location_from_node(self, node_id: str) -> str:
        return self.node_location_mapping[node_id]

    def get_nodes(self) -> List[str]:
        return list(self.node_location_mapping.keys())

    def get_locations(self) -> List[str]:
        return list(self.node_location_mapping.values())

    def get_order_locations(self) -> List[str]:
        return list(loc for loc in self.location_node_mapping.keys()
                    if self.node_is_order(self.location_node_mapping[loc][0]))

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
        self.factories_df = pd.read_excel("../../data/factories.xlsx", sheet_name="factories", skiprows=[0])
        self.factories_df['location_id'] = self.factories_df['location_id'].astype('str')

    def write_test_instance_to_file(self, out_filepath: str,
                                    vessel_names: List[str],
                                    factory_locations: List[str],
                                    time_periods: int,
                                    time_period_length: int,
                                    hours_production_stop: int,
                                    no_orders: int,
                                    no_products: int,
                                    share_red_nodes: float,
                                    radius_red: int,
                                    radius_yellow: int,
                                    share_bag_locations: float,
                                    share_small_fjord_locations: float,
                                    small_fjord_radius: int,
                                    delivery_delay_unit_penalty: int,
                                    min_wait_if_sick_hours: int,
                                    orders_from_company: str = None,
                                    ):

        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w',
                                           options={'strings_to_formulas': False})

        orders = orders_generator.sample_orders_df(no_orders, company=orders_from_company, no_products=no_products)
        order_locations = list(orders.index)
        self.nlm = NodeLocationMapping(order_locations, factory_locations)

        vessels: List[str] = [self.vessels_df.index[self.vessels_df['vessel_name'] == v].tolist()[0]
                              for v in vessel_names]

        # Write sheets to file
        self.write_demands_to_file(orders)
        self.write_transport_times_to_file(relevant_vessels=vessels,
                                           time_period_length=time_period_length)
        self.write_transport_costs_to_file(relevant_vessels=vessels)
        self.write_loading_times_to_file(relevant_vessels=vessels, orders=orders,
                                         time_period_length=time_period_length)
        self.write_time_windows_to_file(time_periods, tw_length=20, earliest_tw_start=5)
        self.write_vessel_capacities_to_file(relevant_vessels=vessels)
        self.write_vessel_availability_to_file(relevant_vessels=vessels)
        self.write_order_zones_to_file(share_red=share_red_nodes, radius_red=radius_red, radius_yellow=radius_yellow)
        self.write_nodes_for_vessels_to_file(relevant_vessels=vessels, share_bag_locations=share_bag_locations,
                                             share_small_fjord_locations=share_small_fjord_locations,
                                             small_fjord_radius=small_fjord_radius)
        self.write_production_stop_to_file(time_period_length=time_period_length, hours_stop=hours_production_stop,
                                           time_periods=time_periods)
        self.write_key_values_to_file(delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                                      min_wait_if_sick_hours=min_wait_if_sick_hours,
                                      time_period_length=time_period_length)
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

    def get_all_loading_times(self, relevant_vessels: List[str], orders: pd.DataFrame,
                              time_period_length: int) -> pd.DataFrame:
        loading_times_df = pd.DataFrame(self.nlm.get_nodes(), columns=["node"]).set_index("node", drop=True)
        for v in relevant_vessels:
            d = {}
            for node in loading_times_df.index:
                if self.nlm.node_is_factory(node):
                    quantity = self.vessels_df.loc[v, 'max_ton']
                    hours = quantity / self.vessels_df.loc[v, "loading_rate"]
                    d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                else:
                    quantity = sum(orders.loc[node, p] for p in orders.columns)
                    hours = quantity / self.vessels_df.loc[v, "unloading_rate"]
                    d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                loading_times_df[v] = loading_times_df.index.to_series().map(d)
        return loading_times_df

    def get_vessel_transport_times(self, relevant_vessels: List[str],
                                   time_period_length: int) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        relevant_locations = list(int(self.nlm.get_location_from_node(node)) for node in self.nlm.get_nodes())
        distances_df = self.distances_df.loc[list(set(relevant_locations)), list(set(relevant_locations))]
        idx = pd.MultiIndex.from_product([self.nlm.get_nodes(), relevant_vessels], names=['node', 'vessel'])
        transport_times_df = pd.DataFrame(0, index=idx, columns=self.nlm.get_nodes())
        for v in relevant_vessels:
            speed = vessels_df.loc[v, 'knop_avg'] * 1.85  # 1 knot = 1.85 km/h
            for orig in self.nlm.get_nodes():
                for dest in self.nlm.get_nodes():
                    hours = (int(math.floor(distances_df.loc[int(self.nlm.get_location_from_node(orig)),
                                                             int(self.nlm.get_location_from_node(
                                                                 dest))]) / 1000) / speed)
                    transport_times_df.loc[(orig, v), dest] = 0 if orig == dest else _hours_to_time_periods(hours,
                                                                                                            time_period_length)
        return transport_times_df

    def get_vessel_capacities(self, relevant_vessels: List[str]) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        vessel_capacities_df: pd.DataFrame = pd.DataFrame(relevant_vessels, columns=["vessel"]).set_index("vessel",
                                                                                                          drop=True)
        d_tons = {}
        d_nprods = {}
        for v in relevant_vessels:
            d_tons[v] = vessels_df.loc[v, 'max_ton']
            d_nprods[v] = vessels_df.loc[v, 'max_nprod']
        vessel_capacities_df['capacity [t]'] = vessel_capacities_df.index.to_series().map(d_tons)
        vessel_capacities_df['capacity [nProd]'] = vessel_capacities_df.index.to_series().map(d_nprods)
        return vessel_capacities_df

    def get_vessel_transport_costs(self, relevant_vessels: List[str]) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        transport_costs_df: pd.DataFrame = pd.DataFrame(relevant_vessels, columns=["vessel"]).set_index("vessel",
                                                                                                        drop=True)
        d = {}
        for v in relevant_vessels:
            d[v] = vessels_df.loc[v, 'unit_transport_cost']
        transport_costs_df['unit_transport_cost'] = transport_costs_df.index.to_series().map(d)
        return transport_costs_df

    def get_vessel_name(self, vessel_id: str) -> str:
        return self.vessels_df.loc[vessel_id, "vessel_name"]

    def get_vessel_availability(self, relevant_vessels: List[str]) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        vessel_availability_df: pd.DataFrame = pd.DataFrame(relevant_vessels, columns=["vessel"]).set_index("vessel",
                                                                                                            drop=True)
        d_time = {}
        d_loc = {}
        for v in relevant_vessels:
            d_time[v] = vessels_df.loc[v, 'availability_time']
            loc = vessels_df.loc[v, 'availability_factoryname']
            d_loc[v] = self.factories_df.loc[self.factories_df['factory_location'] == loc, 'location_id'].iloc[0]
        vessel_availability_df['time_period'] = vessel_availability_df.index.to_series().map(d_time)
        vessel_availability_df['location'] = vessel_availability_df.index.to_series().map(d_loc)
        return vessel_availability_df

    def generate_order_zones(self, share_red: float, radius_red: int, radius_yellow: int) -> pd.DataFrame:
        orders = self.nlm.get_order_locations()
        num_sick = math.floor(len(orders) * share_red)

        first_sick = orders[random.randint(0, len(orders) - 1)]
        sick: Set[str] = {first_sick}
        queue: Set[str] = self.get_nearby_order_locations_set(loc_id=first_sick, radius=radius_red)

        while len(sick) < num_sick:
            queue = queue.difference(sick)  # queue of healthy locations to become sick
            if len(queue) > 0:
                new_sick = queue.pop()
            else:
                new_sick = set(self.nlm.get_order_locations()).difference(sick).pop()
            sick.add(new_sick)
            queue = queue.union(self.get_nearby_order_locations_set(new_sick, radius=radius_red))

        order_zones_df: pd.DataFrame = pd.DataFrame(self.nlm.get_order_nodes(),
                                                    columns=["order_node"]).set_index("order_node", drop=True)

        d_color = {}
        for node_id in self.nlm.get_order_nodes():
            d_color[node_id] = "green"
        for loc_id in sick:
            for nearby_loc in self.get_nearby_order_locations_set(loc_id, radius=radius_yellow):
                for node_id in self.nlm.location_node_mapping[nearby_loc]:
                    d_color[node_id] = "yellow"
        for loc_id in sick:
            for node_id in self.nlm.location_node_mapping[loc_id]:
                d_color[node_id] = "red"

        order_zones_df['zone'] = order_zones_df.index.to_series().map(d_color)
        return order_zones_df

    def get_nearby_order_locations_set(self, loc_id: str, radius: int) -> Set[str]:
        distances = self.distances_df[[int(i) for i in self.nlm.get_order_locations()]].loc[int(loc_id)]
        nearby_nodes: Set[str] = set()
        for id, dist in distances.items():
            if dist <= radius and id != loc_id:
                nearby_nodes.add(str(id))
        return nearby_nodes

    def generate_nodes_for_vessels(self, relevant_vessels: List[str], share_bag_locations: float,
                                   share_small_fjord_locations: float, small_fjord_radius: int) -> pd.DataFrame:
        order_locations = self.nlm.get_order_locations()

        # Default
        d: Dict[Tuple[str, str], int] = {}  # (vessel, node)
        for node in self.nlm.get_nodes():
            for vessel in relevant_vessels:
                d[(vessel, node)] = 1

        # Order nodes requiring bag deliveries (not possible to deliver for silo vessels)
        num_bag = math.floor(len(order_locations) * share_bag_locations)
        bag_location_idxs = random.sample(range(0, len(order_locations)), num_bag)
        bag_nodes = list(set(node
                             for idx in bag_location_idxs
                             for node in self.nlm.location_node_mapping[order_locations[idx]]))

        silo_vessels = [v for v in relevant_vessels if self.vessels_df.loc[v, 'type'] == 'silo']
        for node in bag_nodes:
            for vessel in silo_vessels:
                d[(vessel, node)] = 0

        # Order nodes where large vessels cannot drive (located close to each other)
        num_fjord = math.floor(len(order_locations) * share_small_fjord_locations)
        small_fjord_locs: Set[str] = set()
        loc_candidates: Set[str] = set()

        while len(small_fjord_locs) < num_fjord:
            found = False
            if len(loc_candidates) > 0:
                for _ in range(len(loc_candidates)):
                    loc_id = loc_candidates.pop()
                    add = np.random.choice(np.array([True, False]), p=(np.array([0.5, 0.5])))  # add order with 50% prob
                    if add:
                        found = True
                        break
            if not found:
                loc_id = set(order_locations).difference(small_fjord_locs).pop()
            small_fjord_locs.add(loc_id)
            loc_candidates = self.get_nearby_order_locations_set(loc_id=loc_id, radius=small_fjord_radius)

        avg_vessel_size = sum(self.vessels_df.loc[v, 'max_ton'] for v in relevant_vessels) / len(relevant_vessels)
        for v in relevant_vessels:
            if self.vessels_df.loc[v, 'max_ton'] > avg_vessel_size:
                for loc_id in small_fjord_locs:
                    for node_id in self.nlm.location_node_mapping[loc_id]:
                        d[(v, node_id)] *= 1 if np.random.choice(np.array([True, False]),
                                                                 p=(np.array([0.5, 0.5]))) else 0

        nodes_for_vessels_df: pd.DataFrame = pd.DataFrame(relevant_vessels,
                                                          columns=["vessel"]).set_index("vessel", drop=True)

        for node_id in self.nlm.get_nodes():
            nodes_for_vessels_df[node_id] = [d[(v, node_id)] for v in relevant_vessels]
        return nodes_for_vessels_df

    def generate_production_stops(self, time_period_length: int, hours_stop: int,
                                  time_periods: int) -> pd.DataFrame:
        time_periods_stopped = _hours_to_time_periods(hours_stop, time_period_length)
        d = {(f, t): 1 for f in self.nlm.get_factory_nodes() for t in range(time_periods)}
        for f in self.nlm.get_factory_nodes():
            start_of_stop = random.randint(0, time_periods - 1)
            for t in range(start_of_stop, min(time_periods - 1, start_of_stop + time_periods_stopped)):
                d[(f, t)] = 0
        production_stops_df: pd.DataFrame = pd.DataFrame([i for i in range(time_periods)],
                                                         columns=["time_period"]).set_index("time_period", drop=True)
        for f in self.nlm.get_factory_nodes():
            production_stops_df[f] = [d[(f, t)] for t in range(time_periods)]
        return production_stops_df

    def get_key_values(self, delivery_delay_unit_penalty: int, min_wait_if_sick_hours: int, time_period_length: int) -> pd.DataFrame:
        key_values_df: pd.DataFrame = pd.DataFrame(['min_wait_if_sick' 'external_delivery_unit_penalty'],
                                                   columns=["key"]).set_index("key", drop=True)
        key_values_df['min_wait_if_sick'] = _hours_to_time_periods(min_wait_if_sick_hours, time_period_length)
        key_values_df['external_delivery_unit_penalty'] = delivery_delay_unit_penalty
        return key_values_df

    def write_time_windows_to_file(self, time_periods: int, tw_length: int, earliest_tw_start: int) -> None:
        tw_df = self.generate_random_time_windows(time_periods, tw_length, earliest_tw_start)
        tw_df.to_excel(self.excel_writer, sheet_name="time_windows_for_order", startrow=1)

    def write_transport_times_to_file(self, relevant_vessels: List[str],
                                      time_period_length: int) -> None:
        transport_times_df = self.get_vessel_transport_times(relevant_vessels, time_period_length)
        transport_times_df.to_excel(self.excel_writer, sheet_name="transport_time", startrow=1)

    def write_loading_times_to_file(self, relevant_vessels: List[str], orders: pd.DataFrame,
                                    time_period_length: int) -> None:
        loading_times_df = self.get_all_loading_times(relevant_vessels, orders, time_period_length)
        loading_times_df.to_excel(self.excel_writer, sheet_name="loading_unloading_time", startrow=1)

    def write_demands_to_file(self, orders: pd.DataFrame) -> None:
        orders.index = [f'o_{i}' for i in range(len(orders))]
        orders.to_excel(self.excel_writer, sheet_name="demand", startrow=1)

    def write_transport_costs_to_file(self, relevant_vessels: List[str]) -> None:
        transport_costs_df = self.get_vessel_transport_costs(relevant_vessels=relevant_vessels)
        transport_costs_df.to_excel(self.excel_writer, sheet_name="transport_cost", startrow=1)

    def write_vessel_capacities_to_file(self, relevant_vessels: List[str]) -> None:
        vessel_capacities_df = self.get_vessel_capacities(relevant_vessels=relevant_vessels)
        vessel_capacities_df.to_excel(self.excel_writer, sheet_name="vessel_capacity", startrow=1)

    def write_vessel_availability_to_file(self, relevant_vessels: List[str]) -> None:
        vessel_availability_df = self.get_vessel_availability(relevant_vessels=relevant_vessels)
        vessel_availability_df.to_excel(self.excel_writer, sheet_name="vessel_availability", startrow=1)

    def write_order_zones_to_file(self, share_red: float, radius_red: int, radius_yellow: int) -> None:
        order_zones_df = self.generate_order_zones(share_red=share_red, radius_red=radius_red,
                                                   radius_yellow=radius_yellow)
        order_zones_df.to_excel(self.excel_writer, sheet_name="order_zones", startrow=1)

    def write_nodes_for_vessels_to_file(self, relevant_vessels: List[str], share_bag_locations: float,
                                        share_small_fjord_locations: float,
                                        small_fjord_radius: int) -> None:
        nodes_for_vessels_df = self.generate_nodes_for_vessels(relevant_vessels=relevant_vessels,
                                                               share_bag_locations=share_bag_locations,
                                                               share_small_fjord_locations=share_small_fjord_locations,
                                                               small_fjord_radius=small_fjord_radius)
        nodes_for_vessels_df.to_excel(self.excel_writer, sheet_name="nodes_for_vessel", startrow=1)

    def write_production_stop_to_file(self, time_period_length: int, hours_stop: int, time_periods: int) -> None:
        production_stop_df = self.generate_production_stops(time_period_length=time_period_length,
                                                            hours_stop=hours_stop,
                                                            time_periods=time_periods)
        production_stop_df.to_excel(self.excel_writer, sheet_name="production_stop", startrow=1)

    def write_key_values_to_file(self, delivery_delay_unit_penalty: int, min_wait_if_sick_hours: int, time_period_length: int):
        key_values_df = self.get_key_values(delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                                            min_wait_if_sick_hours=min_wait_if_sick_hours,
                                            time_period_length=time_period_length)
        key_values_df.to_excel(self.excel_writer, sheet_name="key_values", startrow=1)

if __name__ == '__main__':
    tdg = TestDataGenerator()
    tdg.write_test_instance_to_file(out_filepath="../../data/testoutputfile.xlsx",
                                    vessel_names=["Ripnes", "Vågsund", "Pirholm"],
                                    factory_locations=["2022", "482"],
                                    no_orders=12,
                                    no_products=8,
                                    orders_from_company=None,
                                    time_periods=100,
                                    time_period_length=2,
                                    hours_production_stop=12,
                                    share_red_nodes=0.2,
                                    radius_red=50000,
                                    radius_yellow=300000,
                                    share_bag_locations=0.25,
                                    share_small_fjord_locations=0.25,
                                    small_fjord_radius=50000,
                                    min_wait_if_sick_hours=12,
                                    delivery_delay_unit_penalty=1000)

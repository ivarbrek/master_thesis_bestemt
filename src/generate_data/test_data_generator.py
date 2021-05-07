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
        self.distances_df = pd.read_excel("../../data/distance_matrix.xlsx", sheet_name="distances", index_col=[0])
        self.factories_df = pd.read_excel("../../data/factories.xlsx", sheet_name="factories", index_col=[0],
                                          skiprows=[0])
        self.factories_df.index = self.factories_df.index.astype('str')

    def write_test_instance_to_file(self, out_filepath: str,
                                    vessel_names: List[str],
                                    factory_locations: List[str],
                                    time_periods: int,
                                    time_period_length: int,
                                    hours_production_stop: int,
                                    no_orders: int,
                                    no_products: int,
                                    no_product_groups: int,
                                    factory_level: float,
                                    ext_depot_level: float,
                                    quay_activity_level: float,
                                    share_red_nodes: float,
                                    radius_red: int,
                                    radius_yellow: int,
                                    share_bag_locations: float,
                                    share_small_fjord_locations: float,
                                    share_time_periods_vessel_availability: float,
                                    small_fjord_radius: int,
                                    delivery_delay_unit_penalty: int,
                                    min_wait_if_sick_hours: int,
                                    orders_from_company: str = None,
                                    ):

        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w',
                                           options={'strings_to_formulas': False})

        # Set orders and factories
        orders = orders_generator.sample_orders_df(no_orders, company=orders_from_company, no_products=no_products)
        order_locations = list(orders.index)
        self.nlm = NodeLocationMapping(order_locations, factory_locations)
        self.factories_df = self.factories_df.loc[factory_locations]
        vessels: List[str] = [self.vessels_df.index[self.vessels_df['vessel_name'] == v].tolist()[0]
                              for v in vessel_names]

        # Write sheets to file
        self.write_demands_to_file(orders)
        self.write_transport_times_to_file(relevant_vessels=vessels,
                                           time_period_length=time_period_length)
        self.write_transport_costs_to_file(relevant_vessels=vessels)
        self.write_loading_times_to_file(relevant_vessels=vessels, orders=orders,
                                         time_period_length=time_period_length)
        self.write_time_windows_to_file(time_periods, tw_length=120//time_period_length, earliest_tw_start=5)
        self.write_factory_max_vessel_destination_to_file(vessels)
        self.write_factory_max_vessel_loading_to_file(quay_activity_level, time_periods, time_period_length)
        self.write_initial_inventory_to_file(factory_level, ext_depot_level, orders)
        self.write_inventory_capacity_to_file()
        self.write_inventory_cost_to_file()
        self.write_production_start_cost_to_file(no_products)
        self.write_product_group_to_file(no_product_groups, no_products)
        self.write_production_lines_for_factory_to_file()
        self.write_production_max_capacity_to_file(no_products, time_period_length)
        self.write_prod_line_min_time_to_file(time_period_length, no_products)
        self.write_vessel_availability_to_file(vessels, share_time_periods_vessel_availability, time_periods)
        self.write_vessel_capacities_to_file(relevant_vessels=vessels)
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
        df.columns = ['order', 'tw_start', 'tw_end']
        df = df.set_index('order')
        self.time_windows_df = df
        return df

    def generate_initial_inventory(self, factory_level: float, ext_depot_level: float, orders: pd.DataFrame):
        # TODO: Change ext_depot_usage
        """
        :param factory_level: percentage of total demand stored on factories (excl. external depots)
        :param ext_depot_level: percentage of inventory capacity stored on external depots
        :param orders: DataFrame with orders
        :return: DataFrame with initial inventories
        """
        no_products = len(orders.columns)
        orders = orders.join(self.time_windows_df)
        orders = orders.sort_values('tw_start')
        orders = orders.drop(columns=['tw_start', 'tw_end'])
        # Note: In this function we distinguish factories form external depots
        factories = [node for node in self.nlm.get_factory_nodes() if not self._is_external_depot_node(node)]
        ext_depots = [node for node in self.nlm.get_factory_nodes() if self._is_external_depot_node(node)]
        inv_capacities = {node: self.factories_df.loc[self.nlm.get_location_from_node(node), 'inventory_capacity']
                          for node in factories + ext_depots}
        total_demand = orders.sum().sum()

        # Initialize inventory with small values
        init_inventory = {node: np.zeros(no_products)
                          for node in factories}

        # Add a proportion of earliest orders until desired factory_level is reached
        for idx, row in orders.iterrows():
            chosen_factory = random.choice(factories)
            init_inventory[chosen_factory] += (row.to_numpy() * 0.8).astype('int64')
            if sum(inventory.sum() for factory, inventory in init_inventory.items()) > factory_level * total_demand:
                break

        # Check that capacity is not violated
        for factory in factories:
            assert init_inventory[factory].sum() <= inv_capacities[factory], "Init. inventory > capacity "

        # Add inventory for external depots
        for depot in ext_depots:
            # Linear decrease in initial inventory for products,
            # where p_0 gets no_products times more than p_n,
            # and s.t. total inventory == ext_depot_level * inv_capacities[depot]
            init_inventory[depot] = [int(2 * ext_depot_level/(no_products + 1) *
                                         (no_products - i)/no_products * inv_capacities[depot])
                                     for i in range(no_products)]

        orders = pd.DataFrame(init_inventory)
        orders.index = [f'p_{i}' for i in range(no_products)]
        return orders.transpose()

    def get_factory_max_vessel_destination(self, vessels: List[str]) -> pd.DataFrame:
        no_vessels = len(vessels)
        no_factories = len(self.nlm.get_factory_nodes())

        # no_vessels // no_factories + 1 for factories, 0 for external depots
        data = [(node, (no_vessels // no_factories + 1)) for node in self.nlm.get_factory_nodes()]
        df = pd.DataFrame(data)
        df.columns = ['factory', 'vessel_number']
        df = df.set_index('factory')
        return df

    def get_factory_max_vessel_loading(self, quay_activity_level: float, time_periods: int,
                                       time_period_length: int) -> pd.DataFrame:
        # Note: In this function we distinguish factories form external depots
        factories = [node for node in self.nlm.get_factory_nodes() if not self._is_external_depot_node(node)]
        ext_depots = [node for node in self.nlm.get_factory_nodes() if self._is_external_depot_node(node)]

        # "rm" denotes raw material
        rm_loading_time = 12   # 12 hours
        rm_loading_periods = rm_loading_time // time_period_length
        rm_loading_insert_count = math.ceil(time_periods * time_period_length * quay_activity_level / rm_loading_time)

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

        # Assume ext. depots always have one spot for loading
        for ext_dep in ext_depots:
            max_vessels_loading.update({ext_dep: [1] * time_periods})

        return pd.DataFrame(max_vessels_loading)

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

    def get_production_max_capacity(self, no_products: int, time_period_length: int) -> pd.DataFrame:
        data = {'product': [f'p_{i}' for i in range(no_products)]}
        for factory, prod_lines in self.prod_lines_for_factory.items():
            if prod_lines:
                prod_cap_hour = self.factories_df.loc[self.nlm.get_location_from_node(factory), 'production_capacity']
                data.update({line: [prod_cap_hour * time_period_length] * no_products for line in prod_lines})
        df = pd.DataFrame(data)
        df = df.set_index('product')
        return df

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

    def get_vessel_availability(self, relevant_vessels: List[str], share_time: float,
                                time_periods: int) -> pd.DataFrame:
        """
        :param relevant_vessels: relevant vessels
        :param share_time: Availability time period is drawn random from {0, share_time * time_periods}
        :param time_periods: number of time periods
        :return:
        """
        factories = self.nlm.get_factory_nodes()
        data = []
        factory_assignments = {f: 0 for f in factories}
        max_vessels_destination_df = self.get_factory_max_vessel_destination(relevant_vessels)
        for vessel in relevant_vessels:
            available_factories = [f for f in factories
                                   if factory_assignments[f] < max_vessels_destination_df.loc[f, 'vessel_number']]
            chosen_factory = random.choice(available_factories)
            chosen_time_period = int(random.triangular(0, int(share_time * time_periods), 0))
            data.append((vessel, chosen_time_period, chosen_factory))
        df = pd.DataFrame(data)
        df.columns = ['vessel', 'time_period', 'location']
        df = df.set_index('vessel')
        return df

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

    def get_key_values(self, delivery_delay_unit_penalty: int, min_wait_if_sick_hours: int,
                       time_period_length: int) -> pd.DataFrame:
        key_values = []
        key_values.append(('min_wait_if_sick', _hours_to_time_periods(min_wait_if_sick_hours, time_period_length)))
        key_values.append(('external_delivery_unit_penalty', delivery_delay_unit_penalty))
        df = pd.DataFrame(key_values)
        df.columns = ['key', 'value']
        df = df.set_index('key')
        return df

    def write_transport_times_to_file(self, relevant_vessels: List[str], time_period_length: int) -> None:
        transport_times_df = self.get_vessel_transport_times(relevant_vessels, time_period_length)
        transport_times_df.to_excel(self.excel_writer, sheet_name="transport_time", startrow=1)

    def write_loading_times_to_file(self, relevant_vessels: List[str], orders: pd.DataFrame,
                                    time_period_length: int) -> None:
        loading_times_df = self.get_all_loading_times(relevant_vessels, orders, time_period_length)
        loading_times_df.to_excel(self.excel_writer, sheet_name="loading_unloading_time", startrow=1)

    def write_demands_to_file(self, orders: pd.DataFrame) -> None:
        orders.index = [f'o_{i}' for i in range(len(orders))]
        orders.index.name = 'order'
        orders.to_excel(self.excel_writer, sheet_name="demand", startrow=1)

    def write_transport_costs_to_file(self, relevant_vessels: List[str]) -> None:
        transport_costs_df = self.get_vessel_transport_costs(relevant_vessels=relevant_vessels)
        transport_costs_df.to_excel(self.excel_writer, sheet_name="transport_cost", startrow=1)

    def write_vessel_capacities_to_file(self, relevant_vessels: List[str]) -> None:
        vessel_capacities_df = self.get_vessel_capacities(relevant_vessels=relevant_vessels)
        vessel_capacities_df.to_excel(self.excel_writer, sheet_name="vessel_capacity", startrow=1)

    def write_vessel_availability_to_file(self, relevant_vessels: List[str], share_time: float,
                                          time_periods: int) -> None:
        vessel_availability_df = self.get_vessel_availability(relevant_vessels, share_time, time_periods)
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

    def write_production_max_capacity_to_file(self, no_products: int, time_period_length: int) -> None:
        df = self.get_production_max_capacity(no_products, time_period_length)
        df.to_excel(self.excel_writer, sheet_name="production_max_capacity", startrow=1)

    def write_production_lines_for_factory_to_file(self) -> None:
        df = self.get_production_lines_for_factory()
        df.to_excel(self.excel_writer, sheet_name="production_lines_for_factory", startrow=1)

    def write_prod_line_min_time_to_file(self, time_period_length: int, no_products: int) -> None:
        df = self.get_production_line_min_time(time_period_length, no_products)
        df.to_excel(self.excel_writer, sheet_name="production_line_min_time", startrow=1)

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
    tdg.write_test_instance_to_file(out_filepath="../../data/input_data/gurobi_testing/f1-v3-o10-t168-tw5.xlsx",
                                    # Input parameters varying:
                                    vessel_names=["Borgenfjord", "Høydal", "Nyksund"],
                                    factory_locations=["2016"],  # Valsneset
                                    orders_from_company="Mowi Feed AS",
                                    no_orders=60,
                                    factory_level=0.2,
                                    time_periods=168,
                                    # Input parameters kept constant:
                                    ext_depot_level=0.1,  # no external depots in computational study instances
                                    time_period_length=2,
                                    no_products=10,
                                    no_product_groups=4,
                                    quay_activity_level=0.1,
                                    hours_production_stop=12,
                                    share_red_nodes=0.1,
                                    radius_red=10000,
                                    radius_yellow=30000,
                                    share_bag_locations=0.2,
                                    share_small_fjord_locations=0.05,
                                    share_time_periods_vessel_availability=0.5,
                                    small_fjord_radius=50000,
                                    min_wait_if_sick_hours=12,
                                    delivery_delay_unit_penalty=10000)  # not used in read_problem_data anymore



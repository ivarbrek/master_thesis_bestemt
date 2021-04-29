import math
from typing import List, Dict
import pandas as pd


def _hours_to_time_periods(hours: float, time_period_length: float) -> int:
    if hours % time_period_length < 0.25:  # TODO: Is this OK? Rounding down if less than 0.25 over time period threshold
        time_floor = math.floor(hours / time_period_length)
        time = time_floor if time_floor > 0 else 1
    else:
        time = math.ceil(hours / time_period_length)
    return time


class LocationNodeMapping:
    location_node_mapping: Dict[str, str] = {}
    node_location_mapping: Dict[str, str] = {}

    def __init__(self,
                 order_locations: List[str],
                 factory_locations: List[str]):
        for i in range(len(order_locations)):
            self.location_node_mapping[order_locations[i]] = "o_" + str(i+1)
            self.node_location_mapping["o_" + str(i+1)] = order_locations[i]
        for i in range(len(factory_locations)):
            self.location_node_mapping[factory_locations[i]] = "f_" + str(i+1)
            self.node_location_mapping["f_" + str(i+1)] = factory_locations[i]

    def get_node_from_location(self, loc_id: str) -> str:
        return self.location_node_mapping[loc_id]

    def get_location_from_node(self, node_id: str) -> str:
        return self.node_location_mapping[node_id]

    def get_all_nodes(self) -> List[str]:
        return list(n for n in self.node_location_mapping.keys())

    def get_all_locations(self) -> List[str]:
        return list(l for l in self.location_node_mapping.keys())

    def is_factory(self, id: str) -> bool:
        if id in self.get_all_locations():
            return self.get_node_from_location(id).startswith("f")
        return id.startswith("f")

    def is_order(self, id: str) -> bool:
        if id in self.get_all_locations():
            return self.get_node_from_location(id).startswith("o")
        return id.startswith("o")


class TestDataGenerator:
    excel_writer: pd.ExcelWriter

    def __init__(self):
        self.vessels_df = pd.read_excel("../data/vessels.xlsx", sheet_name="vessels", index_col=[0])
        self.distances_df = pd.read_excel("../data/distance_matrix.xlsx", sheet_name="distances", index_col=[0])

    def write_test_instance_to_file(self, lnm: LocationNodeMapping, out_filepath: str,
                                    relevant_vessels: List[str], time_period_length: int):
        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', options={'strings_to_formulas': False})

        # Write sheets to file
        self.write_transport_times_to_file(lnm=lnm, out_sheetname="transport_time",
                                           relevant_vessels=relevant_vessels, relevant_nodes=lnm.get_all_nodes(),
                                           time_period_length=time_period_length)
        self.write_loading_times_to_file(lnm=lnm, out_sheetname="loading_unloading_time",
                                         relevant_vessels=relevant_vessels, relevant_nodes=lnm.get_all_nodes(),
                                         time_period_length=time_period_length)
        self.write_transport_costs_to_file(relevant_vessels=relevant_vessels)

        self.excel_writer.save()
        self.excel_writer.close()

    def get_all_loading_times(self, lnm: LocationNodeMapping, relevant_vessels: List[str], relevant_nodes: List[str],
                              time_period_length: int) -> pd.DataFrame:
        loading_times_df = pd.DataFrame(relevant_nodes, columns=["node"]).set_index("node", drop=True)
        for v in relevant_vessels:
            d = {}
            for node in loading_times_df.index:
                try:
                    if lnm.is_factory(node):
                        hours = 100 / self.vessels_df.loc[v, "loading_rate"]  # TODO: Fix quantity read (in elif also)
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    else:
                        hours = 100 / self.vessels_df.loc[v, "unloading_rate"]
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    loading_times_df[v] = loading_times_df.index.to_series().map(d)
                except KeyError:
                    print(f"Failed to find rate for vessel {v} and node {node}")
        return loading_times_df

    def get_vessel_transport_times(self, lnm: LocationNodeMapping, relevant_vessels: List[str],
                                   relevant_nodes: List[str],
                                   time_period_length: int) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        relevant_locations = list(int(lnm.get_location_from_node(node)) for node in relevant_nodes)
        distances_df = self.distances_df.loc[relevant_locations, relevant_locations]

        idx = pd.MultiIndex.from_product([relevant_nodes, relevant_vessels], names=['node', 'vessel'])
        transport_times_df = pd.DataFrame(0, index=idx, columns=relevant_nodes)

        for v in relevant_vessels:
            speed = vessels_df.loc[v, 'knop_avg'] * 1.85  # 1 knot = 1.85 km/h
            for orig in relevant_nodes:
                for dest in relevant_nodes:
                    hours = (int(math.floor(distances_df.loc[int(lnm.get_location_from_node(orig)),
                                                            int(lnm.get_location_from_node(dest))]) / 1000) / speed)
                    transport_times_df.loc[(orig, v), dest] = _hours_to_time_periods(hours=hours,
                                                                                     time_period_length=time_period_length)
        return transport_times_df

    def get_vessel_transport_costs(self, relevant_vessels: List[str]) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        transport_costs_df = pd.DataFrame(relevant_vessels, columns=["vessel"]).set_index("vessel", drop=True)
        d = {}
        for v in relevant_vessels:
            d[v] = vessels_df.loc[v, 'unit_transport_cost']
        transport_costs_df['unit_transport_cost'] = transport_costs_df.index.to_series().map(d)
        return transport_costs_df

    def get_vessel_name(self, vessel_id: str) -> str:
        return self.vessels_df.loc[vessel_id, "vessel_name"]

    def write_transport_times_to_file(self, lnm: LocationNodeMapping, out_sheetname: str, relevant_vessels: List[str],
                                      relevant_nodes: List[str],
                                      time_period_length: int) -> None:
        transport_times_df = self.get_vessel_transport_times(lnm, relevant_vessels, relevant_nodes, time_period_length)
        transport_times_df.to_excel(self.excel_writer, sheet_name=out_sheetname,
                                    startrow=1)  # startrow=1 because of skiprows in read

    def write_loading_times_to_file(self, lnm: LocationNodeMapping, out_sheetname: str, relevant_vessels: List[str],
                                    relevant_nodes: List[str],
                                    time_period_length: int) -> None:
        loading_times_df = self.get_all_loading_times(lnm, relevant_vessels, relevant_nodes, time_period_length)
        loading_times_df.to_excel(self.excel_writer, sheet_name=out_sheetname,
                                  startrow=1)  # startrow=1 because of skiprows in read

    def write_transport_costs_to_file(self, relevant_vessels: List[str]) -> None:
        transport_costs_df = self.get_vessel_transport_costs(relevant_vessels=relevant_vessels)
        transport_costs_df.to_excel(self.excel_writer, sheet_name="transport_cost", startrow=1)


if __name__ == '__main__':
    lnm = LocationNodeMapping(factory_locations=["2022", "482"],
                              order_locations=["12003","13055","14035","26435","39197","11736","37357","13518","21336","2213","29816","27315","30236","30757","36797","11338"])
    tdg = TestDataGenerator()
    tdg.write_test_instance_to_file(lnm=lnm,
                                    out_filepath="../data/testoutputfile.xlsx",
                                    relevant_vessels=["v_1", "v_2", "v_3"],
                                    time_period_length=2)
import math
from typing import List
import pandas as pd


def _hours_to_time_periods(hours: float, time_period_length: float) -> int:
    if hours % time_period_length < 0.25:  # TODO: Is this OK? Rounding down if less than 0.25 over time period threshold
        time = math.floor(hours / time_period_length)
    else:
        time = math.ceil(hours / time_period_length)
    return time


class TestDataGenerator:
    excel_writer: pd.ExcelWriter

    def __init__(self):
        self.vessels_df = pd.read_excel("../data/vessels.xlsx", sheet_name="vessels", index_col=[0])
        self.distances_df = pd.read_excel("../data/distance_matrix.xlsx", sheet_name="distances", index_col=[0])
        self.location_node_mapping = pd.read_excel("../data/location_node_mapping.xlsx", sheet_name="mapping",
                                                   index_col=[0])

    def write_test_instance_to_file(self, out_filepath: str,
                                    relevant_vessels: List[str], relevant_nodes: List[str], time_period_length: int):
        self.excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', options={'strings_to_formulas': False})

        # Write sheets to file
        self.write_transport_times_to_file(out_sheetname="transport_time",
                                           relevant_vessels=relevant_vessels, relevant_nodes=relevant_nodes,
                                           time_period_length=time_period_length)
        self.write_loading_times_to_file(out_sheetname="loading_unloading_time",
                                         relevant_vessels=relevant_vessels, relevant_nodes=relevant_nodes,
                                         time_period_length=time_period_length)

        self.excel_writer.save()
        self.excel_writer.close()

    def get_all_loading_times(self, relevant_vessels: List[str], relevant_nodes: List[str],
                              time_period_length: int) -> pd.DataFrame:
        loading_times_df = pd.DataFrame(relevant_nodes, columns=["node"]).set_index("node", drop=True)
        for v in relevant_vessels:
            d = {}
            for node in loading_times_df.index:
                try:
                    if node.startswith("f"):
                        hours = 100 / self.vessels_df.loc[v, "loading_rate"]  # TODO: Fix quantity read (in elif also)
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    else:
                        hours = 100 / self.vessels_df.loc[v, "unloading_rate"]
                        d[node] = _hours_to_time_periods(hours=hours, time_period_length=time_period_length)
                    loading_times_df[v] = loading_times_df.index.to_series().map(d)
                except KeyError:
                    print(f"Failed to find rate for vessel {v} and node {node}")
        return loading_times_df

    def get_vessel_transport_times(self, relevant_vessels: List[str], relevant_nodes: List[str],
                                   time_period_length: int) -> pd.DataFrame:
        vessels_df = self.vessels_df.loc[relevant_vessels]
        relevant_locations = list(int(self.location_node_mapping.loc[node]) for node in relevant_nodes)
        distances_df = self.distances_df.loc[relevant_locations, relevant_locations]

        idx = pd.MultiIndex.from_product([relevant_nodes, relevant_vessels], names=['node', 'vessel'])
        transport_times_df = pd.DataFrame(0, index=idx, columns=relevant_nodes)

        for v in relevant_vessels:
            speed = vessels_df.loc[v, 'knop_avg']
            for orig in relevant_nodes:
                for dest in relevant_nodes:
                    hours = int(
                        distances_df.loc[self.get_node_location_id(orig), self.get_node_location_id(dest)] * speed)
                    transport_times_df.loc[(orig, v), dest] = _hours_to_time_periods(hours=hours,
                                                                                     time_period_length=time_period_length)
        # TODO: Must implement reading of 3D table in read_problem_data
        return transport_times_df

    def get_vessel_name(self, vessel_id: str) -> str:
        return self.vessels_df.loc[vessel_id, "vessel_name"]

    def get_node_location_id(self, node_id: str) -> int:
        return int(self.location_node_mapping.loc[node_id])

    def write_transport_times_to_file(self, out_sheetname: str, relevant_vessels: List[str], relevant_nodes: List[str],
                                      time_period_length: int) -> None:
        transport_times_df = self.get_vessel_transport_times(relevant_vessels, relevant_nodes, time_period_length)
        transport_times_df.to_excel(self.excel_writer, sheet_name=out_sheetname,
                                    startrow=1)  # startrow=1 because of skiprows in read

    def write_loading_times_to_file(self, out_sheetname: str, relevant_vessels: List[str], relevant_nodes: List[str],
                                    time_period_length: int) -> None:
        loading_times_df = self.get_all_loading_times(relevant_vessels, relevant_nodes, time_period_length)
        loading_times_df.to_excel(self.excel_writer, sheet_name=out_sheetname,
                                  startrow=1)  # startrow=1 because of skiprows in read


if __name__ == '__main__':
    tdg = TestDataGenerator()
    tdg.write_test_instance_to_file(out_filepath="../data/testoutputfile.xlsx",
                                    relevant_vessels=["v_1", "v_2", "v_3", "v_4"],
                                    relevant_nodes=["f_1", "f_2", "o_1", "o_2"],
                                    time_period_length=2)

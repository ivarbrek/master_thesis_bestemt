import openpyxl
import pandas as pd
from typing import List, Dict, Tuple
import requests
from time import time
import math

# New version of this file in folder generate_data

def get_distances(temp_df: pd.DataFrame, ids: List[int], to_id: int) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    count = 0
    for from_id in ids:
        count += 1
        if count % 100 == 0:
            print(f"Completed {count} calculations for to_id {to_id}...")
        if from_id == to_id:
            d = 0
        else:
            try:
                d = temp_df.loc[to_id, from_id]
            except KeyError:
                url = f"https://routes.anteo.no/rpc/trip?from={from_id}&to={to_id}"
                response = requests.get(url)
                d = response.json()['properties']['cost']
        dist[from_id] = d
    return dist


def make_distance_matrix_sheet(locations: List[int], file_path: str, to_sheet: str):
    distances_df = pd.DataFrame(locations, columns=["from_id"]).set_index("from_id", drop=True)
    t0 = time()
    l = len(locations)
    count = 0
    for to_id in locations:
        d = get_distances(temp_df=distances_df, ids=locations, to_id=to_id)
        distances_df[to_id] = distances_df.index.to_series().map(d)
        count += 1
        print(f"{round(time() - t0, 1)}s - location with ID {to_id} complete - {round((count/l)*100, 2)}% complete")

    print(distances_df)

    excel_writer = pd.ExcelWriter(file_path, engine='openpyxl', options={'strings_to_formulas': False})
    distances_df.to_excel(excel_writer, sheet_name=to_sheet)
    excel_writer.save()
    excel_writer.close()


def _validate_transport_time_triangle_inequality(distance_matrix: pd.DataFrame) -> None:
    nodes = list(n for n in distance_matrix.index if (n != 11771 and n != 11772))
    for origin in nodes:
        for intermediate in nodes:
            for destination in nodes:
                if origin != intermediate and intermediate != destination:
                    try:
                        assert (math.floor(distance_matrix.loc[origin, destination]) <=
                                math.floor(distance_matrix.loc[origin, intermediate] +
                                distance_matrix.loc[intermediate, destination]))
                    except ValueError:
                        print(f"Value not found for origin {origin}, intermediate {intermediate} and destination {destination}")
                    except AssertionError:
                        print(f"{origin} -> {intermediate} -> {destination}: "
                              f"{distance_matrix.loc[origin, intermediate] + distance_matrix.loc[intermediate, destination]}")
                        print(f"{origin} -> {destination}: {distance_matrix.loc[origin, destination]}")


if __name__ == '__main__':
    pass
    # locations = list(pd.read_excel("../data/locations.xlsx", index_col=[0])["location id"])
    # print(f"Investigating {len(locations)} locations...")

    # Init
    # make_distance_matrix_sheet(locations=locations, file_path="../data/old/distance_matrix_OLD.xlsx", to_sheet="distances")

    # Triangle inequality check
    # distance_matrix = pd.read_excel("../data/distance_matrix.xlsx", sheet_name="distances", index_col=[0])
    # _validate_transport_time_triangle_inequality(distance_matrix=distance_matrix)
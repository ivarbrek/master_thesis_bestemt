import pandas as pd
from typing import List, Dict
import requests
from time import time


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


if __name__ == '__main__':
    locations = list(pd.read_excel("../data/locations.xlsx", index_col=[0])["location id"])
    print(f"Investigating {len(locations)} locations...")

    # Init
    make_distance_matrix_sheet(locations=locations, file_path="../data/old/distance_matrix_OLD.xlsx", to_sheet="distances")

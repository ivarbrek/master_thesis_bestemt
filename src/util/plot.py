import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
import plotly.graph_objects as go


def plot_alns_history(solution_costs: List[Tuple[int, int]], lined: bool = False, legend: str = "") -> None:
    x, y = zip(*solution_costs)
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    plt.scatter(x, y, s=7, alpha=0.4, c='black')
    if lined:
        plt.plot(x, y, label=legend)
    if legend != "":
        plt.legend()
    # plt.yscale('log')
    plt.show()


def plot_operator_weights(operator_scores: Dict[str, List[float]], x_values: List[int] = None) -> None:
    # plt.figure(figsize=(10, 7))  # (8, 6) is default
    legend = []

    for operator, scores in operator_scores.items():
        if x_values:
            plt.plot(x_values, scores)
        else:
            plt.plot(scores)
        legend.append(_get_operator_legend_name(operator))
    plt.legend(legend)
    plt.xlabel("Iteration")
    plt.show()


def _get_operator_legend_name(operator_name: str) -> str:
    mapping = {
        'true': 'insertion with noise',
        'false': 'insertion without noise',
        'r_greedy': 'greedy',
        'r_2regret': '2-regret',
        'r_3regret': '3-regret',
        'd_random': 'random',
        'd_worst': 'worst',
        'd_voyage_random': 'random voyage',
        'd_voyage_worst': 'worst voyage',
        'd_route_random': 'random route',
        'd_route_worst': 'worst route',
        'd_related_location_time': 'spatial temporal related',
        'd_related_location_precedence': 'spatial disease related',

    }
    return mapping[operator_name]


def plot_alns_history_with_production_feasibility(solution_costs: List[Tuple[int, int]],
                                                  production_feasibility: List[bool]) -> None:

    df = pd.DataFrame(dict(iter=[elem[0] for elem in solution_costs],
                           cost=[elem[1] for elem in solution_costs],
                           feasible=production_feasibility))

    fig, ax = plt.subplots()
    colors = {False: 'red', True: 'green'}

    ax.scatter(df['iter'], df['cost'], c=df['feasible'].apply(lambda x: colors[x]))
    plt.show()


def plot_locations(locations_ids: List[str], special_locations: List[Tuple[float, float]] = None, save_to: str=None):
    # Special locations:
    # 0482: (59.3337534309431, 5.30413145167106), 2022: (11.2786502472518,64.857954476573), 2015: (15.0646427525587,68.9141123038669)
    loc_data = pd.read_csv('../../data/locations.csv')
    loc_data.set_index("loknr", inplace=True)
    farm_size = 7
    factory_size = 15
    farm_color = '#0067b5' #skyblue'
    factory_color = 'black'
    factory_marker = 'square'
    farm_marker = 'circle'
    relevant_locations_and_coords = [(loc_id,
                                      loc_data.loc[int(loc_id), "breddegrader"],
                                      loc_data.loc[int(loc_id), "lengdegrader"], farm_color, farm_size, farm_marker)
                                     for loc_id in locations_ids if int(loc_id) in loc_data.index]
    relevant_locations_and_coords += [(000, coord[0], coord[1], factory_color, factory_size, factory_marker)
                                      for coord in special_locations]
    df = pd.DataFrame(relevant_locations_and_coords)
    df.columns = ['loc_id', 'lat', 'long', 'color', 'size', 'marker']
    # color = c if c else "LightSkyBlue"

    # with open("../../data/custom.geo.json", "r", encoding="utf-8") as f:
    #     geometry = geojson.load(f)
    # pprint(geometry)
    # trace1 = go.Choropleth(geojson=geometry,
    #                        locations=["Norway"],
    #                        z=[0],
    #                        text=['Norway-text']
    #                        )

    trace2 = go.Scattergeo(
            lon=df['long'],
            lat=df['lat'],
            text=df['loc_id'],
            mode='markers',
            marker=dict(
                color=df['color'],
                size=df['size'],
                symbol=df['marker'],
                line=dict(color='black', width=0),
                opacity=1

            )
        )
    fig = go.Figure([trace2])

    # fig.update_layout(
    #     title='Locations',
    #     geo_scope='europe',
    # )
    fig.update_geos(
        fitbounds="locations",
        resolution=50,
        # visible=False,
        showframe=False,
        projection={"type": "mercator"},
    )
    if save_to:  # Save figure
        fig.write_html(save_to)
    fig.show()


def plot_clustered_locations(locations_ids_list: List[List[str]],
                             special_locations_list: List[List[Tuple[float, float]]] = None, save_to: str=None):
    loc_data = pd.read_csv('../../data/locations.csv')
    loc_data.set_index("loknr", inplace=True)
    farm_size = 9
    factory_size = 15
    factory_color = 'black'
    farm_colors = ['#0067b5', '#e67512', '#006700', '#bb00bb']
    factory_marker = 'square'
    farm_marker = 'circle'
    traces = []
    for locations_ids, special_locations, farm_color in zip(locations_ids_list, special_locations_list, farm_colors):
        relevant_locations_and_coords = [(loc_id,
                                          loc_data.loc[int(loc_id), "breddegrader"],
                                          loc_data.loc[int(loc_id), "lengdegrader"], farm_color, farm_size, farm_marker)
                                         for loc_id in locations_ids if int(loc_id) in loc_data.index]
        relevant_locations_and_coords += [(000, coord[0], coord[1], factory_color, factory_size, factory_marker)
                                          for coord in special_locations]
        df = pd.DataFrame(relevant_locations_and_coords)
        df.columns = ['loc_id', 'lat', 'long', 'color', 'size', 'marker']
        # color = c if c else "LightSkyBlue"

        trace = go.Scattergeo(
                lon=df['long'],
                lat=df['lat'],
                text=df['loc_id'],
                mode='markers',
                marker=dict(
                    color=df['color'],
                    size=df['size'],
                    symbol=df['marker'],
                    line=dict(color='black', width=0),
                    opacity=1

                )
            )
        traces.append(trace)

    fig = go.Figure(traces)

    # fig.update_layout(
    #     title='Locations',
    #     geo_scope='europe',
    # )
    fig.update_geos(
        fitbounds="locations",
        resolution=50,
        # visible=False,
        showframe=False,
        projection={"type": "mercator"},
    )
    if save_to:  # Save figure
        fig.write_html(save_to)
    fig.show()

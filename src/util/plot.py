import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd

def plot_alns_history(solution_costs: List[Tuple[int, int]]) -> None:
    x, y = zip(*solution_costs)
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    plt.scatter(x, y, s=7, alpha=0.4, c='black')
    # plt.yscale('log')
    plt.show()


def plot_operator_weights(operator_scores: Dict[str, List[float]]):
    plt.figure(figsize=(10, 7))  # (8, 6) is default
    legend = []
    for operator, scores in operator_scores.items():
        plt.plot(scores)
        legend.append(operator)
    plt.legend(legend)
    plt.show()


def plot_alns_history_with_production_feasibility(solution_costs: List[Tuple[int, int]],
                                                  production_feasibility: List[bool]) -> None:

    df = pd.DataFrame(dict(iter=[elem[0] for elem in solution_costs],
                           cost=[elem[1] for elem in solution_costs],
                           feasible=production_feasibility))

    fig, ax = plt.subplots()
    colors = {False: 'red', True: 'green'}

    ax.scatter(df['iter'], df['cost'], c=df['feasible'].apply(lambda x: colors[x]))
    plt.show()


def plot_locations(locations_ids: List[str]):
    loc_data = pd.read_csv('../../data/locations.csv')
    loc_data.set_index("loknr", inplace=True)
    relevant_locations_and_coords = [(loc_id,
                                      loc_data.loc[int(loc_id), "breddegrader"],
                                      loc_data.loc[int(loc_id), "lengdegrader"])
                                     for loc_id in locations_ids if int(loc_id) in loc_data.index]
    df = pd.DataFrame(relevant_locations_and_coords)
    df.columns = ['loc_id', 'lat', 'long']
    # color = c if c else "LightSkyBlue"
    fig = go.Figure(data=go.Scattergeo(
        lon=df['long'],
        lat=df['lat'],
        text=df['loc_id'],
        mode='markers',
        marker=dict(
            # color=color,
            size=12,
            line=dict(color='MediumPurple', width=1)
        )
    ))

    fig.update_layout(
        title='Locations',
        geo_scope='europe',
    )
    fig.update_geos(resolution=50)
    # if save_to:  # Save figure
    #     fig.write_html(save_to)
    fig.show()

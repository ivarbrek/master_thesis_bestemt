from src.models.basic_model import BasicModel
from src.read_problem_data import ProblemData


def model_create(problem_data: ProblemData, extensions=False):
    return BasicModel(nodes=problem_data.get_nodes(),
                      factory_nodes=problem_data.get_factory_nodes(),
                      order_nodes=problem_data.get_order_nodes(),
                      orders_for_zones=problem_data.get_zone_orders_dict(),
                      nodes_for_vessels=problem_data.get_nodes_for_vessels_dict(),
                      arcs_for_vessels=problem_data.get_arcs_for_vessel_dict(),
                      products=problem_data.get_products(),
                      vessels=problem_data.get_vessels(),
                      time_periods=problem_data.get_time_periods(),
                      time_periods_for_vessels=problem_data.get_time_periods_for_vessels_dict(),
                      vessel_initial_locations=problem_data.get_vessel_first_location(),
                      time_windows_for_orders=problem_data.get_time_windows_for_orders_dict(),
                      max_tw_violation=problem_data.get_max_time_window_violation(),
                      tw_violation_unit_cost=problem_data.get_tw_violation_cost(),
                      min_wait_if_sick=problem_data.get_min_wait_if_sick(),
                      vessel_ton_capacities=problem_data.get_vessel_ton_capacities_dict(),
                      vessel_nprod_capacities=problem_data.get_vessel_nprod_capacities_dict(),
                      vessel_initial_loads=problem_data.get_vessel_initial_loads_dict(),
                      factory_inventory_capacities=problem_data.get_inventory_capacities_dict(),
                      factory_initial_inventories=problem_data.get_initial_inventories_dict(),
                      inventory_unit_costs=problem_data.get_inventory_unit_costs_dict(),
                      transport_unit_costs=problem_data.get_transport_costs_dict(),
                      transport_times=problem_data.get_transport_times_dict(),
                      loading_unloading_times=problem_data.get_loading_unloading_times_dict(),
                      demands=problem_data.get_demands_dict(),
                      production_stops=problem_data.get_production_stops_dict(),
                      production_start_costs=problem_data.get_production_start_costs_dict(),
                      production_min_capacities=problem_data.get_production_min_capacities_dict(),
                      production_max_capacities=problem_data.get_production_max_capacities_dict(),
                      production_lines=problem_data.get_production_lines(),
                      production_lines_for_factories=problem_data.get_production_lines_for_factories_list(),
                      production_line_min_times=problem_data.get_production_line_min_times_dict(),
                      product_groups=problem_data.get_product_groups_dict(),
                      factory_max_vessels_destination=problem_data.get_factory_max_vessels_destination_dict(),
                      factory_max_vessels_loading=problem_data.get_factory_max_vessels_loading_dict(),
                      inventory_targets=problem_data.get_inventory_targets(),
                      inventory_unit_rewards=problem_data.get_inventory_unit_rewards_dict(),
                      external_delivery_penalty=problem_data.get_key_value("external_delivery_penalty"),
                      extended_model=extensions
                      )


# problem_data = ProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
# problem_data = ProblemData('../../data/input_data/small_testcase.xlsx')
problem_data = ProblemData('../../data/input_data/medium_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/large_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/larger_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/larger_testcase_4vessels.xlsx')

extensions = False
problem_data.soft_tw = extensions
model = model_create(problem_data, extensions)
model.solve(time_limit=30)
model.print_result()

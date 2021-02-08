from src.models.basic_model import BasicModel
from src.read_problem_data import ProblemData


def model_create():
    return BasicModel(nodes=problem_data.get_nodes(),
                      factory_nodes=problem_data.get_factory_nodes(),
                      order_nodes=problem_data.get_order_nodes(),
                      nodes_for_vessels=problem_data.get_nodes_for_vessels_dict(),
                      products=problem_data.get_products(),
                      vessels=problem_data.get_vessels(),
                      time_periods=problem_data.get_time_periods(),
                      time_periods_for_vessels=problem_data.get_time_periods_for_vessels_dict(),
                      vessel_initial_locations=problem_data.get_vessel_first_location(),
                      time_windows_for_orders=problem_data.get_time_windows_for_orders_dict(),
                      vessel_ton_capacities=problem_data.get_vessel_ton_capacities_dict(),
                      vessel_nprod_capacities=problem_data.get_vessel_nprod_capacities_dict(),
                      vessel_initial_loads=problem_data.get_vessel_initial_loads_dict(),
                      factory_inventory_capacities=problem_data.get_inventory_capacities_dict(),
                      factory_initial_inventories=problem_data.get_initial_inventories_dict(),
                      inventory_unit_costs=problem_data.get_inventory_unit_costs_dict(),
                      transport_unit_costs=problem_data.get_transport_costs_dict(),
                      transport_times=problem_data.get_transport_times_dict(),
                      unloading_times=problem_data.get_unloading_times_dict(),
                      loading_times=problem_data.get_loading_times_dict(),
                      demands=problem_data.get_demands_dict(),
                      production_unit_costs=problem_data.get_production_unit_costs_dict(),
                      production_min_capacities=problem_data.get_production_min_capacities_dict(),
                      production_max_capacities=problem_data.get_production_max_capacities_dict(),
                      production_lines=problem_data.get_production_lines(),
                      production_lines_for_factories=problem_data.get_production_lines_for_factories_list(),
                      production_line_min_times=problem_data.get_production_line_min_times_dict()
                      )


problem_data = ProblemData('../../data/input_data/small_testcase.xlsx')

model = model_create()
model.solve()
model.print_result()

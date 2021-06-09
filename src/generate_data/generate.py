from src.generate_data.test_data_generator import TestDataGenerator


def generate_parameter_tuning_instances():
    # Input parameters and info that varies for each instance:
    base_settings = ['3-1', '5-2']
    all_orders = [[20, 40, 60], [20, 40, 60]]
    all_planning_horizon_days = [[6, 9, 14], [5, 8, 12]]
    all_vessels = [["Borgenfjord", "Nyksund", "Høydal"], ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]]
    all_factories = [["2016"], ["2022", "482"]]
    companies = ["Mowi Feed AS", "BioMar AS"]
    inventory_levels = [0.2, 0.5]
    inventory_level_encoding = {0.2: "l", 0.5: "h"}

    # Input parameters kept constant:
    tw_length_hours = 4 * 24
    time_period_length = 2
    no_products = 10
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    for no_orders_group, planning_horizon_days_group, vessels, factories, company, setting_name in zip(
            all_orders, all_planning_horizon_days, all_vessels, all_factories, companies, base_settings):
        generator = TestDataGenerator()  # need re-instantiated generator per set of factories
        for no_orders, planning_horizon_days in zip(no_orders_group, planning_horizon_days_group):
            for inventory_level in inventory_levels:
                instance_name = f"tuning-{setting_name}-{no_orders}-{inventory_level_encoding[inventory_level]}"
                time_periods = int(planning_horizon_days * 24 / time_period_length)
                print(instance_name)

                generator.write_test_instance_to_file(
                    # Input parameters varying:
                    out_filepath=f"../../data/input_data/parameter_tuning/{instance_name}.xlsx",
                    vessel_names=vessels,
                    factory_locations=factories,
                    orders_from_company=company,
                    no_orders=no_orders,
                    factory_level=inventory_level,
                    ext_depot_level=0.1,
                    time_periods=time_periods,
                    tw_length_hours=tw_length_hours,
                    # Input parameters kept constant:
                    time_period_length=time_period_length,
                    no_products=no_products,
                    no_product_groups=no_product_groups,
                    quay_activity_level=quay_activity_level,
                    hours_production_stop=hours_production_stop,
                    share_red_nodes=share_red_nodes,
                    radius_red=radius_red,
                    radius_yellow=radius_yellow,
                    share_bag_locations=share_bag_locations,
                    share_small_fjord_locations=share_small_fjord_locations,
                    share_time_periods_vessel_availability=share_time_periods_vessel_availability,
                    small_fjord_radius=small_fjord_radius,
                    min_wait_if_sick_hours=min_wait_if_sick_hours,
                    delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                    earliest_tw_start=earliest_tw_start
                )


def generate_performance_testing_instances():
    # Input parameters and info that varies for each instance:
    instances_per_modification = 3
    base_settings = ['3-1', '5-2']
    all_orders = [[20, 40, 60], [20, 40, 60]]
    all_planning_horizon_days = [[6, 9, 14], [5, 8, 12]]
    all_vessels = [["Borgenfjord", "Nyksund", "Høydal"], ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]]
    all_factories = [["2016"], ["2022", "482"]]
    companies = ["Mowi Feed AS", "BioMar AS"]
    inventory_levels = [0.2, 0.5]
    inventory_level_encoding = {0.2: "l", 0.5: "h"}
    no_products = [10]

    # Input parameters kept constant:
    tw_length_hours = 4 * 24
    time_period_length = 1
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    for no_orders_group, planning_horizon_days_group, vessels, factories, company, setting_name in zip(
            all_orders, all_planning_horizon_days, all_vessels, all_factories, companies, base_settings):
        generator = TestDataGenerator()  # need re-instantiated generator per set of factories
        for no_orders, planning_horizon_days in zip(no_orders_group, planning_horizon_days_group):
            for inventory_level in inventory_levels:
                for no_product in no_products:
                    for i in range(instances_per_modification):
                        instance_name = f"performance-{setting_name}-{no_orders}-" \
                                        f"{inventory_level_encoding[inventory_level]}-{i}"
                        time_periods = int(planning_horizon_days * 24 / time_period_length)
                        print(instance_name)

                        generator.write_test_instance_to_file(
                            # Input parameters varying:
                            out_filepath=f"../../data/input_data/performance_testing/{instance_name}.xlsx",
                            vessel_names=vessels,
                            factory_locations=factories,
                            orders_from_company=company,
                            no_orders=no_orders,
                            factory_level=inventory_level,
                            ext_depot_level=0.1,
                            time_periods=time_periods,
                            tw_length_hours=tw_length_hours,
                            # Input parameters kept constant:
                            time_period_length=time_period_length,
                            no_products=no_product,
                            no_product_groups=no_product_groups,
                            quay_activity_level=quay_activity_level,
                            hours_production_stop=hours_production_stop,
                            share_red_nodes=share_red_nodes,
                            radius_red=radius_red,
                            radius_yellow=radius_yellow,
                            share_bag_locations=share_bag_locations,
                            share_small_fjord_locations=share_small_fjord_locations,
                            share_time_periods_vessel_availability=share_time_periods_vessel_availability,
                            small_fjord_radius=small_fjord_radius,
                            min_wait_if_sick_hours=min_wait_if_sick_hours,
                            delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                            earliest_tw_start=earliest_tw_start
                        )


def generate_time_period_duplicate_instances():
    # Input parameters and info that varies for each instance:
    instances_per_modification = 3
    base_settings = ['3-1', '5-2']
    all_orders = [[30, 60], [30, 60]]
    all_planning_horizon_days = [[8, 14], [5, 12]]
    all_vessels = [["Borgenfjord", "Nyksund", "Høydal"], ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]]
    all_factories = [["2016"], ["2022", "482"]]
    companies = ["Mowi Feed AS", "BioMar AS"]
    time_period_lengths = [1, 2, 3]

    # Input parameters kept constant:
    tw_length_hours = 4 * 24
    inventory_level = 0.2
    no_products = 10
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    for no_orders_group, planning_horizon_days_group, vessels, factories, company, setting_name in zip(
            all_orders, all_planning_horizon_days, all_vessels, all_factories, companies, base_settings):
        generator = TestDataGenerator()  # need re-instantiated generator per set of factories
        for no_orders, planning_horizon_days in zip(no_orders_group, planning_horizon_days_group):
            for i in range(instances_per_modification):
                instance_names = [f"tp_testing-{setting_name}-{no_orders}-tp_length={time_period_length}-{i}"
                                  for time_period_length in time_period_lengths]
                time_periods_list = [int(planning_horizon_days * 24 / time_period_length)
                                     for time_period_length in time_period_lengths]
                print(instance_names)

                generator.write_duplicate_test_instances_to_file_time_periods(
                    # Input parameters varying:
                    out_filepaths=[f"../../data/input_data/time_periods_testing/{instance_name}.xlsx"
                                   for instance_name in instance_names],
                    vessel_names=vessels,
                    factory_locations=factories,
                    orders_from_company=company,
                    no_orders=no_orders,
                    time_periods_list=time_periods_list,
                    time_period_lengths=time_period_lengths,
                    tw_length_hours=tw_length_hours,
                    # Input parameters kept constant:
                    factory_level=inventory_level,
                    ext_depot_level=0.1,
                    no_products=no_products,
                    no_product_groups=no_product_groups,
                    quay_activity_level=quay_activity_level,
                    hours_production_stop=hours_production_stop,
                    share_red_nodes=share_red_nodes,
                    radius_red=radius_red,
                    radius_yellow=radius_yellow,
                    share_bag_locations=share_bag_locations,
                    share_small_fjord_locations=share_small_fjord_locations,
                    share_time_periods_vessel_availability=share_time_periods_vessel_availability,
                    small_fjord_radius=small_fjord_radius,
                    min_wait_if_sick_hours=min_wait_if_sick_hours,
                    delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                    earliest_tw_start_hour=earliest_tw_start
                )


def generate_time_window_duplicate_instances():
    # Input parameters and info that varies for each instance:
    base_settings = ['3-1', '5-2']
    all_orders = [[30, 60], [30, 60]]
    all_planning_horizon_days = [[8, 14], [5, 12]]
    all_vessels = [["Borgenfjord", "Nyksund", "Høydal"], ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]]
    all_factories = [["2016"], ["2022", "482"]]
    companies = ["Mowi Feed AS", "BioMar AS"]
    tw_length_hours_list = [days * 24 for days in [3, 4, 5]]

    # Input parameters kept constant:
    time_period_length = 2
    inventory_level = 0.4
    no_products = 10
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    for no_orders_group, planning_horizon_days_group, vessels, factories, company, setting_name in zip(
            all_orders, all_planning_horizon_days, all_vessels, all_factories, companies, base_settings):
        generator = TestDataGenerator()  # need re-instantiated generator per set of factories
        for no_orders, planning_horizon_days in zip(no_orders_group, planning_horizon_days_group):
            instance_names = [f"tw_testing-{setting_name}-{no_orders}-tw_length={tw_length//24}d"
                              for tw_length in tw_length_hours_list]
            print(instance_names)
            time_periods = planning_horizon_days * 24

            generator.write_duplicate_test_instances_to_file_time_windows(
                # Input parameters varying:
                out_filepaths=[f"../../data/input_data/time_windows_testing/{instance_name}.xlsx"
                               for instance_name in instance_names],
                vessel_names=vessels,
                factory_locations=factories,
                orders_from_company=company,
                no_orders=no_orders,
                tw_length_hours_list=tw_length_hours_list,
                # Input parameters kept constant:
                time_periods=time_periods,
                time_period_length=time_period_length,
                factory_level=inventory_level,
                ext_depot_level=0.1,
                no_products=no_products,
                no_product_groups=no_product_groups,
                quay_activity_level=quay_activity_level,
                hours_production_stop=hours_production_stop,
                share_red_nodes=share_red_nodes,
                radius_red=radius_red,
                radius_yellow=radius_yellow,
                share_bag_locations=share_bag_locations,
                share_small_fjord_locations=share_small_fjord_locations,
                share_time_periods_vessel_availability=share_time_periods_vessel_availability,
                small_fjord_radius=small_fjord_radius,
                min_wait_if_sick_hours=min_wait_if_sick_hours,
                delivery_delay_unit_penalty=delivery_delay_unit_penalty,
                earliest_tw_start_hour=earliest_tw_start
            )

def generate_factory_decompose_instances():
    generator = TestDataGenerator()

    no_instances = 5
    vessels = ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]
    company = "BioMar AS"
    factories = ["2022", "482"]
    no_orders = 80
    inventory_level = 0.4
    time_periods = 24 * 14
    no_product = 10
    tw_length_hours = 4 * 24
    time_period_length = 1
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    for i in range(10, 10 + no_instances):
        instance_name = f'factory_decompose-5-2-{no_orders}-{i}'
        print(instance_name)
        generator.write_test_instance_to_file(
            # Input parameters varying:
            out_filepath=f"../../data/input_data/factory_decompose/{instance_name}.xlsx",
            vessel_names=vessels,
            factory_locations=factories,
            orders_from_company=company,
            no_orders=no_orders,
            factory_level=inventory_level,
            ext_depot_level=0.1,
            time_periods=time_periods,
            tw_length_hours=tw_length_hours,
            # Input parameters kept constant:
            time_period_length=time_period_length,
            no_products=no_product,
            no_product_groups=no_product_groups,
            quay_activity_level=quay_activity_level,
            hours_production_stop=hours_production_stop,
            share_red_nodes=share_red_nodes,
            radius_red=radius_red,
            radius_yellow=radius_yellow,
            share_bag_locations=share_bag_locations,
            share_small_fjord_locations=share_small_fjord_locations,
            share_time_periods_vessel_availability=share_time_periods_vessel_availability,
            small_fjord_radius=small_fjord_radius,
            min_wait_if_sick_hours=min_wait_if_sick_hours,
            delivery_delay_unit_penalty=delivery_delay_unit_penalty,
            earliest_tw_start=earliest_tw_start,
            ensure_vessel_positions=True,
            assign_to_closest_factory=True,
            plot_locations="factory_decompose"

        )


if __name__ == '__main__':
    # generate_parameter_tuning_instances()
    # generate_time_period_duplicate_instances()
    # generate_performance_testing_instances()
    # generate_time_window_duplicate_instances()
    # generate_factory_decompose_instances()


    generator = TestDataGenerator()
    vessels = ["Ripnes", "Vågsund", "Nyksund", "Borgenfjord", "Høydal"]
    company = "BioMar AS"
    factories = ["2022", "482", "2015"]
    inventory_level = 0.2
    time_periods = 24 * 5
    no_product = 3
    instance_name = 'test.xlsx'
    tw_length_hours = 4 * 24
    time_period_length = 1
    no_product_groups = 4
    quay_activity_level = 0.1
    hours_production_stop = 12
    share_red_nodes = 0.1
    radius_red = 10000
    radius_yellow = 30000
    share_bag_locations = 0.2
    share_small_fjord_locations = 0.05
    share_time_periods_vessel_availability = 0.5
    small_fjord_radius = 50000
    min_wait_if_sick_hours = 12
    delivery_delay_unit_penalty = 10000
    earliest_tw_start = 5

    generator.write_test_instance_to_file(
        # Input parameters varying:
        out_filepath=f"../../data/input_data/performance_testing/{instance_name}.xlsx",
        vessel_names=vessels,
        factory_locations=factories,
        orders_from_company=company,
        no_orders=90,
        factory_level=inventory_level,
        ext_depot_level=0.1,
        time_periods=time_periods,
        tw_length_hours=tw_length_hours,
        # Input parameters kept constant:
        time_period_length=time_period_length,
        no_products=no_product,
        no_product_groups=no_product_groups,
        quay_activity_level=quay_activity_level,
        hours_production_stop=hours_production_stop,
        share_red_nodes=share_red_nodes,
        radius_red=radius_red,
        radius_yellow=radius_yellow,
        share_bag_locations=share_bag_locations,
        share_small_fjord_locations=share_small_fjord_locations,
        share_time_periods_vessel_availability=share_time_periods_vessel_availability,
        small_fjord_radius=small_fjord_radius,
        min_wait_if_sick_hours=min_wait_if_sick_hours,
        delivery_delay_unit_penalty=delivery_delay_unit_penalty,
        earliest_tw_start=earliest_tw_start,
        order_size_factor=2,
        plot_locations="basic"
    )

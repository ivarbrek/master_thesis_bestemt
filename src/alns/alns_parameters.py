# CONSTANT PARAMS
weight_min_threshold = 0.2
inventory_reward = False
reinsert_with_ppfc = False
max_iter_same_solution = 50
remove_num_percentage_adjust = 0.05
max_iter_seg = 40
production_infeasibility_strike_max = 0
ppfc_slack_increment = 0.00
relatedness_precedence = {('green', 'yellow'): 6, ('green', 'red'): 10, ('yellow', 'red'): 4}
start_temperature_controlparam = 0.1
percentage_best_solution_production_solved = 0.05  # if 0, then only new best routing solutions are evaluated

# TUNED PARAMS
remove_percentage_interval = (0.1, 0.3)
related_removal_weight_param = {'relatedness_location_time': [0.005, 1], 'relatedness_location_precedence': [0.005, 0.2]}
score_params = [33, 9, 1]
reaction_param = 0.2
noise_param = 0.1
determinism_param = 5
noise_destination_param = 1
permute_chance = 0.05

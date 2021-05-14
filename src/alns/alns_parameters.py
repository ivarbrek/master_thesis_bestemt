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
start_temperature_controlparam = 0.1  # solution 10% worse than best solution is accepted with 50% prob.
# cooling_rate = 0.998

# TUNED PARAMS
remove_percentage_interval = (0.05, 0.2)
related_removal_weight_param = {'relatedness_location_time': [0.01, 0.5], 'relatedness_location_precedence': [0.01, 0.1]}
score_params = [33, 9, 13]  # corresponding to sigma_1, sigma_2, sigma_3 in R&P and L&N
reaction_param = 0.1
noise_param = 0.1
determinism_param = 5
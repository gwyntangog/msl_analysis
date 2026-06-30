from calc_funcs import tau_callibrate

interconnect_copper_kg = 58 # kg per vehicle (total copper content)
interconnect_total_mass_kg = 40 # kg per vehicle (net assembly mass)

interconnect_system_cost_usd = 500 # $ per vehicle (OEM supply price)

material_cost_share = 65 # %
non_material_cost_share = 35 # %

copper_share = 70 # % of conductor material
aluminum_share = 30 # %

copper_conductivity_iacs = 100 # %
aluminum_conductivity_iacs = 61 # %

design_life_years = 12 # years
failure_rate_ppm = 10 # ppm

# attribute weights (OEM decision model)
weight_cost = 40
weight_conductivity = 25
weight_weight = 20
weight_reliability = 15

#######COPPPERRRRRRRRRRRRR

weights_normalized = [weight_cost/100, weight_conductivity/100, weight_weight/100, weight_reliability/100]
features = [interconnect_system_cost_usd, copper_conductivity_iacs, interconnect_total_mass_kg *
copper_share/100, 1- (10/(10**6)) ]

print(weights_normalized)
print(features)

def generalized_utility_func(features, weights):
    """
    Calculates the utility of copper based on product cost, weight, efficiency, and state using
    one constant for each of those attributes, in order.
    """
    assert(len(features)== len(weights))
    utility = 0
    for i in range(len(features)):
        utility += features[i] * weights[i]
    return utility

copper_utility = generalized_utility_func(features, weights_normalized)
print(generalized_utility_func(features, weights_normalized))

########## ALUMINUM

interconnect_aluminum_kg = 45 # kg per vehicle (primary aluminum conductor content)
interconnect_copper_kg = 13  # kg per vehicle (residual copper in joints/connectors)
interconnect_total_mass_kg = 55 # kg per vehicle (net assembly mass)

interconnect_system_cost_usd = 420 # $ per vehicle (OEM supply price)

material_cost_share = 55 # %
non_material_cost_share = 45 # %

copper_share = 23 # % of conductor material
aluminum_share = 77 # %

copper_conductivity_iacs = 100 # %
aluminum_conductivity_iacs = 61 # %

design_life_years = 12 # years
failure_rate_ppm = 12 # ppm

# attribute weights (OEM decision model)
weight_cost = 40
weight_conductivity = 25
weight_weight = 20
weight_reliability = 15

weights_normalized = [weight_cost/100, weight_conductivity/100, weight_weight/100, weight_reliability/100]
features = [interconnect_system_cost_usd, aluminum_conductivity_iacs, interconnect_total_mass_kg *
aluminum_share/100, 1- (failure_rate_ppm/(10**6)) ]

print(weights_normalized)
print(features)

aluminum_utility = generalized_utility_func(features, weights_normalized)
print(generalized_utility_func(features, weights_normalized))

# LOGIT
market_share_cu = 0.7
tau_value = tau_callibrate(market_share_cu,copper_utility, aluminum_utility)
print(tau_value)

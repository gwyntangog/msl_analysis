# Functions for calculating values
import math

# Need to callibrate such that returned market share = today's market share
# Questions: what are A and beta, and what is the cost function
# GLOBAL VARIABLES

MARKET_SHARE_CU = 1
CANONICAL_BETA = 1

# Functions start here

def ms_price_ratio(A, P_cu, P_al, beta):
    """
    Calculates the market share of copper using ratio of price of copper and aluminum.
    P_cu = price of copper
    P_al = price of aluminum
    """
    # and σ is the elasticity of substitution estimated from historical behavior or derived from a calibrated CES model.
    market_share = A*((P_cu/P_al)**beta)
    return market_share

def callibrate_A_val(P_cu, P_al):
    """
    Callibrates the value of A based on current market share and beta value.
    """
    A_val = MARKET_SHARE_CU/((P_cu/P_al)**CANONICAL_BETA)
    return A_val

def ms_logit(utility_cu, utility_al, tau=1):
    """
    Calculates the market share of copper using the logit functions.
    utility_cu = utility of copper
    utility_al = utility of aluminum
    tau = scaling coefficient
    """
    scaled_cu = utility_cu/tau
    scaled_al = utility_al/tau
    cu_value = e**scaled_cu
    al_value = e**scaled_al
    total = cu_value + al_value
    e = math.e
    market_share_cu = cu_value/total
    return market_share_cu

def copper_utility_v1(product_cost, weight, efficiency, constants = [1,1,1,1], state = 0):
    """
    Calculates the utility of copper based on product cost, weight, efficiency, and state using
    one constant for each of those attributes, in order.
    """
    k_1, k_2, k_3, k_4 = constants
    utility = k_1*product_cost + k_2*weight + k_3*efficiency+k_4*state
    return utility

def copper_utility_v2(k, cost, other):
    """
    Calculates the utility of copper based on weight of cost and "other".
    """
    utility = k*cost + (1-k)*other
    return utility

def product_cost(non_material_cost, quantity, price, num_product):
    """
    Calculates the cost per product.
    """
    cost = non_material_cost + quantity*price/num_product
    return cost

def tau_callibrate(market_share_cu, utility_cu, utility_al):
    """
    Callibrate the value of tau such that the logit function returns the same value
    as the current MARKET_SHARE_CU. See derivation in Overleaf file.
    """
    tau = (utility_cu-utility_al)/(math.log(market_share_cu)-math.log(1-market_share_cu))
    return tau

# Make a variation for product cost based on material price
# Do we want a graph --> make a graph
# Derive A and beta from MS
# Test tomorrow

# PIPELINE
# Also automate sending queries etc.

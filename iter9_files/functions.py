from pathlib import Path
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import os

# PHASE 1: Normal finding tau
# PHASE 2: Tau callibrate step

def unzero(n):
    """
    Compresses the number to be in the range from "almost 0" to "almost 1".
    """
    return np.clip(n, 10**(-9), 1-10**(-9))

def tau_callibrate(utility_cu, utility_al, market_share_cu):
    """
    Callibrate the value of tau such that the logit function returns the same value
    as the current MARKET_SHARE_CU. See derivation in Overleaf file.
    Calculates tau based on copper utility, al utility, and market share.
    """
    mc = np.clip(market_share_cu, 10**(-9), 1-10**(-9))
    if mc == 0.5:
        return np.inf
    else:
        tau = (utility_cu-utility_al)/(np.log(mc)-np.log(1-mc))
        return tau

# TODO: Tau callibrate steo=p

def ms_logit(utility_cu, utility_al, tau):
    """
    Calculate the market share via the logit function.
    Note: this is raw, not softmax. see iter8 for softmax code.
    """
    exp_cu = np.exp(utility_cu/tau)
    exp_al = np.exp(utility_al/tau)
    total = exp_cu + exp_al
    result = exp_cu/total
    return result


def make_data_dict(global_df, regional_df, product_df, region, product):
    """
    Assume that you are given global, regional, and product dataframes.
    This function creates a dictionary with data for a specific product in a specific region.
    """
    result = {}
    result["region"] = region
    result["product"] = product
    global_df
    ###### GET MIN AND MAX PER ATTRIBUTE
    for m in ["cu","al"]:
        result[f"{m}_material_cost"]
        result[f"{m}_nonmaterial"]
        result[f"{m}_copper_kg"]
        result[f"{m}_aluminum_kg"]
    for i in range(1,6):
        result[f"a{i}_val"]
        result[f"a{i}_weight"]
        result[f"a{i}_min"]
        result[f"a{i}_max"]


def calc_product_cost(df, cu_material_cost =None, al_material_cost = None, current_product = "cu"):
    """
    Expects df to have these columns
        cu/al_nonmaterial
        cu/al_copper_kg
        cu/al_aluminum_kg
        cu/al_material_cost
    Assumes that there is only one row... hm... maybe make a dictionary instead...
    """
    nonmaterial = df[f"{current_product}_nonmaterial"].iloc[0]
    copper_kg = df[f"{current_product}_copper_kg"].iloc[0]
    aluminum_kg = df[f"{current_product}_aluminum_kg"].iloc[0]
    if cu_material_cost == None:
        cu_material_cost = df["cu_material_cost"].iloc[0]
    if al_material_cost == None:
        al_material_cost = df["al_material_cost"].iloc[0]
    product_cost = nonmaterial + copper_kg * cu_material_cost + al_material_cost * aluminum_kg
    return product_cost[0]

test_df = pd.DataFrame([{"cu_nonmaterial": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
             "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3}])

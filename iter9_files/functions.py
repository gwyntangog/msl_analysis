from pathlib import Path
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import os
import re
import ast

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

def calc_product_cost(input_df, cu_material_cost =None, al_material_cost = None, products = ["cu","al"]):
    """
    Expects df to have these columns
        cu/al_nonmaterial
        cu/al_copper_kg
        cu/al_aluminum_kg
        cu/al_material_cost
    Assumes that there is only one row... hm... maybe make a dictionary instead...
    """
    df = input_df.copy()
    for m in products:
        nonmaterial = df[f"{m}_non_material_cost_per_unit"]
        copper_kg = df[f"{m}_copper_kg"]
        aluminum_kg = df[f"{m}_aluminum_kg"]
        if cu_material_cost is None:
            cu_material_cost = df["copper_price_per_kg"]
        if al_material_cost is None:
            al_material_cost = df["aluminum_price_per_kg"]
        df[f"{m}_attribute_1_value"] = nonmaterial + copper_kg * cu_material_cost + al_material_cost * aluminum_kg
    return df

test_df = pd.DataFrame([{"cu_nonmaterial": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
             "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3}])




def parse_pdf(pdf_path):

    """
    Returns a dataframe that combines global df, regional df, product df
    into one new dataframe.
    """
    variables = {}

    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    for name in ["global_data", "regional_data", "product_data"]:
        match = re.search(
            rf'{name}\s*=\s*(\{{.*?\}}|\[.*?\])',
            text,
            flags=re.DOTALL,
        )
        if match is None:
            match = re.search(
            rf'"{name}"\s*:\s*(\{{.*?\}}|\[.*?\])',
            text,
            flags=re.DOTALL,
        )
        if match:
            variables[name] = ast.literal_eval(match.group(1))


    global_data = variables.get("global_data")
    regional_data = variables.get("regional_data")
    product_data = variables.get("product_data")


    global_df = pd.DataFrame([global_data])
    regional_df = pd.DataFrame(regional_data)
    product_df = pd.DataFrame(product_data)

    copper_df = product_df[product_df["dominant material"] == "cu"]
    copper_df = copper_df.add_prefix('cu_')
    copper_df.rename(columns={'cu_region': 'region'}, inplace=True)

    aluminum_df = product_df[product_df["dominant material"] == "al"]
    aluminum_df = aluminum_df.add_prefix('al_')
    aluminum_df.rename(columns={'al_region': 'region'}, inplace=True)


    num_rows = regional_df.shape[0]
    global_df = global_df.loc[global_df.index.repeat(num_rows)].reset_index(drop=True)
    product_df = pd.merge(copper_df, aluminum_df, on='region')
    merged_df = pd.merge(regional_df, product_df, on='region')
    result = pd.concat([merged_df,global_df], axis = 1)

    # print(result.columns)

    return result


test2_df = pd.DataFrame([{'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':1,
       'attribute_1_max':4, 'attribute_2_min':0, 'attribute_2_max':10}])

def get_true_mins_maxes(df_input, num_attributes = 5):
    """
    This function is applied to the whole dataframe and returns a whole
    new dataframe.
    assume attrbute values for all attributes from 1 to 5.
    assume attribute mins and maxes.
    """
    df = df_input.copy()
    for i in range(1,num_attributes + 1):
        max_observed = max(df[f"cu_attribute_{i}_value"].max(), df[f"al_attribute_{i}_value"].max(), df[f"attribute_{i}_max"].max())
        min_observed = min(df[f"cu_attribute_{i}_value"].min(), df[f"al_attribute_{i}_value"].min(), df[f"attribute_{i}_min"].min())
        if max_observed > df[f"attribute_{i}_max"].max():
            df[f"attribute_{i}_max"] = max_observed
        if min_observed < df[f"attribute_{i}_min"].min():
            df[f"attribute_{i}_min"] = min_observed
    return df

def normalize_attributes(input_df, num_attributes = 5):
    """
    Applied to entire dataframe. Assumes that all values are within range of min max.
    """
    df = input_df.copy()
    for i in range(1, num_attributes + 1):
        direction = df[f"direction_attribute_{i}"]
        if (direction == "positive").all():
            for m in ["cu","al"]:
                df[f"{m}_a{i}_callibrated"] = (df[f"{m}_attribute_{i}_value"]- df[f"attribute_{i}_min"])/(df[f"attribute_{i}_max"]-df[f"attribute_{i}_min"])
        elif (direction == "negative").all():
            for m in ["cu","al"]:
                df[f"{m}_a{i}_callibrated"] = ( df[f"attribute_{i}_max"]-df[f"{m}_attribute_{i}_value"])/(df[f"attribute_{i}_max"]-df[f"attribute_{i}_min"])
    return df


result = parse_pdf("Final Product Variables.pdf")
result = calc_product_cost(result)
result = get_true_mins_maxes(result)
result = normalize_attributes(result)
print(result)
# print(get_true_mins_maxes(test2_df,1))

# global_data, regional_data, product_data = parse_pdf("Final Product Variables.pdf")
# global_data = global_data.loc[global_data.index.repeat(8)].reset_index(drop=True)
# merged_df = pd.merge(regional_data, product_data, on='region')
# result = pd.concat([merged_df,global_data], axis = 1)
# print(merged_df)
# print(result)
########## DATA STRUCTURE NOTES

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


###### ALL FUNCTIONS BELOW ARE APPLIED TO THE WHOLE DF AND HAVE BEEN TESTED
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

def calc_utilities(input_df, num_attributes = 5):
    df = input_df.copy()
    for m in ["cu","al"]:
        df[f"{m}_utility"] = 0
        for i in range(1,num_attributes + 1):
            df[f"{m}_utility"] += (df[f"{m}_a{i}_callibrated"] * df[f"weight_attribute_{i}"])
    return df

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

def tau_callibrate_row(row):
    """
    callibrate tau per row
    """
    return tau_callibrate(row["cu_utility"], row["al_utility"],row["copper_product_market_share"])

def tau_callibrate_df(input_df):
    df = input_df.copy()
    df["tau_value"] = df.apply(tau_callibrate_row, axis = 1)
    return df

#################################
def calc_product_cost_row(row, cu_material_cost = None, al_material_cost = None, variable = "cu"):
    nonmaterial = row[f"{variable}_non_material_cost_per_unit"]
    copper_kg = row[f"{variable}_copper_kg"]
    aluminum_kg = row[f"{variable}_aluminum_kg"]
    if cu_material_cost is None:
        cu_material_cost = row["copper_price_per_kg"]
    if al_material_cost is None:
        al_material_cost = row["aluminum_price_per_kg"]
    result = nonmaterial + copper_kg * cu_material_cost + al_material_cost * aluminum_kg
    return result

def normalize_product_cost_row(row, product_cost):
    max_product_cost = row[f"attribute_1_max"]
    min_product_cost = row[f"attribute_1_min"]
    result = (max_product_cost - product_cost)/(max_product_cost - min_product_cost)
    return result

def calc_utility_row(row, cu_material_cost = None, al_material_cost = None, variable = "cu", num_attributes = 5):
    # Get copper
    cu_product_cost = calc_product_cost_row(row, cu_material_cost =cu_material_cost, al_material_cost=al_material_cost, variable = "cu")
    cu_product_cost = normalize_product_cost_row(row, cu_product_cost)
    cu_utility = cu_product_cost * row["weight_attribute_1"]
    for i in range(2,num_attributes + 1):
        cu_utility += (row[f"cu_a{i}_callibrated"] * row[f"weight_attribute_{i}"])
    # Get aluminum
    al_product_cost = calc_product_cost_row(row, cu_material_cost =cu_material_cost, al_material_cost=al_material_cost, variable = "al")
    al_product_cost = normalize_product_cost_row(row, al_product_cost)
    al_utility = al_product_cost * row["weight_attribute_1"]
    for i in range(2,num_attributes + 1):
        al_utility += (row[f"al_a{i}_callibrated"] * row[f"weight_attribute_{i}"])
    if variable == "cu":
        return cu_utility
    elif variable == "al":
        return al_utility

def point_generation_price(input_df, region, price_range = np.arange(0,5, 0.01), variable="cu", num_attributes = 5):
    """
    return a series of points. Assumes only one row with the region value.
    """
    ms_points = []
    region_row = input_df.loc[input_df["region"] == region].iloc[0]
    tau = region_row["tau_value"]
    if variable == "cu":
        for price in price_range:
            al_utility = calc_utility_row(region_row, cu_material_cost = price, num_attributes=num_attributes, variable = "al")
            cu_utility = calc_utility_row(region_row,cu_material_cost = price, num_attributes = num_attributes, variable = "cu")
            ms = ms_logit(cu_utility, al_utility, tau)
            ms_points.append(ms)
    elif variable == "al":
        for price in price_range:
            al_utility = calc_utility_row(region_row, al_material_cost = price, num_attributes=num_attributes, variable = "al")
            cu_utility = calc_utility_row(region_row, al_material_cost = price, num_attributes = num_attributes, variable = "cu")
            ms = ms_logit(cu_utility, al_utility, tau)
            ms_points.append(ms)
    else:
        raise ValueError("Invalid input")
    return ms_points

# def point_generation_ratio(input_df, region, hold = "al", hold_value = 2,ratio_range =np.arange(0.1,2, 0.01),num_attributes=5):
#     ms_points = []
#     region_row = input_df.loc[input_df["region"] == region].iloc[0]
#     tau = region_row["tau_value"]
#     if hold == "al":
#         al_utility = calc_utility_row(region_row, hold_value, "al", num_attributes=num_attributes)
#         for ratio in ratio_range:
#             price = ratio*hold_value
#             cu_utility = calc_utility_row(region_row,price, "cu", num_attributes = num_attributes)
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     elif hold == "cu":
#         cu_utility = calc_utility_row(region_row, hold_value, "cu", num_attributes=num_attributes)
#         for ratio in ratio_range:
#             price = hold_value/ratio
#             cu_utility = calc_utility_row(region_row, price, "al",num_attributes = num_attributes)
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     else:
#         raise ValueError("Invalid input")
#     return ms_points

def generate_graph(df, region, x,y):
    plt.figure(figsize=(7, 4.5))

    # main line (model)
    plt.plot(x, y, linewidth=2.5, color="tab:blue", label="Computed Points")
    # axes limits
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    region_row = df[df["region"] == region].iloc[0]
    x_special = region_row["copper_price_per_kg"]
    y_special = region_row["copper_product_market_share"]
    plt.scatter(
        x_special,
        y_special,
        color="red",
        s=100,           # marker size
        marker="o",      # circle
        zorder=5,        # draw on top of the line
        label="Observed point"
    )
    # plt.xlabel(xlabel, fontsize=11)
    plt.ylabel("Copper Product Market Share", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()
####################### TESTING

result = parse_pdf("Final Product Variables.pdf")
result = calc_product_cost(result)
# print(f"max is {result["attribute_1_max"]}")
# print(f"min is {result["attribute_1_min"]}")
result = get_true_mins_maxes(result)
# print(f"max is {resulgit["attribute_1_max"]}")
# print(f"min is {result["attribute_1_min"]}")
result = normalize_attributes(result)
result = calc_utilities(result)
result = tau_callibrate_df(result)
print(result.columns)
print(result[result["region"] == "India"][["copper_product_market_share", "copper_price_per_kg"]])
x = np.arange(0.1,20, 0.1)
y = point_generation_price(result,"India", price_range = x)
generate_graph(result, "India", x, y)
current_row = result.loc[result["region"]== "India"].iloc[0]
# print(y)

# cu_utility = calc_utility_row(current_row, 10.1)
# cu_utility2 = calc_utility_row(current_row)
# al_utility = calc_utility_row(current_row, variable = "al")
# normal1 = normalize_price_row(current_row, 10.1)
# normal2 = current_row["cu_a1_callibrated"]
# tau_val = current_row["tau_value"]
# print(current_row["direction_attribute_1"])
# print(f"max is {current_row["attribute_1_max"]}")
# print(f"min is {current_row["attribute_1_min"]}")
# print(f"normal_1 is {normal1}")
# print(f"normal_2 is {normal2}")
# print(f"cu_utility_1 is {cu_utility}")
# print(f"cu_utility_2 is {cu_utility2}")
# print(al_utility)
# print(tau_val)
# print(ms_logit(cu_utility, al_utility,tau_val))
# print(get_true_mins_maxes(test2_df,1))

# global_data, regional_data, product_data = parse_pdf("Final Product Variables.pdf")
# global_data = global_data.loc[global_data.index.repeat(8)].reset_index(drop=True)
# merged_df = pd.merge(regional_data, product_data, on='region')
# result = pd.concat([merged_df,global_data], axis = 1)
# print(merged_df)
# print(result)
########## DATA STRUCTURE NOTES

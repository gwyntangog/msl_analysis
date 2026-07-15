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
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit

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

####################333 PARSING
from io import StringIO
import tokenize

def strip_comments(text):
    tokens = tokenize.generate_tokens(StringIO(text).readline)
    return tokenize.untokenize(
        (tok_type, tok_string)
        for tok_type, tok_string, *_ in tokens
        if tok_type != tokenize.COMMENT
    )

def parse_pdf(pdf_path):

    """
    Returns a dataframe that combines global df, regional df, product df
    into one new dataframe.
    """
    variables = {}

    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    text = strip_comments(text)
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

def point_generation_ratio(input_df, region, hold = "al", hold_value = 2,ratio_range =np.arange(0.1,2, 0.01),num_attributes=5):
    """
    Change ratios.
    """
    ms_points = []
    region_row = input_df.loc[input_df["region"] == region].iloc[0]
    tau = region_row["tau_value"]
    if hold == "al":
        for ratio in ratio_range:
            al_material_cost = hold_value
            cu_material_cost = hold_value*ratio
            al_utility = calc_utility_row(region_row, cu_material_cost = cu_material_cost, al_material_cost=al_material_cost, num_attributes=num_attributes, variable = "al")
            cu_utility = calc_utility_row(region_row,cu_material_cost = cu_material_cost, al_material_cost = al_material_cost, num_attributes = num_attributes, variable = "cu")
            ms = ms_logit(cu_utility, al_utility, tau)
            ms_points.append(ms)
    elif hold == "cu":
        for ratio in ratio_range:
            al_material_cost = hold_value/ratio
            cu_material_cost = hold_value
            al_utility = calc_utility_row(region_row, cu_material_cost = cu_material_cost, al_material_cost=al_material_cost, num_attributes=num_attributes, variable = "al")
            cu_utility = calc_utility_row(region_row,cu_material_cost = cu_material_cost, al_material_cost = al_material_cost, num_attributes = num_attributes, variable = "cu")
            ms = ms_logit(cu_utility, al_utility, tau)
            ms_points.append(ms)
    else:
        raise ValueError("Invalid input")
    return ms_points

def generate_graph(df, region, x,y, xlabel = None, material = None, save = True):
    product = df["cu_product"].iloc[0]
    if save:
        plt.figure(figsize=(7, 4.5))
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.plot(x, y, linewidth=2.5, color="tab:blue", label="Computed Points")
        plt.suptitle(f"{product}, {region}",fontsize=13, fontweight="bold")
        point_label = "Observed point"
    else:
        plt.plot(x, y, linewidth=2.5, label=region)
        plt.suptitle(f"{product}, All regions",fontsize=13, fontweight="bold")
        point_label = f"Observed point ({region})"
    region_row = df[df["region"] == region].iloc[0]
    if material == "cu":
        x_special = region_row["copper_price_per_kg"]
        y_special = region_row["copper_product_market_share"]
        plt.scatter(
            1000*x_special,
            y_special,
            s=100,           # marker size
            marker="o",      # circle
            zorder=5,        # draw on top of the line
            label = point_label

        )
    elif material == "al":
        x_special = region_row["aluminum_price_per_kg"]
        y_special = region_row["copper_product_market_share"]
        plt.scatter(
            1000*x_special,
            y_special,
            s=100,           # marker size
            marker="o",      # circle
            zorder=5,        # draw on top of the line
            label=point_label
        )
    if xlabel:
        plt.xlabel(xlabel, fontsize=11)
        plt.title(f"{xlabel} vs Copper Product Market Share")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.ylabel("Copper Product Market Share", fontsize=11)
    plt.legend()
    plt.tight_layout()
    if save:
        os.makedirs(f"iter9_graphs/{product}/{xlabel}", exist_ok=True)
        plt.savefig(
                f"iter9_graphs/{product}/{xlabel}/{region}.png",
                dpi=300,
                bbox_inches="tight"
            )
        plt.close()

def generate_master_graph(df,x, xlabel = None, material = None):
    plt.figure(figsize=(7, 4.5))
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    for region in df["region"].tolist():
        if material == "cu":
            y = point_generation_price(df, region, price_range = x, variable="cu", num_attributes = 5)
            generate_graph(df, region, 1000*x,y, xlabel, material, save = False)
        elif material == "al":
            y = point_generation_price(df, region, price_range = x, variable="al", num_attributes = 5)
            generate_graph(df, region, 1000*x,y, xlabel, material, save = False)
        else:
            y = point_generation_ratio(df, region, hold = "al", hold_value = 2,ratio_range = x,num_attributes=5)
            generate_graph(df, region, x,y, xlabel, material, save = False)
    product = df["cu_product"].iloc[0]
    plt.legend(fontsize="x-small")
    os.makedirs(f"iter9_graphs/{product}/{xlabel}", exist_ok=True)
    plt.savefig(
            f"iter9_graphs/{product}/{xlabel}/All.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()

def step_tau_row(row, max_steps = 50):
    #'weight_attribute_1', 'weight_attribute_2', 'weight_attribute_3',
    #'weight_attribute_4', 'weight_attribute_5'
    new_row = row.copy()
    cu_utility = calc_utility_row(row, variable = "cu")
    al_utility = calc_utility_row(row, variable = "al")
    market_share_cu = row["copper_product_market_share"]
    current_tau = tau_callibrate(cu_utility, al_utility, market_share_cu)
    new_row["tau_value"] = current_tau
    if current_tau <= 0:
        new_row["weight_attribute_2"] = new_row["weight_attribute_2"]+0.1
        new_row["weight_attribute_1"] = new_row["weight_attribute_1"]-0.025
        new_row["weight_attribute_3"] = new_row["weight_attribute_3"]-0.025
        new_row["weight_attribute_4"] = new_row["weight_attribute_4"]-0.025
        new_row["weight_attribute_5"] = new_row["weight_attribute_5"]-0.025
        return step_tau_row(new_row, max_steps -1)
    else:
        return new_row

def step_tau_df(df):
    """
    returns a new df
    """
    result = pd.DataFrame()
    for _, row in df.iterrows():
        new_row = step_tau_row(row)
        result = pd.concat([result, new_row], axis = 1)
    result = result.T
    return result

def sanity_check(df, num_attributes = 5):
    """
    Checks the following:
        1. Were min maxes true or not?
        2. Are all normalized or not?
    """
    result = {}
    # CHECK MINS AND MAXES
    for i in range(1,num_attributes + 1):
        max_observed = max(df[f"cu_attribute_{i}_value"].max(), df[f"al_attribute_{i}_value"].max(), df[f"attribute_{i}_max"].max())
        min_observed = min(df[f"cu_attribute_{i}_value"].min(), df[f"al_attribute_{i}_value"].min(), df[f"attribute_{i}_min"].min())
        if max_observed > df[f"attribute_{i}_max"].max():
            result[f"a{i}_max"] = "ERROR"
        else:
            result[f"a{i}_max"] = "CORRECT"
        if min_observed < df[f"attribute_{i}_min"].min():
             result[f"a{i}_min"] = "ERROR"
        else:
            result[f"a{i}_min"] = "CORRECT"
    # CHECK NORMALIZED
    for i in range(1,num_attributes + 1):
        att_values = df[f"cu_attribute_{i}_value"].tolist() + df[f"al_attribute_{i}_value"].tolist()
        total = len(att_values)
        num_normalized = 0
        for one_val in att_values:
            if  (0 <= one_val and  one_val <= 1):
                num_normalized += 1
        if num_normalized == total:
            result[f"a{i}_values"] = "NORMALIZED"
        elif num_normalized == 0:
            result[f"a{i}_values"] = "NOT NORMALIZED"
        else:
            result[f"a{i}_values"] = "INCONSISTENT"
    return result

################################################ Add to all rows, do testing

def find_poly_fit(x,y, max_deg = 1):
    """
    test power curve
    """
    ##### POLY FIT
    polyfits = {}
    errors = {}
    current_error = np.inf
    current_eqtn = None
    current_degree = None
    for i in range(1,max_deg +1):
        p_fitted = Polynomial.fit(x, y, i)
        y_pred = p_fitted(x)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        polyfits[f"fit_{i}"] = p_fitted
        errors[f"fit_{i}"] = rmse
        if rmse < current_error:
            current_degree = i
            current_error = rmse
            current_eqtn = p_fitted
    return {"poly_error": current_error, "poly_degree": current_degree,"poly_equation": current_eqtn}

def find_power_fit(x,y):

    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    popt, pcov = curve_fit(exponential_model, x, y)

    # print(f"Best-fit equation: y = {popt[0]:.2f} * e^({popt[1]:.2f} * x)")

    def equation(x):
        return (popt[0] * (np.e**(popt[1] * x)))

    y_pred = equation(x)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    alpha, beta = popt
    return {"power_error": rmse, "power_alpha": alpha, "power_beta": beta}

def find_logit_fit(ratios, shares, s_min=0, s_max = 1):
    shares = np.array(shares)
    ratios = np.array(ratios)
    assert np.all(shares > s_min),  "S_c must be > S_min for log to be defined"
    assert np.all(shares < s_max),  "S_c must be < S_max for log to be defined"
    assert len(shares) >= 2,        "Need at least 2 data points"

    # ── Build linearised system ──────────────────────────────────────────────────
    #   A * x  -  B  =  y
    #   where A = alpha,  B = alpha * beta
    # ────────────────────────────────────────────────────────────────────────────
    y = np.log((s_max - s_min) / (shares - s_min) - 1)   # RHS
    x = ratios                                      # normalised pressure

    M = np.column_stack([x, -np.ones_like(x)])          # design matrix [x | -1]

    # ── Solve (least squares if overdetermined) ──────────────────────────────────
    (A, B), residuals, rank, sv = np.linalg.lstsq(M, y, rcond=None)

    alpha = A
    beta  = B / A

    print(f"alpha = {alpha:.6f}")
    print(f"beta  = {beta:.6f}")

    # ── Verification: reconstruct S_c and compare ─────────────────────────────────
    y_pred = s_min + (s_max - s_min) / (1 + np.exp(alpha * (x - beta)))

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return {"logit_error": rmse, "logit_alpha":alpha, "logit_beta":beta}

def find_fit(filename, region = "India"):
    """Try to find a best fit line given that x and y where y is the ratio"""
    result = parse_pdf(filename)
    result = calc_product_cost(result)
    result = get_true_mins_maxes(result)
    result = normalize_attributes(result)
    result = calc_utilities(result)
    result = tau_callibrate_df(result)
    result = step_tau_df(result)

    x = np.arange(0.1,3,0.1)
    y = point_generation_ratio(result,region, ratio_range = x)

    # line_fit = find_poly_fit(x,y)
    # power_fit = find_power_fit(x,y)
    # logit_fit = find_logit_fit(x, y, s_min=0, s_max=1)

    return try_all_fits(x,y)

def try_all_fits(x,y):
    poly_fit = find_poly_fit(x,y)
    power_fit = find_power_fit(x,y)
    logit_fit = find_logit_fit(x, y, s_min=0, s_max=1)
    poly_error = poly_fit["poly_error"]
    power_error = power_fit["power_error"]
    logit_error = logit_fit["logit_error"]
    best_dict = {}
    if poly_error < power_error and poly_error < logit_error:
        best_dict["best"] = "Poly"
    elif power_error < poly_error and power_error < logit_error:
        best_dict["best"] = "Power"
    else:
        best_dict["best"] = "Logit"
    return poly_fit | power_fit | logit_fit | best_dict


################RUN ALL
def run_through_file(filename):
    result = parse_pdf(filename)
    result = calc_product_cost(result)

    product = result["cu_product"].iloc[0]
    sanity_df = sanity_check(result)
    sanity_df = pd.DataFrame([sanity_df])
    sanity_df.to_csv(f"iter9_graphs/{product}/sanity_check.csv")

    result = get_true_mins_maxes(result)
    result = normalize_attributes(result)
    result = calc_utilities(result)
    result = tau_callibrate_df(result)
    result = step_tau_df(result)
    result.to_csv(f"iter9_graphs/{product}/overall_data.csv", index=False)

    fit_results = []


    # GRAPHING
    regions_list = result["region"].tolist()
    for region in regions_list:
        x = np.arange(0.1,20, 0.1)
        y = point_generation_price(result,region, price_range = x, variable = "cu")
        generate_graph(result, region, 1000*x, y, xlabel ="Copper Price (dollars per tonne)", material = "cu")
        y = point_generation_price(result,region, price_range = x, variable = "al")
        generate_graph(result, region, 1000*x, y, xlabel ="Aluminum Price (dollars per tonne)", material = "al")
        x = np.arange(0.1,3,0.1)
        y = point_generation_ratio(result,region, ratio_range = x)
        generate_graph(result, region, x, y, xlabel ="Ratio of Copper Price to Aluminum Price")
        fit_results.append({"region":region}|try_all_fits(x,y))
    x = np.arange(0.1,20, 0.1)
    generate_master_graph(result, x, xlabel ="Copper Price (dollars per tonne)", material = "cu" )
    generate_master_graph(result, x,  xlabel ="Aluminum Price (dollars per tonne)", material = "al" )
    x = np.arange(0.1,3,0.1)
    generate_master_graph(result, x,  xlabel ="Ratio of Copper Price to Aluminum Price", material = None )

    fit_results = pd.DataFrame(fit_results)
    fit_results.to_csv(f"iter9_graphs/{product}/fit_results.csv", index=False)
    return


####################### TESTING

# print(find_fit('iter9_pdfs/wire_harness.pdf'))
# run_through_file('iter9_pdfs/interconnect.pdf')
# run_through_file('iter9_pdfs/busbar.pdf')
# run_through_file('iter9_pdfs/motor_winding.pdf')
run_through_file('iter9_pdfs/wire_harness.pdf')
# run_through_file('iter9_pdfs/ice_busbar.pdf')
# run_through_file('iter9_pdfs/ice_wire_harness.pdf')
# run_through_file('iter9_pdfs/ice_alternator.pdf')

# folder_path = Path("iter9_pdfs")

# # Loop through all CSV files in the folder
# for file_path in folder_path.glob("*.pdf"):
#     run_through_file(file_path)

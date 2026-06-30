#############

from pathlib import Path
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import os

############ PARSE PDFs HERE

from pathlib import Path
import fitz
import re
import ast
import pandas as pd

#TODO: normalize attribute 1 based on copper price and aluminum price and product price

############
def analysis(global_df, regional_df, product_df, product_name):


    def tau_callibrate(df):
        """
        Callibrate the value of tau such that the logit function returns the same value
        as the current MARKET_SHARE_CU. See derivation in Overleaf file.
        """
        utility_cu = df["cu_utility"]
        utility_al = df["al_utility"]
        market_share_cu = df["cu_market_share"]

        mc = min(max(market_share_cu, 1e-6), 1-1e-6)
        tau = (utility_cu-utility_al)/(math.log(mc)-math.log(1-mc))
        return tau

    ######## tau_callibrate_step

    def tau_callibrate_step(current_df, max_iter=50):

        for iter_num in range(max_iter):

            current_tau = tau_callibrate(current_df)
            #assert(ms_logit(current_df["cu_utility"],current_df["al_utility"],current_tau)==current_df["cu_market_share"])
            # assert np.isclose(
            #     ms_logit(current_df["cu_utility"],
            #             current_df["al_utility"],
            #             current_tau),
            #     current_df["cu_market_share"],
            #     rtol=1e-9,
            #     atol=1e-12
            # )
            # pred = ms_logit(current_df["cu_utility"],current_df["al_utility"],current_tau)
            # if not np.isclose(pred, current_df["cu_market_share"], rtol=1e-9, atol=1e-12):
            #     print("----------------")
            #     print(current_df["region"], current_df["dominantmaterial"])
            #     print("u_cu =", current_df["cu_utility"])
            #     print("u_al =", current_df["al_utility"])
            #     print("market =", current_df["cu_market_share"])
            #     print("tau =", current_tau)
            #     print("pred =", pred)
            #     print("difference =", pred - current_df["cu_market_share"])
            #     raise AssertionError
            if current_tau > 0:
                current_df["num_callibrations"] = iter_num
                current_df["tau_value"] = current_tau
                return current_df



            current_df["a2_weight"] += 0.1
            current_df["a1_weight"] -= 0.025
            current_df["a3_weight"] -= 0.025
            current_df["a4_weight"] -= 0.025
            current_df["a5_weight"] -= 0.025

            # Bound
            current_df["a1_weight"] = np.clip(current_df["a1_weight"], 0.01, 0.99)
            current_df["a2_weight"] = np.clip(current_df["a2_weight"], 0.01, 0.99)
            current_df["a3_weight"] = np.clip(current_df["a3_weight"], 0.01, 0.99)
            current_df["a4_weight"] = np.clip(current_df["a4_weight"], 0.01, 0.99)
            current_df["a5_weight"] = np.clip(current_df["a5_weight"], 0.01, 0.99)

            # recompute utilities
            new_cu = 0
            new_al = 0
            for i in range(1,6):
                new_cu += current_df[f"a{i}_weight"] * current_df[f"cu_a{i}_val"]
                new_al += current_df[f"a{i}_weight"] * current_df[f"al_a{i}_val"]

            current_df["cu_utility"] = new_cu
            current_df["al_utility"] = new_al
            current_df["num_callibrations"] = iter_num
            current_df["tau_value"] = current_tau

        return current_df

    def ms_logit(u_cu, u_al, tau):
        if tau <= 0 or not np.isfinite(tau):
            return np.nan

        z_cu = u_cu / tau
        z_al = u_al / tau

        z_max = max(z_cu, z_al)
        # softmax form
        exp_cu = np.exp(z_cu - z_max)
        exp_al = np.exp(z_al - z_max)

        total = exp_cu + exp_al

        if total == 0:
            return np.nan

        result = exp_cu/total
        result = np.clip(result, 10**(-9),10**(9))
        return result

    # COLLECT DATA FOR EACH REGION

    product_df.to_csv("test_product.csv")

    regions_list = regional_df["region"].unique()
    new_df_data = []

    for region in regions_list:
        current_region = regional_df[regional_df["region"] == region]

        cu_price = current_region["copper_price_per_kg"].iloc[0]
        al_price = current_region["aluminum_price_per_kg"].iloc[0]

        region_result = {}

        region_result["region"] = region
        region_result["cu_material_cost"] = cu_price
        region_result["al_material_cost"] = al_price
        region_result["cu_market_share"] = current_region["copper_product_market_share"].iloc[0]
        region_result["al_market_share"] = current_region["aluminum_product_market_share"].iloc[0]

        for i in range(1, 6):
            region_result[f"a{i}_weight"] = current_region[f"weight_attribute_{i}"].iloc[0]

        for i in range(1, 6):
            region_result[f"a{i}_direction"] = global_df[f"direction_attribute_{i}"].iloc[0]

        for i in range(1, 6):
            region_result[f"a{i}_max"] = global_df[f"attribute_{i}_max"].iloc[0]

        for i in range(1, 6):
            region_result[f"a{i}_min"] = global_df[f"attribute_{i}_min"].iloc[0]


        # for i in range(1, 6):
        #     region_result[f"a{i}_direction"] = global_df[f"direction_attribute_{i}"].iloc[0]

        region_df = product_df[product_df["region"] == region]

        for _, row in region_df.iterrows():

            if row["dominant material"] == "cu":
                region_result["cu_copper_kg"] = row["copper_kg"]
                region_result["cu_aluminum_kg"] = row["aluminum_kg"]
                region_result["cu_nonmaterial"] = row["non_material_cost_per_unit"]

                region_result["cu_a1_val"] = (
                    row["non_material_cost_per_unit"]
                    + row["copper_kg"] * cu_price
                    + row["aluminum_kg"] * al_price
                )

                for i in range(2, 6):
                    region_result[f"cu_a{i}_val"] = row[f"attribute_{i}_value"]


            elif row["dominant material"] == "al":
                region_result["al_copper_kg"] = row["copper_kg"]
                region_result["al_aluminum_kg"] = row["aluminum_kg"]
                region_result["al_nonmaterial"] = row["non_material_cost_per_unit"]

                region_result["al_a1_val"] = (
                    row["non_material_cost_per_unit"]
                    + row["copper_kg"] * cu_price
                    + row["aluminum_kg"] * al_price
                )

                for i in range(2, 6):
                    region_result[f"al_a{i}_val"] = row[f"attribute_{i}_value"]

        new_df_data.append(region_result)

    new_df = pd.DataFrame(new_df_data)

    # # CREATE NEW DATAFRAME

    new_df.to_csv("test_post.csv")

    for att_num in range(1,6):
        # asset min and max
        att_direction = new_df[f"a{att_num}_direction"][0]
        max_val =  new_df[f"a{att_num}_max"][0]
        min_val =  new_df[f"a{att_num}_min"][0]
        max_val = max(max(new_df[f"cu_a{att_num}_val"]),max(new_df[f"al_a{att_num}_val"]),max_val)
        min_val = min(min(new_df[f"cu_a{att_num}_val"]),min(new_df[f"al_a{att_num}_val"]), min_val)
        difference = abs(max_val - min_val)
        # update columns
        new_df[f"a{att_num}_max"] = max_val
        new_df[f"a{att_num}_min"] = min_val
        if difference == 0:
            difference = 1e-9
        if att_direction == "positive":
            new_df[f"cu_a{att_num}_callibrated"] = (new_df[f"cu_a{att_num}_val"]- min_val)/difference
            new_df[f"al_a{att_num}_callibrated"] = (new_df[f"al_a{att_num}_val"]- min_val)/difference
        elif att_direction == "negative":
            new_df[f"cu_a{att_num}_callibrated"] = (max_val - new_df[f"cu_a{att_num}_val"])/difference
            new_df[f"al_a{att_num}_callibrated"] = (max_val - new_df[f"al_a{att_num}_val"])/difference


    # CALCULATE UTILITIES

    for m in ["cu","al"]:
        new_df[f"{m}_utility"] = 0
        for i in range(1,6):
            addition = new_df[f"{m}_a{i}_callibrated"]*new_df[f"a{i}_weight"]
            new_df[f"{m}_utility"] += addition
    new_df["cu_other_utility"] = new_df["cu_utility"] - new_df["cu_a1_callibrated"] * new_df["a1_weight"]
    new_df["al_other_utility"] = new_df["al_utility"] - new_df["al_a1_callibrated"] * new_df["a1_weight"]
    # CALCULATE TAU VALUES
    new_df.to_csv("test_tau.csv")
    # CALLIBRATE TAU VALUES
    new_df["num_callibrations"] = None
    new_df["tau_value"] = None

    new_df = new_df.apply(tau_callibrate_step, axis=1)

    new_df.to_csv("test_after_tau.csv")
    # GENERATE MARKET SHARES

    #edit this error
    def calc_product_cost(df, cu_material_cost =None, al_material_cost = None, current_product = "cu"):
        nonmaterial = df[f"{current_product}_nonmaterial"]
        copper_kg = df[f"{current_product}_copper_kg"]
        aluminum_kg = df[f"{current_product}_aluminum_kg"]
        if cu_material_cost == None:
            cu_material_cost = df["cu_material_cost"]
        if al_material_cost == None:
            al_material_cost = df["al_material_cost"]
        product_cost = nonmaterial + copper_kg * cu_material_cost + al_material_cost * aluminum_kg
        return product_cost

    def callibrate_product_cost(df, product_cost):
        max_val = df["a1_max"]
        difference = df["a1_max"] - df["a1_min"]
        result = (max_val - product_cost)/difference
        return result

    def calc_utility(df, cu_cost, al_cost, product = "cu"):
        if product == "cu":
            product_cost = calc_product_cost(df, cu_cost, al_cost, "cu")
        elif product == "al":
            product_cost = calc_product_cost(df, cu_cost, al_cost, "al")
        callibrated_product_cost = callibrate_product_cost(df, product_cost)
        utility = df["cu_other_utility"] + callibrated_product_cost * df["a1_weight"]
        return utility

    def gen_graph(region, prices, ms_vals, xlabel = ""):
        plt.ylim(-0.1, 1.1)
        plt.plot(prices, ms_vals)
        plt.xlabel(xlabel)
        plt.ylabel("Market Share of Copper Product")
        plt.title(f"Market Share Trend for {product_name} in {region}")
        os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
        plt.savefig(f"iter8_graphs/{product_name}/{region}_{xlabel}.png", dpi=300, bbox_inches="tight")
        plt.clf()
        # ZOOMED IN
        # plt.ylim(min(ms_vals), max(ms_vals))
        # plt.plot(prices, ms_vals)
        # plt.xlabel(xlabel)
        # plt.ylabel("Market Share of Copper Product")
        # plt.title(f"Market Share Trend for {product_name} in {region}")
        # os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
        # plt.savefig(f"iter8_graphs/{product_name}/{region}_{xlabel}_zoomed.png", dpi=300, bbox_inches="tight")
        # plt.clf()

    def get_power_constants(base_ratio, x,y):
        plt.plot(x, y)
        plt.show()
        plt.clf()

        BASE = base_ratio

        log_x = np.log(x)
        log_y = np.log(y)
        log_x = np.log(x)
        log_y = np.log(y)
        X = sm.add_constant(log_x)
        results = sm.OLS(log_y, X).fit()
        ln_A, beta = results.params
        beta = beta/np.log(base_ratio)
        A = np.exp(ln_A)
        results = {"A_val": A, "beta_val": beta}
        print("A,beta,base_ratio")
        print(A, beta, base_ratio)
        return results


    # GENERATE VALUES AND GRAPHS FOR EACH REGION
    # VARY COPPER AND GET POWER CONSTANTS
    power_constants = {}
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        prices = np.arange(5,20,0.1) #TODO Copper range
        al_cost = row["al_material_cost"]
        for cu_cost in prices:
            utility_cu = calc_utility(row, cu_cost, al_cost, "cu")
            utility_al = calc_utility(row, cu_cost, al_cost, "al")
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        gen_graph(region, prices, ms_vals, "Copper Price")

    # VARY THE RATIO
    # base_ratio = row["cu_material_cost"]/row["al_material_cost"]
    # power_constants[region] = get_power_constants(base_ratio, prices, ms_vals)



    # VARY ALUMINUM
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        prices = np.arange(1.5,5,0.1) #TODO Aluminum range
        cu_cost = row["cu_material_cost"]
        for al_cost in prices:
            # utility_cu = row["cu_utility"]
            # utility_al = calc_utility(row, cost, "al")
            utility_cu = calc_utility(row, cu_cost, al_cost, "cu")
            utility_al = calc_utility(row, cu_cost, al_cost, "al")
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        gen_graph(region, prices, ms_vals, "Aluminum Price")

    # VARY RATIO
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        al_price = 2 # default
        ratios = np.arange(2,6, 0.1)
        cu_prices = al_price* ratios
        # prices = range(1,150) #TODO Aluminum range
        for cu_price in cu_prices:
            utility_cu = calc_utility(row, cu_cost, al_cost, "cu")
            utility_al = calc_utility(row, cu_cost, al_cost, "al")
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        gen_graph(region, ratios, ms_vals, "Ratio of Copper Price to Aluminum Price")

    # delta_u = np.linspace(-5, 5, 500)

    # # Logit market share
    # market_share_cu = 1 / (1 + np.exp(-delta_u))

    # for key, value in power_constants.items():
    #     delta_u = np.linspace(-5, 5, 500)

    # # Logit market share
    #     A = value["A_val"]
    #     beta = value["beta_val"]
    #     ratio_points = np.array(np.arange(0.01,5, 0.1)) #TODO Chance ratio range
    #     # market_shares = A * (ratio_points** beta)
    #     log_market = np.log(A) + beta * np.log(ratio_points)
    #     log_market = np.clip(log_market, -700, 700)
    #     market_shares = np.exp(log_market)
    #     market_shares = np.clip(market_shares,0,1)
    #     # print(region)
    #     # print(f"ratio points {ratio_points}")
    #     # print(f"market  shares {market_shares}")
    #     gen_graph(key, ratio_points, market_shares, "Ratio of Copper Price to Aluminum Price")

    ###########3TEST
    return new_df[["num_callibrations","tau_value"]]

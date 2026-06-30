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

        mc = min(max(market_share_cu, 10**(-9)), 1-10**(-9))
        tau = (utility_cu-utility_al)/(np.log(mc)-np.log(1-mc))
        return tau

    ######## tau_callibrate_step

    def tau_callibrate_step(current_df, max_iter=50):

        for iter_num in range(max_iter):

            current_tau = tau_callibrate(current_df)
            print(f"at {iter_num} with tau_val {current_tau}")
            # if prev_tau is not None and abs(current_tau - prev_tau) < 1e-6:
            if current_tau >= 0.0000001:
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

            new_cu_other = 0
            new_al_other = 0
            for i in range(2,6):
                new_cu += current_df[f"a{i}_weight"] * current_df[f"cu_a{i}_val"]
                new_al += current_df[f"a{i}_weight"] * current_df[f"al_a{i}_val"]

            current_df["cu_utility"] = new_cu
            current_df["al_utility"] = new_al
            current_df["cu_other_utility"] = new_cu_other
            current_df["al_other_utility"] = new_al_other
            # RECALCULATE CU_OTHER UTILITIES
            current_df["num_callibrations"] = iter_num
            current_df["tau_value"] = current_tau

        return current_df

    ###### LOGIT HELPER
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
        result = np.clip(result, 10**(-9),1-10**(-9))
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
        if product == "cu":
            utility = df["cu_other_utility"] + callibrated_product_cost * df["a1_weight"]
        elif product == "al":
            utility = df["al_other_utility"] + callibrated_product_cost * df["a1_weight"]
        return utility

    def gen_graph(region, prices, ms_vals, xlabel="", subtitle=None):
        plt.figure(figsize=(7, 4.5))

        # main line (model)
        plt.plot(prices, ms_vals, linewidth=2.5, color="tab:blue", label="Model")

        # add light markers (helps see actual sampled points)
        plt.scatter(prices[::5], ms_vals[::5], s=18, color="black", alpha=0.6, label="Sample points")

        # axes limits
        plt.ylim(-0.05, 1.05)

        # grid (huge readability boost)
        plt.grid(True, linestyle="--", alpha=0.3)

        # labels
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel("Copper Market Share", fontsize=11)

        # titles
        plt.suptitle(f"{product_name} — {region}", fontsize=13, fontweight="bold")

        if subtitle:
            plt.title(subtitle, fontsize=10)

        # legend
        plt.legend()

        # layout tightening (prevents clipping)
        plt.tight_layout()

        # save
        os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
        plt.savefig(
            f"iter8_graphs/{product_name}/{region}_{xlabel}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    # def gen_graph(region, prices, ms_vals, xlabel = "", subtitle = None):
    #     plt.ylim(-0.1, 1.1)
    #     plt.plot(prices, ms_vals)
    #     plt.xlabel(xlabel)
    #     plt.ylabel("Market Share of Copper Product")
    #     plt.suptitle(f"Market Share Trend for {product_name} in {region}")
    #     if subtitle is not None:
    #         plt.title(subtitle)
    #     os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
    #     plt.savefig(f"iter8_graphs/{product_name}/{region}_{xlabel}.png", dpi=300, bbox_inches="tight")
    #     plt.clf()
        # ZOOMED IN
        # plt.ylim(min(ms_vals), max(ms_vals))
        # plt.plot(prices, ms_vals)
        # plt.xlabel(xlabel)
        # plt.ylabel("Market Share of Copper Product")
        # plt.title(f"Market Share Trend for {product_name} in {region}")
        # os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
        # plt.savefig(f"iter8_graphs/{product_name}/{region}_{xlabel}_zoomed.png", dpi=300, bbox_inches="tight")
        # plt.clf()

    # Get power constants

    # def get_power_constants(base_ratio, x,y):
    #     plt.plot(x, y)
    #     plt.show()
    #     plt.clf()

    #     BASE = base_ratio

    #     log_x = np.log(x)
    #     log_y = np.log(y)
    #     log_x = np.log(x)
    #     log_y = np.log(y)
    #     X = sm.add_constant(log_x)
    #     results = sm.OLS(log_y, X).fit()
    #     ln_A, beta = results.params
    #     beta = beta/np.log(base_ratio)
    #     A = np.exp(ln_A)
    #     results = {"A_val": A, "beta_val": beta}
    #     # print("A,beta,base_ratio")
    #     # print(A, beta, base_ratio)
    #     return results

    def fit_power_law(x, y):
        x = np.array(x)
        y = np.array(y)

        # remove invalid values
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]

        log_x = np.log(x)
        log_y = np.log(y)

        X = sm.add_constant(log_x)
        model = sm.OLS(log_y, X).fit()

        ln_a = model.params[0]
        b = model.params[1]
        a = np.exp(ln_a)

        return a, b, model

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
        # print(f"ms_vals are {ms_vals}")
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
        # print(f"ms_vals are {ms_vals}")
        gen_graph(region, prices, ms_vals, "Aluminum Price")

    # GRAPH
        # BASELINE CHECK THE POINT
        # INCLUDE THE AMOUNT OF ERROR
        # ADD TWO LINES TO BOTH
        # MAKE IT CUTE

    # VARY RATIO
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        al_price = 2.5 # default
        ratios = np.arange(2,6, 0.1)
        cu_prices = al_price* ratios
        # prices = range(1,150) #TODO Aluminum range
        for cu_price in cu_prices:
            utility_cu = calc_utility(row, cu_price, al_price, "cu")
            utility_al = calc_utility(row, cu_price, al_price, "al")
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        # print(f"ms_vals are {ms_vals}")
        print("power law results")
        print(fit_power_law(ratios, ms_vals))
        gen_graph(region, ratios, ms_vals, "Ratio of Copper Price to Aluminum Price")

        # POWER LAW RESULTS
        a,b, model = fit_power_law(ratios, ms_vals)
        x = np.arange(2,6, 0.1)
        y = a*(x**b)
        gen_graph(region, x, y, f"Fitted Ratio of Copper Price to Aluminum Price", f"Power curve with A = {np.round(a,3)}, beta = {np.round(b,3)}")

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
    return new_df[["cu_market_share", "al_market_share","cu_utility", "al_utility","num_callibrations","tau_value"]]

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

# all_pdfs = list(Path("iter8_pdfs").glob("*.pdf"))
# for pdf_path in all_pdfs:
#     variables = {}
#     with fitz.open(pdf_path) as doc:
#         text = "\n".join(page.get_text() for page in doc)

#     for name in ["global_data", "regional_data", "product_data"]:
#         match = re.search(
#             rf'{name}\s*=\s*(\{{.*?\}}|\[.*?\])',
#             text,
#             flags=re.DOTALL,
#         )
#         if match is None:
#             match = re.search(
#             rf'"{name}"\s*:\s*(\{{.*?\}}|\[.*?\])',
#             text,
#             flags=re.DOTALL,
#         )
#         if match:
#             variables[name] = ast.literal_eval(match.group(1))


#     global_data = variables.get("global_data")
#     regional_data = variables.get("regional_data")
#     product_data = variables.get("product_data")
#     product_name = product_data[0]["product"]


#     global_df = pd.DataFrame([global_data])
#     regional_df = pd.DataFrame(regional_data)
#     product_df = pd.DataFrame(product_data)




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
        # print(market_share_cu)
        mc = min(max(market_share_cu, 1e-6), 1-1e-6)
        tau = (utility_cu-utility_al)/(math.log(mc)-math.log(1-mc))
        return tau

    ######## tau_callibrate_step

    def tau_callibrate_step(df, max_iter=50):
        current_df = df.copy()

        for _ in range(max_iter):
            current_tau = tau_callibrate(current_df)

            if current_tau > 0:
                return current_tau

            current_df["a2_weight"] += 0.1
            current_df["a1_weight"] -= 0.2
            current_df["a3_weight"] -= 0.2
            current_df["a4_weight"] -= 0.2
            current_df["a5_weight"] -= 0.2

            # Bound
            current_df["a1_weight"] = np.clip(current_df["a1_weight"], 0.01, 0.99)
            current_df["a2_weight"] = np.clip(current_df["a2_weight"], 0.01, 0.99)
            current_df["a3_weight"] = np.clip(current_df["a3_weight"], 0.01, 0.99)
            current_df["a4_weight"] = np.clip(current_df["a4_weight"], 0.01, 0.99)
            current_df["a5_weight"] = np.clip(current_df["a5_weight"], 0.01, 0.99)

            # recompute utilities (MISSING IN YOUR CODE)
            new_cu = 0
            new_al = 0
            for i in range(1,6):
                new_cu += current_df[f"a{i}_weight"] * current_df[f"cu_a{i}_val"]
                new_al += current_df[f"a{i}_weight"] * current_df[f"al_a{i}_val"]

            current_df["cu_utility"] = new_cu
            current_df["al_utility"] = new_al

        return current_tau

    # def tau_callibrate_step(df, max_iter=50):
    #     current_df = df.copy()
    #     current_tau = tau_callibrate(current_df)
    #     if current_tau > 0:
    #         return current_tau
    #     else:
    #         # current_tau = tau_callibrate(current_df)
    #         # print(current_tau)
    #         # if current_tau > 0:
    #         #     return current_tau

    #         current_df["a2_weight"] += 0.1
    #         current_df["a1_weight"] -= 0.025
    #         current_df["a3_weight"] -= 0.025
    #         current_df["a4_weight"] -= 0.025
    #         current_df["a5_weight"] -= 0.025

    #         # recompute utilities (MISSING IN YOUR CODE)
    #         new_cu = 0
    #         new_al = 0
    #         for i in range(1,6):
    #             new_cu += current_df[f"a{i}_weight"] * current_df[f"cu_a{i}_val"]
    #             new_al += current_df[f"a{i}_weight"] * current_df[f"al_a{i}_val"]

    #         current_df["cu_utility"] = new_cu
    #         current_df["al_utility"] = new_al
    #         tau_callibrate_step(current_df)
    #     return


    # def tau_callibrate_step(df):
    #     current_tau = tau_callibrate(df)
    #     values = current_tau
    #     print(values)
    #     print(f"current tau {current_tau}")
    #     while current_tau<=0:
    #         new_df = df.copy()
    #         new_df["a2_weight"] = new_df["a2_weight"] + 0.1
    #         new_df["a1_weight"] = new_df["a1_weight"] - 0.20
    #         new_df["a3_weight"] = new_df["a3_weight"] - 0.20
    #         new_df["a4_weight"] = new_df["a4_weight"] - 0.20
    #         new_df["a5_weight"] = new_df["a5_weight"] - 0.20
    #         new_cu_utility = 0
    #         new_al_utility = 0
    #         for i in range(1,6):
    #             new_cu_utility += new_df[f"a{i}_weight"] * new_df[f"cu_a{i}_val"]
    #             new_al_utility += new_df[f"a{i}_weight"] * new_df[f"al_a{i}_val"]
    #         new_df["cu_utility"] = new_cu_utility
    #         new_df["al_utility"] = new_al_utility
    #         return tau_callibrate_step(new_df)
    #     return current_tau

    ####################

    # def ms_logit(utility_cu, utility_al, tau):
    #     """
    #     Calculates the market share of copper using the logit functions.
    #     utility_cu = utility of copper
    #     utility_al = utility of aluminum
    #     tau = scaling coefficient
    #     """
    #     e = math.e
    #     scaled_cu = utility_cu/tau
    #     print(f"tau value is {tau}")
    #     print(f"scaled cu value is {scaled_cu}")
    #     scaled_al = utility_al/tau
    #     cu_value = e**scaled_cu
    #     al_value = e**scaled_al
    #     total = cu_value + al_value
    #     market_share_cu = cu_value/total
    #     return market_share_cu
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

        return exp_cu / total

    # COLLECT DATA FOR EACH REGION

    product_df.to_csv("test_product.csv")
    # print(global_df.iloc[:,:2])

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
                    # print(f"cu a{i} val is { row[f"attribute_{i}_value"]} ")

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
        print(max_val, min_val)
        if difference == 0:
            difference = 1e-9
        if att_direction == "positive":
            new_df[f"cu_a{att_num}_callibrated"] = (new_df[f"cu_a{att_num}_val"]- min_val)/difference
            new_df[f"al_a{att_num}_callibrated"] = (new_df[f"al_a{att_num}_val"]- min_val)/difference
            # print(f"att_num is {att_num}")
            # print(f"max for a{att_num} is {max_val}")
            # print(f"min for a{att_num} is {min_val}")
            # print(f"difference is {difference}")
            # print(f"cu val is {new_df[f"cu_a{att_num}_val"]}")
            # print(f"al val is {new_df[f"al_a{att_num}_val"]}")
            # print(f"cu callibrated a{att_num} is {new_df[f"cu_a{att_num}_callibrated"]}")
            # print(f"al callibrated a{att_num} is {new_df[f"al_a{att_num}_callibrated"]}")
        elif att_direction == "negative":
            new_df[f"cu_a{att_num}_callibrated"] = (max_val - new_df[f"cu_a{att_num}_val"])/difference
            new_df[f"al_a{att_num}_callibrated"] = (max_val - new_df[f"al_a{att_num}_val"])/difference
            # print(f"att_num is {att_num}")
            # print(f"max for a{att_num} is {max_val}")
            # print(f"min for a{att_num} is {min_val}")
            # print(f"difference is {difference}")
            # print(f"cu val is {new_df[f"cu_a{att_num}_val"]}")
            # print(f"al val is {new_df[f"al_a{att_num}_val"]}")
            # print(f"cu callibrated a{att_num} is {new_df[f"cu_a{att_num}_callibrated"]}")
            # print(f"al callibrated a{att_num} is {new_df[f"al_a{att_num}_callibrated"]}")

    # CALCULATE UTILITIES

    for m in ["cu","al"]:
        new_df[f"{m}_utility"] = 0
        for i in range(1,6):
            addition = new_df[f"{m}_a{i}_callibrated"]*new_df[f"a{i}_weight"]
            new_df[f"{m}_utility"] += addition
            print(f"addition is {addition}")
            print(f"utility is { new_df[f"{m}_utility"]}")
    new_df["cu_other_utility"] = new_df["cu_utility"] - new_df["cu_a1_callibrated"] * new_df["a1_weight"]
    new_df["al_other_utility"] = new_df["al_utility"] - new_df["al_a1_callibrated"] * new_df["a1_weight"]
    # CALCULATE TAU VALUES
    new_df.to_csv("test_tau.csv")
    print(new_df[["region","cu_market_share","al_market_share","cu_utility","al_utility"]])

    # CALLIBRATE TAU VALUES

    new_df["tau_value"] = new_df.apply(tau_callibrate_step, axis=1)

    new_df.to_csv("test_after_tau.csv")
    # GENERATE MARKET SHARES

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

    def calc_utility(df, material_cost, material = "cu"):
        if material == "cu":
            product_cost = calc_product_cost(df, material_cost, None, "cu")
        elif material == "al":
            product_cost = calc_product_cost(df, None, material_cost, "al")
        callibrated_product_cost = callibrate_product_cost(df, product_cost)
        utility = df["cu_other_utility"] + callibrated_product_cost * df["a1_weight"]
        return utility

    def gen_graph(region, prices, ms_vals, xlabel = ""):
        plt.plot(prices, ms_vals)
        plt.xlabel(xlabel)
        plt.ylabel("Market Share of Copper Product")
        plt.title(f"Market Share Trend for {product_name} in {region}")
        os.makedirs(f"iter8_graphs/{product_name}", exist_ok=True)
        plt.savefig(f"iter8_graphs/{product_name}/{region}_{xlabel}.png", dpi=300, bbox_inches="tight")
        plt.clf()

    def get_power_constants(x,y):
        x = np.array(x)
        y = np.array(y)

        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return {"A_val": np.nan, "beta_val": np.nan}

        # log_x = np.log(x)
        # log_y = np.log(y)
        log_x = np.log(x)
        log_y = np.log(y)
        X = sm.add_constant(log_x)
        results = sm.OLS(log_y, X).fit()
        ln_A, beta = results.params
        A = np.exp(ln_A)
        results = {"A_val": A, "beta_val": beta}
        return results


    # GENERATE VALUES AND GRAPHS FOR EACH REGION
    # VARY COPPER AND GET POWER CONSTANTS
    power_constants = {}
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        prices = range(1,100)
        for cost in prices:
            utility_cu = calc_utility(row, cost,"cu")
            utility_al = row["al_utility"]
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        gen_graph(region, prices, ms_vals, "Copper Price")
        power_constants[region] = get_power_constants(prices, ms_vals)

    # VARY ALUMINUM
    for index, row in new_df.iterrows():
        region = row["region"]
        ms_vals = []
        prices = range(1,100)
        for cost in prices:
            utility_cu = row["cu_utility"]
            utility_al = calc_utility(row, cost, "al")
            tau = row["tau_value"]
            ms = ms_logit(utility_cu, utility_al, tau)
            ms_vals.append(ms)
        gen_graph(region, prices, ms_vals, "Aluminum Price")

    # VARY RATIO

    for key, value in power_constants.items():
        A = value["A_val"]
        beta = value["beta_val"]
        ratio_points = np.array(np.arange(0.1,2, 0.001))
        market_shares = A * (ratio_points** beta)
        gen_graph(key, ratio_points, market_shares, "Ratio of Copper Price to Aluminum Price")

    ###########3TEST

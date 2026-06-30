import pandas as pd
import numpy as np
import statsmodels.api as sm

# ----------------------------
# 1. LOAD DATA (you download these CSVs)
# ----------------------------

# material prooduction data
cu_data = pd.read_csv("copper_data.csv")
cu_data = cu_data[["Year","Primary production"]]
al_data = pd.read_csv("aluminum_data.csv")
al_data = al_data[["Year","Primary production"]]


# price data
cu_prices = pd.read_csv("copper-price.csv")
cu_prices["Year"] = cu_prices["date"].str[:4]
cu_prices["Year"] = cu_prices["Year"].astype(int)
cu_prices["price"] = cu_prices["copper_price_usd"]

al_prices = pd.read_csv("aluminum-price.csv")
al_prices["Year"] = al_prices["date"].str[:4]
al_prices["Year"] = al_prices["Year"].astype(int)
al_prices["price"] = al_prices["aluminum_price"]

def get_year_averages(df):
    """
    Given data set for prices, condense it to have one price per year.
    """
    years = cu_prices["Year"].unique()
    means = []
    for year in years:
        one_year_data = df[df["Year"]==year]
        year_mean = one_year_data["price"].mean()
        means.append(year_mean)
    result_df = pd.DataFrame(years, means).reset_index()
    result_df = result_df.rename(columns={'index': 'price', 0:'Year'})
    return result_df

cu_cleaned_prices = get_year_averages(cu_prices)
al_cleaned_prices = get_year_averages(al_prices)

print(cu_cleaned_prices)
print(al_cleaned_prices)
# ----------------------------
# 2. MERGE DATA
# ----------------------------

cu_df = cu_data.merge(cu_cleaned_prices, on="Year")
cu_df = cu_df.rename(columns={'Primary production': 'cu_tons', 'price': 'cu_price'})
cu_df['cu_tons'] = cu_df['cu_tons'].str.replace(',', '', regex=False)
cu_df['cu_tons'] = cu_df['cu_tons'].astype(int)
print(cu_df)

al_df = al_data.merge(al_cleaned_prices, on="Year")
al_df = al_df.rename(columns={'Primary production': 'al_tons', 'price': 'al_price'})
al_df['al_tons'] = al_df['al_tons'].str.replace(',', '', regex=False)
al_df['al_tons'] = al_df['al_tons'].astype(int)
print(al_df)

df = cu_df.merge(al_df, on = "Year")
print(df)

# ----------------------------
# 3. CREATE MARKET SHARE VARIABLES
# ----------------------------

df["q_ratio"] = df["cu_tons"] / df["al_tons"]
df["p_ratio"] = df["cu_price"] / df["al_price"]

# df = df.dropna()

df["ln_q"] = np.log(df["q_ratio"])
df["ln_p"] = np.log(df["p_ratio"])

print(df)

# ----------------------------
# 4. REGRESSION (CES SHARE EQUATION)
# ----------------------------

X = sm.add_constant(df["ln_p"])
y = df["ln_q"]

model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["ln_p"]

print(model.summary())

print("\nEstimated parameters:")
print("alpha =", alpha)
print("beta  =", beta)

# # ----------------------------
# # 5. RECOVER STRUCTURAL K
# # ----------------------------

K = np.exp(alpha)

print("\nStructural interpretation:")
print("K =", K)

# use the logit then fit to equation to find alpha and beta
# To Dos: get data for all locations
# Experimenting adding percentage to state (like 60%)
# 4 k's add up to 1 :3 catto

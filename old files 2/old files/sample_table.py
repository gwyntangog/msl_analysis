
# table
import pandas as pd

data_dict = {
        "region": "US",
        "material": "Al",
        "interconnect_copper_kg": 13,
        "interconnect_aluminum_kg": 45,
        "interconnect_total_mass_kg": 55,
        "interconnect_system_cost_usd": 420,
        "material_cost_share": 55,
        "non_material_cost_share": 45,
        "copper_share": 23,
        "aluminum_share": 77,
        "copper_conductivity_iacs": 100,
        "aluminum_conductivity_iacs": 61,
        "design_life_years": 12,
        "failure_rate_ppm": 12,
        "weight_cost": 40,
        "weight_conductivity": 25,
        "weight_weight": 20,
        "weight_reliability": 15,
    },

df = pd.DataFrame(data_dict)

print(df)


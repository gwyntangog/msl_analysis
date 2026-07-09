import unittest
import numpy as np
import pandas as pd
from functions import tau_callibrate, tau_callibrate_row, tau_callibrate_df, ms_logit, unzero
from functions import calc_product_cost, get_true_mins_maxes, normalize_attributes, calc_utilities
from functions import calc_product_cost_row, normalize_product_cost_row,  calc_utility_row, point_generation_price, point_generation_ratio

class testUnzero(unittest.TestCase):
    """Partitions
        - Normal case: a value between 0 to 1
        - Edge cases: 0 or less
        - Edge cases: 1 or higher
    """
    def test_unzero(self):
        self.assertEqual(unzero(-1),10**(-9))
        self.assertEqual(unzero(0),10**(-9))
        self.assertEqual(unzero(1),1-10**(-9))
        self.assertEqual(unzero(2),1-10**(-9))
        self.assertEqual(unzero(0.5),0.5)

class testTauFunctions(unittest.TestCase):
    """
    """
    def test_tau_callibrate1(self):
        utility_cu = 0.2
        utility_al = 0.8
        market_share_cu = 0.2
        expected_tau = 0.432809
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        tau = round(tau,6)
        self.assertEqual(tau, expected_tau)
    def test_tau_callibrate2(self):
        utility_cu = 0.7
        utility_al = 0.7
        market_share_cu = 0.5
        expected_tau = np.inf
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        self.assertEqual(tau, expected_tau)
    def test_tau_callibrate3(self):
        utility_cu = 0.7
        utility_al = 0.5
        market_share_cu = 0.3
        expected_tau = -0.236045
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        tau = round(tau,6)
        self.assertEqual(tau, expected_tau)

class testLogitFunctions(unittest.TestCase):
    """
    """
    def test_logit1(self):
        utility_cu = 0.2
        utility_al = 0.8
        tau = np.inf
        ms = ms_logit(utility_cu,utility_al,tau)
        expected_ms = 0.5
        self.assertEqual(ms, expected_ms)
    def test_logit2(self):
        utility_cu = 0.5
        utility_al = 0.5
        tau = 1
        ms = ms_logit(utility_cu,utility_al,tau)
        expected_ms = 0.5
        self.assertEqual(ms, expected_ms)
    def test_logit3(self):
        utility_cu = 0.2
        utility_al = 0.8
        tau = 2
        ms = ms_logit(utility_cu,utility_al,tau)
        ms = round(ms,6)
        expected_ms = 0.425557
        self.assertEqual(ms, expected_ms)

##########################################################################################################


class testProductCalc(unittest.TestCase):
    """
    """
    test_df1 = pd.DataFrame([{"cu_non_material_cost_per_unit": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "copper_price_per_kg": 7,
             "al_non_material_cost_per_unit": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "aluminum_price_per_kg": 3}])
    expected1 = pd.DataFrame([{"cu_non_material_cost_per_unit": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "copper_price_per_kg": 7,
             "al_non_material_cost_per_unit": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "aluminum_price_per_kg": 3,
             "cu_attribute_1_value":541, "al_attribute_1_value":573}])
    def test_prod_calc1(self):
        new_result= calc_product_cost(self.test_df1)
        pd.testing.assert_frame_equal(new_result, self.expected1)
#######
class testMinsMaxes(unittest.TestCase):
    """
    """
    test_df = pd.DataFrame([{'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':1,
       'attribute_1_max':4, 'attribute_2_min':0, 'attribute_2_max':10}])
    expected_df = pd.DataFrame([{'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10}])
    num_test_atts = 2
    def test_min_maxes(self):
        result = get_true_mins_maxes(self.test_df,self.num_test_atts)
        print(result)
        pd.testing.assert_frame_equal(result, self.expected_df)

##########

class testNormalizeAttributes(unittest.TestCase):
    """
    """
    test_df = pd.DataFrame([{'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                             'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10}])
    expected_df = pd.DataFrame([{'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0}])
    num_test_atts = 2
    def test_normalize(self):
        result = normalize_attributes(self.test_df,self.num_test_atts)
        print(result)
        pd.testing.assert_frame_equal(result, self.expected_df)


class testUtilityCalcs(unittest.TestCase):
    """
    """
    test_df = pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0}])
    expected_df = pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0,
       "cu_utility":0.65,"al_utility":0.7}])
    num_test_atts = 2
    def test_utility_df(self):
        result = calc_utilities(self.test_df,self.num_test_atts)
        print(result)
        pd.testing.assert_frame_equal(result, self.expected_df)

####

class testTauFunctions(unittest.TestCase):
    """
    """
    def test_tau_callibrate1(self):
        utility_cu = 0.2
        utility_al = 0.8
        market_share_cu = 0.2
        expected_tau = 0.432809
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        tau = round(tau,6)
        self.assertEqual(tau, expected_tau)
    def test_tau_callibrate2(self):
        utility_cu = 0.7
        utility_al = 0.7
        market_share_cu = 0.5
        expected_tau = np.inf
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        self.assertEqual(tau, expected_tau)
    def test_tau_callibrate3(self):
        utility_cu = 0.7
        utility_al = 0.5
        market_share_cu = 0.3
        expected_tau = -0.236045
        tau = tau_callibrate(utility_cu,utility_al,market_share_cu)
        tau = round(tau,6)
        self.assertEqual(tau, expected_tau)
    def test_tau_row(self):
        row = pd.DataFrame([{"cu_utility":0.7, "al_utility": 0.5, "copper_product_market_share": 0.3}]).iloc[0]
        expected = -0.236045
        result = round(tau_callibrate_row(row),6)
        self.assertEqual(result, expected)
    def test_tau_df(self):
        df = pd.DataFrame([{"cu_utility":0.7, "al_utility": 0.5, "copper_product_market_share": 0.3}])
        expected = -0.236045
        new_df = tau_callibrate_df(df)
        result = new_df["tau_value"].iloc[0]
        result = round(result,6)
        self.assertEqual(result, expected)

######################

from functions import sanity_check

class testSanityCheck(unittest.TestCase):
    """
    """
    test_df = pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':2,'al_attribute_2_value':2,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0}])
    def test_sanity1(self):
        expected = {'a1_max': 'CORRECT', 'a1_min': 'CORRECT', 'a1_values': 'NOT NORMALIZED', 'a2_max': 'CORRECT', 'a2_min': 'CORRECT', 'a2_values': 'NOT NORMALIZED'}
        result = sanity_check(self.test_df, 2)
        self.assertDictEqual(result, expected)
    test_df2 = pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':2,'al_attribute_2_value':0.5,'attribute_1_min':0,
       'attribute_1_max':1, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0}])
    def test_sanity2(self):
        expected = {'a1_max': 'ERROR', 'a1_min': 'CORRECT', 'a1_values': 'NOT NORMALIZED', 'a2_max': 'CORRECT', 'a2_min': 'CORRECT', 'a2_values': 'INCONSISTENT'}
        result = sanity_check(self.test_df2, 2)
        self.assertDictEqual(result, expected)


class testProductCostRow(unittest.TestCase):
    """
    """
    row = pd.DataFrame([{"cu_non_material_cost_per_unit":100,"al_non_material_cost_per_unit":50,
                             "cu_copper_kg": 5,  "al_copper_kg": 2,
                              "cu_aluminum_kg":1, "al_aluminum_kg":4,
                                "copper_price_per_kg": 5, "aluminum_price_per_kg":2}]).iloc[0]
    def test_cost_row1(self):
        result = calc_product_cost_row(self.row, cu_material_cost = None, al_material_cost = None, variable = "cu")
        expected = 127
        self.assertEqual(result,expected)
    def test_cost_row2(self):
        result = calc_product_cost_row(self.row, cu_material_cost = None, al_material_cost = None, variable = "al")
        expected = 68
        self.assertEqual(result,expected)
    def test_cost_row3(self):
        result = calc_product_cost_row(self.row, cu_material_cost = 10, al_material_cost = None, variable = "cu")
        expected = 152
        self.assertEqual(result,expected)
    def test_cost_row4(self):
        result = calc_product_cost_row(self.row, cu_material_cost = 10, al_material_cost = 0, variable = "cu")
        expected = 150
        self.assertEqual(result,expected)
    def test_cost_row5(self):
        result = calc_product_cost_row(self.row, cu_material_cost = 10, al_material_cost = None, variable = "al")
        expected = 78
        self.assertEqual(result,expected)
    def test_cost_row6(self):
        result = calc_product_cost_row(self.row, cu_material_cost = 10, al_material_cost = 0, variable = "al")
        expected = 70
        self.assertEqual(result,expected)


class testNormalizeProductCostRow(unittest.TestCase):
    """
    """
    row = pd.DataFrame([{"attribute_1_max":100,"attribute_1_min":10}]).iloc[0]
    def test_normalize_product_row1(self):
        result = normalize_product_cost_row(self.row, 100)
        result = round(result,2)
        expected = round(0,2)
        self.assertEqual(result, expected)
    def test_normalize_product_row2(self):
        result = normalize_product_cost_row(self.row, 10)
        result = round(result,2)
        expected = round(1,2)
        self.assertEqual(result, expected)
    def test_normalize_product_row3(self):
        result = normalize_product_cost_row(self.row, 50)
        result = round(result,2)
        expected = round(5/9,2)
        self.assertEqual(result, expected)

class testUtilityRow(unittest.TestCase):
    """
    """
    row =  pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':2,'al_attribute_2_value':2,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':10, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.4,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0,
       "cu_non_material_cost_per_unit":1, "cu_copper_kg":2, "cu_aluminum_kg": 2,
       "al_non_material_cost_per_unit":1, "al_copper_kg":1, "al_aluminum_kg": 1,
       "copper_price_per_kg": 1, "aluminum_price_per_kg":1}]).iloc[0]
    def test_utility_row1(self):
        result = calc_utility_row(self.row, num_attributes = 2)
        result = round(result,2)
        expected = round(0.35,2)
        self.assertEqual(result, expected)
    def test_utility_row2(self):
        result = calc_utility_row(self.row, variable = "al", num_attributes = 2)
        result = round(result,2)
        expected = round(0.82,2)
        self.assertEqual(result, expected)

class testPointGenPrice(unittest.TestCase):
    """
    """
    df =  pd.DataFrame([{"region": "India","weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':2,'al_attribute_2_value':2,'attribute_1_min':0,
       'attribute_1_max':20, 'attribute_2_min':10, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.4,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0,
       "cu_non_material_cost_per_unit":1, "cu_copper_kg":2, "cu_aluminum_kg": 2,
       "al_non_material_cost_per_unit":1, "al_copper_kg":1, "al_aluminum_kg": 1,
       "copper_price_per_kg": 1, "aluminum_price_per_kg":1, "tau_value":1}])
    def test_point_gen1(self):
        result = point_generation_price(self.df, region = "India", price_range = [3,5], num_attributes = 2)
        result = np.round(result,2)
        result = result.tolist()
        expected = [0.4,0.39]
        self.assertListEqual(result, expected)

class testPointGenRatio(unittest.TestCase):
    """
    """
    df =  pd.DataFrame([{"region": "India","weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':2,'al_attribute_2_value':2,'attribute_1_min':0,
       'attribute_1_max':20, 'attribute_2_min':10, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.4,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0,
       "cu_non_material_cost_per_unit":1, "cu_copper_kg":2, "cu_aluminum_kg": 2,
       "al_non_material_cost_per_unit":1, "al_copper_kg":1, "al_aluminum_kg": 1,
       "copper_price_per_kg": 1, "aluminum_price_per_kg":1, "tau_value":1}])
    def test_point_ratio1(self):
        result = point_generation_ratio(self.df, region = "India", ratio_range = [1,2], num_attributes = 2)
        result = np.round(result,2)
        result = result.tolist()
        expected = [0.40,0.39]
        self.assertListEqual(result, expected)

# def point_generation_price(input_df, region, price_range = np.arange(0,5, 0.01), variable="cu", num_attributes = 5):
#     """
#     return a series of points. Assumes only one row with the region value.
#     """
#     ms_points = []
#     region_row = input_df.loc[input_df["region"] == region].iloc[0]
#     tau = region_row["tau_value"]
#     if variable == "cu":
#         for price in price_range:
#             al_utility = calc_utility_row(region_row, cu_material_cost = price, num_attributes=num_attributes, variable = "al")
#             cu_utility = calc_utility_row(region_row,cu_material_cost = price, num_attributes = num_attributes, variable = "cu")
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     elif variable == "al":
#         for price in price_range:
#             al_utility = calc_utility_row(region_row, al_material_cost = price, num_attributes=num_attributes, variable = "al")
#             cu_utility = calc_utility_row(region_row, al_material_cost = price, num_attributes = num_attributes, variable = "cu")
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     else:
#         raise ValueError("Invalid input")
#     return ms_points

# def point_generation_ratio(input_df, region, hold = "al", hold_value = 2,ratio_range =np.arange(0.1,2, 0.01),num_attributes=5):
#     """
#     Change ratios.
#     """
#     ms_points = []
#     region_row = input_df.loc[input_df["region"] == region].iloc[0]
#     tau = region_row["tau_value"]
#     if hold == "al":
#         for ratio in ratio_range:
#             al_material_cost = hold_value
#             cu_material_cost = hold_value*ratio
#             al_utility = calc_utility_row(region_row, cu_material_cost = cu_material_cost, al_material_cost=al_material_cost, num_attributes=num_attributes, variable = "al")
#             cu_utility = calc_utility_row(region_row,cu_material_cost = cu_material_cost, al_material_cost = al_material_cost, num_attributes = num_attributes, variable = "cu")
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     elif hold == "cu":
#         for ratio in ratio_range:
#             al_material_cost = hold_value/ratio
#             cu_material_cost = hold_value
#             al_utility = calc_utility_row(region_row, cu_material_cost = cu_material_cost, al_material_cost=al_material_cost, num_attributes=num_attributes, variable = "al")
#             cu_utility = calc_utility_row(region_row,cu_material_cost = cu_material_cost, al_material_cost = al_material_cost, num_attributes = num_attributes, variable = "cu")
#             ms = ms_logit(cu_utility, al_utility, tau)
#             ms_points.append(ms)
#     else:
#         raise ValueError("Invalid input")
#     return ms_points

# write tests for the following: point_generation_price, point_generation_ratio

######################

if __name__ == '__main__':
    unittest.main()

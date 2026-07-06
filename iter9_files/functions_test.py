import unittest
import numpy as np
import pandas as pd
from functions import tau_callibrate, tau_callibrate_row, tau_callibrate_df, ms_logit, unzero
from functions import calc_product_cost, get_true_mins_maxes, normalize_attributes, calc_utilities
from functions import calc_utility_row, normalize_price_row, point_generation_price

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

class testPriceNormalize(unittest.TestCase):
    row = pd.DataFrame([{"weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0}]).iloc[0]
    def test_price_normalize(self):
        result = normalize_price_row(self.row, price = 2)
        expected = 0.6
        result = round(result,1)
        self.assertEqual(result, expected)



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
    def test_utility_row1(self):
        row = self.test_df.iloc[0]
        result_cu = calc_utility_row(row, num_attributes = self.num_test_atts)
        result_al = calc_utility_row(row, variable = "al",num_attributes = self.num_test_atts)
        result_cu = round(result_cu,2)
        result_al = round(result_al, 2)
        self.assertEqual(0.65, result_cu)
        self.assertEqual(0.70, result_al)
    def test_utility_row2(self):
        row = self.test_df.iloc[0]
        result_cu = calc_utility_row(row, price= 1, num_attributes = self.num_test_atts)
        result_cu = round(result_cu,2)
        self.assertEqual(0.59, result_cu)


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
class testPointGeneration(unittest.TestCase):
    """
    """
    test_df = pd.DataFrame([{"region": "India","weight_attribute_1": 0.3, "weight_attribute_2":0.7,'direction_attribute_1': "positive", 'direction_attribute_2': "negative",
                                 'cu_attribute_1_value':5,'cu_attribute_2_value':5,
       'al_attribute_1_value':0,'al_attribute_2_value':0,'attribute_1_min':0,
       'attribute_1_max':5, 'attribute_2_min':0, 'attribute_2_max':10,
       "cu_a1_callibrated":1.0, "al_a1_callibrated":0.0,"cu_a2_callibrated":0.5, "al_a2_callibrated":1.0, "al_utility":0.2, "tau_value":0.2}])
    def test_point_generation(self):
        result = point_generation_price(self.test_df, "India", price_range = [1,2], variable = "cu", num_attributes = 2)
        expected = [0.88, 0.84]
        self.assertEqual(round(result[0],2), expected[0])
        self.assertEqual(round(result[1],2), expected[1])

if __name__ == '__main__':
    unittest.main()

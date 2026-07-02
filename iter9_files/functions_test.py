import unittest
import numpy as np
import pandas as pd
from functions import tau_callibrate, ms_logit, unzero
from functions import calc_product_cost, get_true_mins_maxes, normalize_attributes, calc_utilities

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

class testNormalizeAttributes(unittest.TestCase):
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
    def test_normalize(self):
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
    #############TEST AS ROW
    #############TEST AS DF


if __name__ == '__main__':
    unittest.main()

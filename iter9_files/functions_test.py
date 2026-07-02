import unittest
import numpy as np
import pandas as pd
from functions import tau_callibrate, ms_logit, unzero
from functions import calc_product_cost, get_true_mins_maxes

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

############


class testProductCalc(unittest.TestCase):
    """
    """
    test_df1 = pd.DataFrame([{"cu_nonmaterial": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
             "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3}])
    def test_prod_calc1(self):
        new_result = self.test_df1.copy()
        new_result["pc"] = calc_product_cost(self.test_df1)
        pc = new_result["pc"][0]
        expected_pc = 541
        self.assertEqual(pc, expected_pc)
    test_df2 = pd.DataFrame([{"cu_nonmaterial": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
             "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3},{"cu_nonmaterial": 300, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
             "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3}])
    def test_prod_calc2(self):
        new_result = self.test_df2.copy()
        new_result["pc"] = self.test_df2.apply(calc_product_cost,axis = 1)
        pc1 = new_result["pc"][0]
        pc2 = new_result["pc"][1]
        expected_pc1 = 541
        expected_pc2 = 341
        self.assertEqual(pc1, expected_pc1)
        self.assertEqual(pc2, expected_pc2)

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

if __name__ == '__main__':
    unittest.main()

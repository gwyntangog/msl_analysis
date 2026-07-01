import unittest
import numpy as np
import pandas as pd
from functions import tau_callibrate, ms_logit, unzero
from functions import calc_product_cost

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


# class testProductCalc(unittest.TestCase):
#     """
#     """
#     test_df = pd.DataFrame([{"cu_nonmaterial": 500, "cu_copper_kg": 5, "cu_aluminum_kg": 2, "cu_material_cost": 7,
#              "al_nonmaterial": 550, "al_copper_kg": 2, "al_aluminum_kg":3, "al_material_cost": 3}])
#     def test_prod_calc1(self):
#         pc = calc_product_cost(self.test_df)
#         print(pc)
#         expected_pc = 541
#         self.assertEqual(pc, expected_pc)
    # def test_logit2(self):
    #     utility_cu = 0.5
    #     utility_al = 0.5
    #     tau = 1
    #     ms = ms_logit(utility_cu,utility_al,tau)
    #     expected_ms = 0.5
    #     self.assertEqual(ms, expected_ms)
    # def test_logit3(self):
    #     utility_cu = 0.2
    #     utility_al = 0.8
    #     tau = 2
    #     ms = ms_logit(utility_cu,utility_al,tau)
    #     ms = round(ms,6)
    #     expected_ms = 0.425557
    #     self.assertEqual(ms, expected_ms)


# def tau_callibrate():
#     return
if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import os

# PHASE 1: Normal finding tau
# PHASE 2: Tau callibrate step

def unzero(n):
    """
    Compresses the number to be in the range from "almost 0" to "almost 1".
    """
    return np.clip(n, 10**(-9), 1-10**(-9))

def tau_callibrate(utility_cu, utility_al, market_share_cu):
    """
    Callibrate the value of tau such that the logit function returns the same value
    as the current MARKET_SHARE_CU. See derivation in Overleaf file.
    Calculates tau based on copper utility, al utility, and market share.
    """
    mc = np.clip((market_share_cu, 10**(-9)), 1-10**(-9))
    tau = (utility_cu-utility_al)/(np.log(mc)-np.log(1-mc))
    return tau

# TODO: Tau callibrate steo=p

def ms_logit(utility_cu, utility_al, tau):
    """
    Calculate the market share via the logit function.
    Note: this is raw, not softmax. see iter8 for softmax code.
    """
    exp_cu = np.exp(utility_cu)
    exp_al = np.exp(utility_al)
    total = exp_cu + exp_al
    total = np.clip(total, 10**(-9),1-10**(-9)) # Make a function that removes 0s
    result = exp_cu/total
    result = np.clip(result, 10**(-9),1-10**(-9))
    return result

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp

def paired_wilcoxon(y1, y2):
    """
    Wilcoxon signed-rank test for paired samples.
    Returns statistic, p-value.
    """
    stat, p = wilcoxon(y1, y2)
    return stat, p

def friedman_test(*args):
    """
    Friedman test for multiple paired samples.
    Input: args = lists of scores for each model
    """
    stat, p = friedmanchisquare(*args)
    return stat, p

def posthoc_nemenyi(data):
    """
    Post-hoc Nemenyi test for Friedman analysis.
    data: 2D array of shape (num_datasets, num_models)
    """
    p_values = sp.posthoc_nemenyi_friedman(data)
    return p_values

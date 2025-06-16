# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

def chi_square_normality_test(series, bins=10):
    series = series.dropna()
    n = len(series)
    hist, bin_edges = np.histogram(series, bins=bins)
    bin_probs = norm.cdf(bin_edges[1:], loc=series.mean(), scale=series.std()) - \
                norm.cdf(bin_edges[:-1], loc=series.mean(), scale=series.std())
    expected = n * bin_probs
    chi_stat = np.sum((hist - expected) ** 2 / expected)
    df = bins - 1 - 2
    return 1 - chi2.cdf(chi_stat, df)

def compute_normality_index_series_from_list(data_list, alpha=0.01):
    idx_vals = []
    idx_dates = []
    for df in data_list:
        passed = sum(chi_square_normality_test(df[col]) >= alpha for col in df.columns)
        idx_vals.append(passed / len(df.columns))
        idx_dates.append(df.index[-1])
    return pd.Series(idx_vals, index=idx_dates)

# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli
"""

import numpy as np
import pandas as pd

def compute_metrics(resid_df, true_df):
    metrics = {}
    for stock in resid_df.columns:
        residual = resid_df[stock]
        true = true_df[stock]
        mask = np.abs(true) > 1e-6
        mape = np.mean(np.abs(residual[mask] / true[mask])) if mask.any() else np.nan
        mse = np.mean(residual**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residual))
        metrics[stock] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    return pd.DataFrame(metrics).T



# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli
"""

import os
import time
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

from config import n_epochs, block_size, val_epochs, patience, BYTES_PER_WORKER
from utils import estimate_max_jobs, format_time
from model import configure_tf_gpu_growth, build_ar_mrnn_model
from data_loader import download_ibex_components, download_prices
from portfolio import efficient_frontier_with_sharpe
from normality import compute_normality_index_series_from_list
from evaluation import compute_metrics

from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt

def fit_model_for_ticker(tk, series, p, n_epochs, k=1):
    tf.keras.backend.clear_session()
    X, y, Z = [], [], []
    for t in range(p, len(series) - 1):
        z = series.iloc[t - (p - 1) - k]
        x_t = series.iloc[t - p : t].values - z
        y_t = series.iloc[t] - z
        X.append(x_t)
        y.append(y_t)
        Z.append(z)

    if len(X) < p + 7:
        return tk, np.nan, np.full(len(y), np.nan)

    X, y, Z = np.array(X), np.array(y), np.array(Z)
    X_tr, y_tr, Z_tr = X[:-7], y[:-7], Z[:-7]
    X_val, y_val, Z_val = X[-7:], y[-7:], Z[-7:]

    model = build_ar_mrnn_model()
    best_rmse = np.inf
    best_w = model.get_weights()
    epochs_done = 0
    no_improve_count = 0

    while epochs_done < n_epochs:
        ne = min(block_size, n_epochs - epochs_done)
        model.fit(X_tr, y_tr, epochs=ne, batch_size=len(X_tr), verbose=0)
        epochs_done += ne
        model.fit(X_val, y_val, epochs=val_epochs, batch_size=len(X_val), verbose=0)
        epochs_done += val_epochs
        yv_pred = model.predict(X_val, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_val, yv_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_w = model.get_weights()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping en serie {tk} tras {epochs_done} epochs")
            break

    model.set_weights(best_w)

    z_last = series.iloc[-(p - 1) - k]
    x_last = series.iloc[-p:].values - z_last
    next_r = float(model.predict(x_last.reshape(1, -1), verbose=0)) + z_last
    y_all_pred_trans = model.predict(X, verbose=0).flatten()
    y_all_pred = y_all_pred_trans + Z
    resid = series.iloc[p:-1].values - y_all_pred

    del y_all_pred_trans
    gc.collect()
    return tk, next_r, resid


if __name__ == "__main__":
    start_time = time.time()
    configure_tf_gpu_growth()
    N_CPU = max(cpu_count() - 1, 1)
    N_FROM_RAM = estimate_max_jobs(BYTES_PER_WORKER)
    N_JOBS = min(N_CPU, N_FROM_RAM)
    print(f"→ Using N_JOBS = {N_JOBS} (CPU cap: {N_CPU}, RAM cap: {N_FROM_RAM})")
    start_date = "2022-07-20"
    end_date = "2025-01-01"
    validation_date = "2022-08-03"
    tickers = download_ibex_components()
    stock_df = download_prices(tickers, start_date, end_date)
    stock_df.columns = stock_df.columns.get_level_values(0)
    index_df = download_prices(["^IBEX"], start_date, end_date)

    good_tk = [t for t in stock_df.columns if stock_df[t].first_valid_index() < pd.Timestamp(validation_date)]
    stock_df = stock_df[good_tk]

    wed_stock = stock_df[stock_df.index.weekday == 2]
    wed_index = index_df[index_df.index.weekday == 2]
    wed_returns = wed_stock.pct_change().dropna()
    index_ret = wed_index.pct_change().dropna()
    common_tk = wed_returns.columns[wed_returns.notna().all()].tolist()
    wed_returns = wed_returns[common_tk]

    p = 4
    window = 26
    start_i = window + p
    wealth_mrnn, wealth_mkv = [100.0], [100.0]
    dates_port, residuals_all, returns_all = [], [], []
    weights_records, performance_records, efficient_frontiers, sharpe_points = [], [], {}, {}

    for i in range(start_i, len(wed_returns) - 1):
        print("Rebalancing week", i)
        train = wed_returns.iloc[i - window - p + 1 : i]
        test = wed_returns.iloc[i : i + 1]
        if train.empty or test.empty:
            continue

        results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
            delayed(fit_model_for_ticker)(tk, train[tk].dropna(), p, n_epochs) for tk in train.columns
        )

        pred_r, resid_mat, valid_tks = [], [], []
        for tk, next_r, resid in results:
            if np.isnan(next_r): continue
            pred_r.append(next_r)
            resid_mat.append(resid)
            valid_tks.append(tk)

        pred_r = pd.Series(pred_r, index=valid_tks).dropna()
        val_col = pred_r.index
        resid_mat = np.array(resid_mat)[[train.columns.get_loc(c) for c in val_col]].T
        df_resid = pd.DataFrame(resid_mat, index=train.index[-resid_mat.shape[0]:], columns=val_col)
        df_rets = train[val_col].loc[df_resid.index]

        residuals_all.append(df_resid)
        returns_all.append(df_rets)
        Sigma_mrnn = pd.DataFrame(np.cov(resid_mat, rowvar=False), index=val_col, columns=val_col)
        mu_mkv, Sigma_mkv = train.mean(), train.cov()
        del resid_mat, df_resid, df_rets
        gc.collect()

        f_mrnn = efficient_frontier_with_sharpe(pred_r, Sigma_mrnn)
        f_mkv = efficient_frontier_with_sharpe(mu_mkv, Sigma_mkv)
        idx_mrnn, idx_mkv = f_mrnn['best_idx'], f_mkv['best_idx']
        if idx_mrnn is None or idx_mkv is None:
            wealth_mrnn.append(wealth_mrnn[-1])
            wealth_mkv.append(wealth_mkv[-1])
            dates_port.append(test.index[0])
            continue

        w_mrnn, w_mkv = f_mrnn['weights'][idx_mrnn], f_mkv['weights'][idx_mkv]
        for tk, w in zip(val_col, w_mrnn):
            weights_records.append({"Date": test.index[0], "Model": "AR-MRNN", "Ticker": tk, "Weight": w})
        for tk, w in zip(train.columns, w_mkv):
            weights_records.append({"Date": test.index[0], "Model": "Markowitz", "Ticker": tk, "Weight": w})

        efficient_frontiers[test.index[0]] = {'mrnn': f_mrnn, 'mkv': f_mkv}
        sharpe_points[test.index[0]] = {
            'mrnn': (f_mrnn['sigma'][idx_mrnn], f_mrnn['mu'][idx_mrnn]),
            'mkv':  (f_mkv['sigma'][idx_mkv],  f_mkv['mu'][idx_mkv])
        }

        r_week = test.iloc[0]
        wealth_mrnn.append(wealth_mrnn[-1] * (1 + np.dot(w_mrnn, r_week[val_col])))
        wealth_mkv.append(wealth_mkv[-1] * (1 + np.dot(w_mkv, r_week.values)))
        dates_port.append(test.index[0])

        mu_r_mrnn = np.dot(w_mrnn, test[val_col].mean())
        performance_records.append({
            "Date": test.index[0], "Model": "AR-MRNN", "Mu_pred": f_mrnn['mu'][idx_mrnn],
            "Sharpe_pred": f_mrnn['sharpe'][idx_mrnn],
            "Mu_real": mu_r_mrnn
        })

        mu_r_mkv = np.dot(w_mkv, test.mean())
        performance_records.append({
            "Date": test.index[0], "Model": "Markowitz", "Mu_pred": f_mkv['mu'][idx_mkv],
           "Sharpe_pred": f_mkv['sharpe'][idx_mkv],
            "Mu_real": mu_r_mkv
        })
        print(f"Tiempo transcurrido: {format_time(time.time() - start_time)}")
   
    # Plot only first and last sample for frontiers to keep the figure compact
    sample_keys = []
    if efficient_frontiers:
        sample_keys.append(next(iter(efficient_frontiers)))       # first date
        sample_keys.append(next(reversed(efficient_frontiers)))   # last date

    if sample_keys:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        for ax, date in zip(axs, sample_keys):
            f1 = efficient_frontiers[date]['mrnn']
            f2 = efficient_frontiers[date]['mkv']
            ax.plot(f1['sigma'], f1['mu'], label='AR-MRNN', lw=2)
            ax.plot(f2['sigma'], f2['mu'], label='Markowitz', lw=2, ls='--')
            ax.scatter(*sharpe_points[date]['mrnn'], color='blue', zorder=5)
            ax.scatter(*sharpe_points[date]['mkv'], color='orange', zorder=5)
            ax.set_title(f"Frontier {date.date()}")
            ax.set_xlabel("Volatility (σ)")
            ax.set_ylabel("Expected Returns (μ)")
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show()

    index_ret = index_ret["^IBEX"]["^IBEX"]
    ibex_weekly = index_ret.reindex(dates_port).ffill()
    ibex_wealth = [100.0]
    for r in ibex_weekly:
        ibex_wealth.append(ibex_wealth[-1] * (1 + r))
    ibex_wealth = ibex_wealth[1:]


    plt.figure(figsize=(12, 6))
    plt.plot(dates_port, wealth_mrnn[1:], label="AR-MRNN Sharpe-Opt", lw=2)
    plt.plot(dates_port, wealth_mkv[1:], label="Markowitz Sharpe-Opt", lw=2, ls='--')
    plt.plot(dates_port, ibex_wealth, label="IBEX-35 Index", lw=2, ls='-.')
    plt.xlabel("Date")
    plt.ylabel("Accumulated Wealth (base 100)")
    plt.title("Wealth Accumulation: AR-MRNN vs Markowitz vs IBEX-35")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    norm_idx_resid = compute_normality_index_series_from_list(residuals_all, alpha=0.01)
    norm_idx_rets = compute_normality_index_series_from_list(returns_all, alpha=0.01)

    combined_metrics = pd.concat(
        [compute_metrics(resid, true) for resid, true in zip(residuals_all, returns_all)],
        keys=range(len(residuals_all)), names=["Prediction", "Stock"]
    )
    mean_resid = norm_idx_resid.mean()
    std_resid = norm_idx_resid.std()
    var_resid = norm_idx_resid.var()
    mean_rets = norm_idx_rets.mean()
    std_rets = norm_idx_rets.std()
    var_rets = norm_idx_rets.var()


    print("\n=== Normality Index Summary (AR-MRNN vs Raw Returns) ===")
    print(f"AR-MRNN Residuals:  Mean={mean_resid:.4f}  Var={var_resid:.5f}  Std={std_resid:.4f}")
    print(f"Raw Returns:        Mean={mean_rets:.4f}  Var={var_rets:.5f}  Std={std_rets:.4f}")

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs[0].bar(norm_idx_resid.index, norm_idx_resid, color='tomato', width=5)
    axs[0].set_ylim(0.75, 1.0)
    axs[0].set_title("(a) Normality Index – AR-MRNN Residuals", loc='left')
    axs[0].set_ylabel("Normality Index")

    axs[1].bar(norm_idx_rets.index, norm_idx_rets, color='royalblue', width=5)
    axs[1].set_ylim(0.75, 1.0)
    axs[1].set_title("(b) Normality Index – Raw Returns", loc='left')
    axs[1].set_ylabel("Normality Index")
    axs[1].set_xlabel("Fecha")

    plt.tight_layout()
    plt.show()
    mean_metrics_per_stock = combined_metrics.groupby("Stock").mean()
    print(mean_metrics_per_stock)
    print('Mean: ', mean_metrics_per_stock.mean())
    print('Var: ', mean_metrics_per_stock.var())
    print('Std: ', mean_metrics_per_stock.std())

    df_w = pd.DataFrame(weights_records)
    df_p = pd.DataFrame(performance_records)
    with pd.ExcelWriter("portfolio_results_summary.xlsx", engine="xlsxwriter") as writer:
        df_w.to_excel(writer, sheet_name="Portfolio Weights", index=False)
        df_p.to_excel(writer, sheet_name="Performance", index=False)

    print(f"\nTiempo total: {format_time(time.time() - start_time)}, Épocas por semana: {n_epochs}")
# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli
"""

import numpy as np
import cvxpy as cp

def efficient_frontier_with_sharpe(mu, Sigma, n_points=200):
    N = len(mu)
    w = cp.Variable(N)
    target_returns = np.linspace(0, 0.25, n_points)

    mu_list, sigma_list, sharpe_list, weights_list = [], [], [], []

    for r_t in target_returns:
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma.values)),
            [cp.sum(w) == 1, mu.values @ w == r_t, w >= 0]
        )
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError:
            continue

        if w.value is None:
            continue

        w_opt = w.value
        sigma = np.sqrt(w_opt @ Sigma.values @ w_opt)
        sharpe = r_t / sigma

        mu_list.append(r_t)
        sigma_list.append(sigma)
        sharpe_list.append(sharpe)
        weights_list.append(w_opt)

    if not sharpe_list:
        return {'mu': np.array([]), 'sigma': np.array([]), 'sharpe': np.array([]),
                'weights': [], 'best_idx': None}

    best_idx = int(np.argmax(sharpe_list))
    return {
        'mu': np.array(mu_list),
        'sigma': np.array(sigma_list),
        'sharpe': np.array(sharpe_list),
        'weights': weights_list,
        'best_idx': best_idx
    }

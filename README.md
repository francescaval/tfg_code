# AR-MRNN Portfolio Optimization (based on paper by Freitas, De Souza and de Almeida)

This repository implements a weekly backtesting system for IBEX-35 stock portfolios using feedforward autoregressive neural networks (AR-MRNN). The strategy is benchmarked against a classical Markowitz (mean-variance) portfolio, with weekly rebalancing based on Sharpe-optimal weights.


## Key Features

- One AR-MRNN model per asset (architecture: `4:16:4:1` with sigmoid activations).
- Weekly backtesting over a specified date range (e.g., 2000–2002). 
- Sharpe-optimal portfolio selection via efficient frontier.
- Comparison with classical Markowitz portfolio.
- Wealth evolution plots vs the IBEX-35 benchmark.
- Normality tests on model residuals vs raw returns (Normality Index).
- Forecast error metrics: MSE, RMSE, MAE and MAPE
- Choose the start date, the end date and the validation date to select the tickers. Also modify the value for the window or the number of epochs-
  

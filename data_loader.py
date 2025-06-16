# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli
"""

import yfinance as yf
import pandas as pd
import time

def download_ibex_components(validation_date):
    
    list1 = ['ACX.MC', 'ANA.MC', 'BBVA.MC', 'BKT.MC', 'ELE.MC', 'FER.MC', 'IBE.MC', 'NTGY.MC', 'RED.MC', 'REP.MC', 'SAB.MC', 'SAN.MC', 'SCYR.MC', 'TEF.MC']
    list2 = ['ACS.MC', 'ACX.MC', 'AMS.MC', 'ANA.MC', 'ANE.MC', 'BBVA.MC', 'BKT.MC', 'CABK.MC', 'CLNX.MC', 'COL.MC', 'AENA.MC', 'ELE.MC', 'ENG.MC', 'FDR.MC', 'FER.MC', 'GRF.MC', 'IAG.MC', 'IBE.MC', 'IDR.MC', 'ITX.MC', 'LOG.MC', 'MAP.MC', 'MRL.MC', 'MTS.MC', 'NTGY.MC', 'RED.MC', 'REP.MC', 'ROVI.MC', 'SAB.MC', 'SAN.MC', 'SCYR.MC', 'SLR.MC', 'TEF.MC', 'UNI.MC']

    if "2000" in validation_date:
        return list1
    elif "2022" in validation_date:
        return list2
    else:
        raise ValueError(f"No defined ticker list for date: {validation_date}")


def download_prices(tickers, start="2000-01-01", end="2002-07-01"):
    close_px = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end)["Close"]
            time.sleep(3)
            if not df.dropna().empty:
                close_px[t] = df
        except Exception:
            continue
    return pd.concat(close_px.values(), axis=1, keys=close_px.keys())


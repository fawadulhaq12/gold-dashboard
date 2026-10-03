"""Models behind the Forecast tab.

Both models are trained ONLY on data before TRAIN_END, so the locked 2025-2026 period stays untouched
for the project's one-time final test. Features and settings match the research scripts:
  * direction  : logistic regression on price + macro features (02_baseline_walkforward.py)
  * volatility : ridge regression on the HAR features, near-zero-range days removed (06_volatility.py)
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gold_pipeline as gp

TRAIN_END = "2025-01-01"
PRICE = ["ret_0", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_5d", "ret_10d", "ret_20d",
         "vol_10", "vol_20", "sma10_ratio", "sma50_ratio", "rsi_14", "macd_hist", "hl_range"]
MACRO = ["vix", "real_yield_10y", "dxy_ret", "oil_ret", "spx_ret", "vix_chg", "real_yield_chg"]
DIR_FEATS = PRICE + MACRO
HAR = ["lr_0", "lr_5", "lr_22"]
VOL_RMSE = 0.537          # walk-forward RMSE of this model on 2019-2024 (log scale), used for the interval


def build_frame(market, fred):
    df = gp.build_dataset(market.drop(columns=["gold_open"], errors="ignore"), fred)
    if "gold_open" in market:
        df["gold_open"] = market["gold_open"].reindex(df.index)
    rng = np.log(df["gold_high"] / df["gold_low"])
    lr = np.log(rng.mask(rng < 1e-4))                        # stale days (high == low) become missing
    df["lr_0"] = lr
    df["lr_5"] = lr.rolling(5, min_periods=3).mean()
    df["lr_22"] = lr.rolling(22, min_periods=15).mean()
    df["target_lr"] = df["lr_0"].shift(-1)
    return df


def train(df):
    tr = df[df.index < TRAIN_END]
    d = tr.dropna(subset=["target_up"] + DIR_FEATS).iloc[:-1]                 # 1-day embargo
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    clf.fit(d[DIR_FEATS], d["target_up"])
    v = tr.dropna(subset=["target_lr"] + HAR).iloc[:-1]
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    reg.fit(v[HAR], v["target_lr"])
    return clf, reg, {"rows": len(d), "train_end": d.index.max().date()}


def latest_complete_row(df, now_utc=None):
    """Last finished trading day (drops today's bar while the futures session is still open)."""
    now_utc = now_utc or datetime.now(ZoneInfo("UTC"))
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    partial = df.index[-1].date() == ny.date() and ny.hour < 17
    row = df.iloc[-2] if partial else df.iloc[-1]
    return row, partial


def forecast(clf, reg, row, df):
    x = row[DIR_FEATS].astype(float).to_frame().T
    p_up = float(clf.predict_proba(x)[0, 1])
    contrib = pd.Series(clf[-1].coef_[0] * clf[0].transform(x)[0], index=DIR_FEATS).sort_values()
    pred_lr = float(reg.predict(row[HAR].astype(float).to_frame().T)[0])
    z = 1.2816                                                # 80% interval on the log scale
    return {
        "p_up": p_up, "contrib": contrib,
        "range_med": float(np.exp(pred_lr)),
        "range_lo": float(np.exp(pred_lr - z * VOL_RMSE)),
        "range_hi": float(np.exp(pred_lr + z * VOL_RMSE)),
        "range_recent": float(np.exp(df["lr_0"].dropna().tail(22).mean())),
    }

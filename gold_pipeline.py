"""
STAGE 2 - Data pipeline for the gold next-day prediction project.

What it does
  1. Downloads daily gold futures + macro series (yfinance, FRED).
  2. Builds features that use ONLY information available at the close of day t.
  3. Builds targets about day t+1 (next-day direction and next-day log return).
  4. Saves everything to gold_dataset.csv.

Run in Google Colab:
    !pip install -q yfinance pandas numpy
    %run 01_data_pipeline.py
"""
import numpy as np
import pandas as pd

START = "2015-01-01"  # GDELT 2.0 (news data, Stage 3) starts in 2015

TICKERS = {
    "gold": "GC=F",       # COMEX gold futures (continuous front month)
    "dxy": "DX-Y.NYB",    # US dollar index
    "vix": "^VIX",        # equity volatility ("fear") index
    "oil": "BZ=F",        # Brent crude futures (WTI went NEGATIVE in Apr 2020, which breaks log returns)
    "spx": "^GSPC",       # S&P 500
}
FRED_SERIES = {"real_yield_10y": "DFII10"}  # 10-year TIPS real yield


# --------------------------------------------------------------------------
# 1. DOWNLOAD
# --------------------------------------------------------------------------
def download_market(start=START, end=None) -> pd.DataFrame:
    import yfinance as yf  # imported here so build_dataset can be tested offline

    raw = yf.download(list(TICKERS.values()), start=start, end=end,
                      auto_adjust=True, progress=False)
    inv = {v: k for k, v in TICKERS.items()}
    close = raw["Close"].rename(columns=inv)

    out = pd.DataFrame(index=close.index)
    out["gold_close"] = close["gold"]
    out["gold_open"] = raw["Open"]["GC=F"]      # dashboard only (candlestick chart)
    out["gold_high"] = raw["High"]["GC=F"]
    out["gold_low"] = raw["Low"]["GC=F"]
    out["gold_volume"] = raw["Volume"]["GC=F"]
    for k in ["dxy", "vix", "oil", "spx"]:
        out[k] = close[k]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def download_fred(start=START) -> pd.DataFrame:
    cols = {}
    for name, sid in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
        s = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        cols[name] = s.iloc[:, 0]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# 2. FEATURES + TARGETS
# --------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_dataset(market: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    df = market.join(fred, how="left")
    df = df.dropna(subset=["gold_close"])
    # Macro series have holidays on different days: forward-fill ONLY
    # (never backward-fill, that would copy the future into the past).
    macro_cols = ["dxy", "vix", "oil", "spx"] + list(fred.columns)
    df[macro_cols] = df[macro_cols].ffill()

    # log returns are only defined for positive prices: fail loudly if that breaks
    for k in ["gold_close", "dxy", "oil", "spx"]:
        assert (df[k].dropna() > 0).all(), f"{k} has non-positive prices"

    c = df["gold_close"]
    ret = np.log(c).diff()                       # today's log return (known at close t)

    # --- price / technical features ---------------------------------------
    df["ret_0"] = ret
    for k in range(1, 5):
        df[f"ret_lag{k}"] = ret.shift(k)
    for n in (5, 10, 20):
        df[f"ret_{n}d"] = np.log(c / c.shift(n))
    df["vol_10"] = ret.rolling(10).std()
    df["vol_20"] = ret.rolling(20).std()
    df["sma10_ratio"] = c / c.rolling(10).mean() - 1
    df["sma50_ratio"] = c / c.rolling(50).mean() - 1
    df["rsi_14"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    df["hl_range"] = (df["gold_high"] - df["gold_low"]) / c

    # --- macro features (changes, not levels, so they are stationary) -------
    for k in ["dxy", "oil", "spx"]:
        df[f"{k}_ret"] = np.log(df[k]).diff()
    df["vix_chg"] = df["vix"].diff()
    df["real_yield_chg"] = df["real_yield_10y"].diff()

    # --- targets: about day t+1 --------------------------------------------
    df["target_ret"] = ret.shift(-1)
    df["target_up"] = (df["target_ret"] > 0).astype(float)
    df.loc[df["target_ret"].isna(), "target_up"] = np.nan  # last row: unknown yet

    # drop warm-up rows where rolling windows are not full yet
    feat_cols = [x for x in df.columns if x not in ("target_ret", "target_up")]
    df = df.dropna(subset=feat_cols)
    return df


def main(out_path="gold_dataset.csv"):
    market = download_market()
    fred = download_fred()
    df = build_dataset(market, fred)
    df.to_csv(out_path)
    print(f"Saved {out_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Share of 'up' days: {df['target_up'].mean():.3f}")
    return df


if __name__ == "__main__":
    main()

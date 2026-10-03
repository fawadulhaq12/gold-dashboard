"""Network access for the dashboard. Every call is cached; failures are returned to the page, never fatal."""
import time

import pandas as pd
import requests
import streamlit as st

import gold_pipeline as gp

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
QUOTES = {"Gold futures": "GC=F", "US dollar index": "DX-Y.NYB", "VIX": "^VIX",
          "Brent crude": "BZ=F", "S&P 500": "^GSPC"}
HOTSPOT_QUERIES = {
    "Russia-Ukraine": "(Russia OR Ukraine) (war OR attack OR missile OR strike) sourcelang:english",
    "Middle East": "(Israel OR Iran OR Gaza) (war OR attack OR missile OR strike) sourcelang:english",
    "India-Pakistan": "India Pakistan (military OR attack OR border OR strike) sourcelang:english",
    "US-Iran": "(\"United States\" OR Washington) Iran (sanctions OR strike OR military OR talks) sourcelang:english",
}
_last_call = [0.0]


def safe(fn, *args):
    """Run fn and return (result, None) or (None, readable error)."""
    try:
        return fn(*args), None
    except Exception as exc:                                    # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_quotes():
    import yfinance as yf
    out = {}
    for name, ticker in QUOTES.items():
        try:
            t = yf.Ticker(ticker)
            try:
                fi = t.fast_info
                last, prev = float(fi.last_price), float(fi.previous_close)
            except Exception:                                    # noqa: BLE001
                h = t.history(period="5d")["Close"].dropna()
                last, prev = float(h.iloc[-1]), float(h.iloc[-2])
            out[name] = (last, prev)
        except Exception:                                        # noqa: BLE001
            continue
    if not out:
        raise RuntimeError("Yahoo Finance returned no quotes")
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday():
    import yfinance as yf
    h = yf.Ticker("GC=F").history(period="2d", interval="5m")["Close"].dropna()
    h.index = pd.DatetimeIndex(h.index).tz_convert("UTC").tz_localize(None)
    return h


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history():
    return gp.download_market(), gp.download_fred()


def _gdelt(params):
    wait = 5.3 - (time.time() - _last_call[0])                  # GDELT allows 1 request per 5 seconds
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    r = requests.get(GDELT, params=params, timeout=25,
                     headers={"User-Agent": "gold-research-dashboard/1.0"})
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_coverage(query, timespan="30d"):
    js = _gdelt({"query": query, "mode": "timelinevol", "format": "json", "timespan": timespan})
    pts = js["timeline"][0]["data"]
    s = pd.Series({pd.to_datetime(p["date"], format="%Y%m%dT%H%M%SZ"): float(p["value"]) for p in pts})
    return s.resample("D").mean()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_headlines(query, n=8):
    js = _gdelt({"query": query, "mode": "artlist", "maxrecords": n, "format": "json",
                 "sort": "datedesc", "timespan": "1d"})
    return [{"title": a.get("title", ""), "domain": a.get("domain", ""), "url": a.get("url", ""),
             "seen": a.get("seendate", "")} for a in js.get("articles", [])]

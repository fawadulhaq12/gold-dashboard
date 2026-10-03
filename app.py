from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import live_data as ld
import modeling as md
import results_data as rd

HERE = Path(__file__).parent
INK, BRASS, UP, DOWN, SLATE = "#14202B", "#A8741A", "#1F7A6D", "#B5483A", "#6B7A89"
HOT_Z = {"Russia-Ukraine": "ukr_share_z", "Middle East": "mideast_share_z",
         "India-Pakistan": "indpak_share_z", "US-Iran": "usiran_share_z"}
HOT_COLOR = {"Russia-Ukraine": "#3B5B92", "Middle East": DOWN, "India-Pakistan": UP, "US-Iran": "#7A4E9C"}
LOOKBACK = {"1M": 22, "3M": 66, "6M": 126, "1Y": 252, "3Y": 756, "Max": 10_000}

st.set_page_config(page_title="Gold and geopolitics", page_icon="🟡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1280px; }
h1 { font-size: 2.5rem !important; margin-bottom: 0; }
.sub { color: #4A5A69; font-size: 1.05rem; max-width: 760px; margin: .2rem 0 1rem 0; }
.note { border-left: 3px solid #A8741A; background: #FAF4E6; padding: .6rem 1rem; border-radius: 2px; margin: .6rem 0 1rem 0; }
.finding { border-left: 3px solid #14202B; padding: .1rem 1rem; margin: .7rem 0; }
.finding b { font-family: 'Source Serif 4', serif; font-size: 1.05rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>""", unsafe_allow_html=True)


def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=8, r=8, t=30, b=8),
                      font=dict(family="IBM Plex Sans", color=INK), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E3E8ED")
    return fig


@st.cache_data(ttl=1800, show_spinner="Loading market history from Yahoo Finance and FRED...")
def load_frame():
    market, fred = ld.fetch_history()
    return md.build_frame(market, fred)


@st.cache_resource(ttl=86400, show_spinner="Training the models on 2015-2024 data...")
def load_models():
    return md.train(load_frame())


@st.cache_data
def load_news():
    p = HERE / "data" / "news_features.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None


# ---------------------------------------------------------------- header and live strip
st.markdown("<h1>Gold and geopolitics</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Does world news help predict gold? A research dashboard with live prices, "
            "news pressure from conflict hotspots, a next-day forecast, and the evidence from pre-registered tests.</p>",
            unsafe_allow_html=True)


@st.fragment(run_every=60)
def live_strip():
    q, err = ld.safe(ld.fetch_quotes)
    if err:
        st.warning(f"Live quotes are unavailable right now. {err}")
        return
    cols = st.columns(len(q))
    for col, (name, (last, prev)) in zip(cols, q.items()):
        col.metric(name, f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
    st.caption(f"Updated {datetime.now(timezone.utc):%H:%M:%S} UTC. Prices come from Yahoo Finance, "
               "delayed by about 15 minutes, and refresh every minute.")


live_strip()

with st.sidebar:
    st.header("View")
    span = st.select_slider("Chart range", list(LOOKBACK), value="6M")
    sma20 = st.checkbox("20-day average", True)
    sma50 = st.checkbox("50-day average", True)
    spikes = st.checkbox("Mark news spikes", True)
    if st.button("Refresh all data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("<div class='note'>Research prototype. Nothing here is financial advice or a trading signal.</div>",
                unsafe_allow_html=True)

df, err = ld.safe(load_frame)
if err:
    st.error(f"Market history could not be loaded, so the charts and forecast are unavailable. {err}")
    st.stop()
news = load_news()
v = df.tail(LOOKBACK[span])

tab_m, tab_n, tab_f, tab_r, tab_a = st.tabs(
    ["Market", "News pressure", "Forecast", "Research evidence", "Method and limits"])

# ---------------------------------------------------------------- market
with tab_m:
    fig = go.Figure(go.Candlestick(x=v.index, open=v["gold_open"], high=v["gold_high"], low=v["gold_low"],
                                   close=v["gold_close"], name="Gold futures",
                                   increasing_line_color=UP, decreasing_line_color=DOWN))
    for on, n_, colr in ((sma20, 20, BRASS), (sma50, 50, SLATE)):
        if on:
            fig.add_trace(go.Scatter(x=v.index, y=df["gold_close"].rolling(n_).mean().reindex(v.index),
                                     name=f"{n_}-day average", line=dict(color=colr, width=1.6)))
    if spikes and news is not None:
        for name, col in HOT_Z.items():
            hit = news[col].reindex(v.index)
            hit = hit[hit >= 2.0].index
            fig.add_trace(go.Scatter(x=hit, y=v.loc[hit, "gold_high"] * 1.008, mode="markers", name=f"{name} spike",
                                     marker=dict(symbol="triangle-down", size=10, color=HOT_COLOR[name])))
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(style(fig, 470))
    if spikes and news is None:
        st.caption("Add data/news_features.csv to mark news spikes on this chart.")

    macro = make_subplots(rows=1, cols=4, subplot_titles=["US dollar index", "VIX", "10-year real yield (%)", "Brent crude"])
    for i, c in enumerate(["dxy", "vix", "real_yield_10y", "oil"], 1):
        macro.add_trace(go.Scatter(x=v.index, y=v[c], line=dict(color=SLATE, width=1.5), showlegend=False), row=1, col=i)
    st.plotly_chart(style(macro, 230))

    with st.expander("Gold today, 5-minute prices"):
        intra, e2 = ld.safe(ld.fetch_intraday)
        if e2:
            st.info(f"Intraday prices are unavailable. {e2}")
        else:
            st.plotly_chart(style(go.Figure(go.Scatter(x=intra.index, y=intra.values, line=dict(color=BRASS, width=1.6))), 260))

# ---------------------------------------------------------------- news pressure
with tab_n:
    st.subheader("How intense is conflict coverage right now?")
    st.markdown("<div class='note'>Intensity is measured against each hotspot's own previous 60 days. "
                "A value of 2 means coverage is two standard deviations above its recent normal, which we call a spike.</div>",
                unsafe_allow_html=True)
    if news is None:
        st.info("The history of GDELT conflict pressure is not loaded. Copy news_features.csv from Colab into the "
                "data folder to see it here.")
    else:
        last = news.index.max()
        cols = st.columns(4)
        for col, (name, c) in zip(cols, HOT_Z.items()):
            z = float(news[c].iloc[-1])
            col.metric(name, f"{z:+.1f}", "Spike" if z >= 2 else "Normal range", delta_color="inverse" if z >= 2 else "off")
        st.caption(f"Latest value in the saved GDELT history: {last:%d %b %Y}.")
        nv = news.tail(LOOKBACK[span])
        zf = go.Figure([go.Scatter(x=nv.index, y=nv[c], name=n, line=dict(color=HOT_COLOR[n], width=1.4))
                        for n, c in HOT_Z.items()])
        zf.add_hline(y=2, line_dash="dot", line_color=INK, annotation_text="Spike level")
        st.plotly_chart(style(zf, 340))

    st.subheader("Live news pulse")
    st.caption("Pulled from the GDELT news index. The first load takes about 25 seconds because GDELT limits request speed.")
    if st.button("Load live news pulse") or st.session_state.get("pulse"):
        st.session_state["pulse"] = True
        pf, any_err = go.Figure(), None
        with st.spinner("Reading GDELT..."):
            for name, q in ld.HOTSPOT_QUERIES.items():
                s, e = ld.safe(ld.fetch_coverage, q)
                if e:
                    any_err = e
                    continue
                pf.add_trace(go.Scatter(x=s.index, y=s / s.mean(), name=name, line=dict(color=HOT_COLOR[name], width=1.8)))
        if any_err and not pf.data:
            st.warning(f"GDELT could not be reached. {any_err}")
        else:
            pf.add_hline(y=1, line_dash="dot", line_color=SLATE)
            pf.update_yaxes(title="Coverage vs 30-day average (1.0 = normal)")
            st.plotly_chart(style(pf, 330))
            pick = st.selectbox("Latest headlines for", list(ld.HOTSPOT_QUERIES))
            arts, e = ld.safe(ld.fetch_headlines, ld.HOTSPOT_QUERIES[pick])
            if e:
                st.info(f"Headlines are unavailable. {e}")
            for a in arts or []:
                st.markdown(f"[{a['title']}]({a['url']})  \n<small>{a['domain']}</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------- forecast
with tab_f:
    models, e = ld.safe(load_models)
    if e:
        st.error(f"The models could not be trained. {e}")
    else:
        clf, reg, meta = models
        row, partial = md.latest_complete_row(df)
        f = md.forecast(clf, reg, row, df)
        price = float(row["gold_close"])
        st.markdown(f"<div class='note'>Based on the close of <b>{row.name:%A %d %B %Y}</b>"
                    f"{' (today is still trading, so the latest finished day is used)' if partial else ''}. "
                    f"Models were trained on {meta['rows']:,} days up to {meta['train_end']:%d %b %Y}; "
                    "2025 onward is kept aside for the final test.</div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1.25])
        with left:
            st.subheader("Direction: research output only")
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=f["p_up"] * 100, number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Chance gold closes higher next session"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": INK},
                       "steps": [{"range": [0, 48], "color": "#F0D5D0"}, {"range": [48, 52], "color": "#E3E8ED"},
                                 {"range": [52, 100], "color": "#CFE6E1"}],
                       "threshold": {"line": {"color": BRASS, "width": 4}, "value": rd.ALWAYS_UP_ACC * 100}}))
            st.plotly_chart(style(g, 290))
            lean = "no clear lean" if abs(f["p_up"] - .5) < .02 else ("a slight lean up" if f["p_up"] > .5 else "a slight lean down")
            st.markdown(f"The model shows **{lean}**. The brass line marks the 54.3% of days when gold rose in 2019 to 2024. "
                        "In testing, this model scored 53.2%, below simply guessing up every day, so treat it as a demonstration of the method.")
        with right:
            st.subheader("Volatility: the dependable part")
            lo, md_, hi, rec = (f[k] * 100 for k in ("range_lo", "range_med", "range_hi", "range_recent"))
            rf = go.Figure()
            rf.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(color=BRASS, width=30), name="80% likely range"))
            rf.add_trace(go.Scatter(x=[md_], y=[0], mode="markers", name="Model median",
                                    marker=dict(symbol="line-ns", size=34, line=dict(color=INK, width=3))))
            rf.add_vline(x=rec, line_dash="dot", line_color=SLATE, annotation_text="Last 22 days")
            rf.update_yaxes(visible=False, range=[-1, 1])
            rf.update_xaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(style(rf, 200))
            st.markdown(f"Next session's high-low range is expected near **{md_:.2f}% (about ${price * md_ / 100:,.0f})**, "
                        f"likely between {lo:.2f}% and {hi:.2f}%. Over the last 22 days it averaged {rec:.2f}%. "
                        "In testing this model explained 45% of the variation in daily range, well above chance.")

        st.subheader("What is pushing the direction estimate")
        c = f["contrib"]
        c = pd.concat([c.head(6), c.tail(6)]).drop_duplicates()
        bf = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=[UP if x > 0 else DOWN for x in c.values]))
        bf.update_xaxes(title="Push toward up (right) or down (left), in model units")
        st.plotly_chart(style(bf, 330))

# ---------------------------------------------------------------- research evidence
with tab_r:
    st.markdown("<div class='note'>Every result below comes from walk-forward tests on 2019 to 2024: each year is predicted using "
                "only earlier data. The 2025 to 2026 data is locked for one final test.</div>", unsafe_allow_html=True)
    for head, body in [
        ("Direction.", "No model beat always guessing up (54.3% of days). The best, price plus macro with logistic regression, "
                       "reached 53.6% (p = 0.71). Adding news made it worse."),
        ("Volatility.", "Recent volatility predicts tomorrow's range well (R2 of 0.45). VIX added little, and news features "
                        "made forecasts worse, in the original and in the data-cleaned runs."),
        ("The news measure works.", "GDELT conflict coverage jumped 4 to 7.5 standard deviations on all five famous events we checked."),
        ("Event study.", "Across 41 to 48 spikes per hotspot, gold's reaction afterwards is statistically indistinguishable from "
                         "ordinary days, in direction and in size.")]:
        st.markdown(f"<div class='finding'><b>{head}</b> {body}</div>", unsafe_allow_html=True)

    es = rd.EVENT_STUDY
    ef = go.Figure(go.Bar(x=es["Hotspot"], y=es["Net move days 1-5"], marker_color=BRASS,
                          text=[f"p = {p:.2f}" for p in es["p (net move)"]], textposition="outside"))
    ef.update_yaxes(title="Average net move over days 1-5 (typical-day units)", range=[-1, 1])
    st.subheader("Gold after a news spike")
    st.plotly_chart(style(ef, 330))
    img = HERE / "data" / "event_study.png"
    if img.exists():
        st.image(str(img), caption="Average path of gold after spikes. Grey is the 95% range expected by chance.")
    st.dataframe(es.round(3), hide_index=True)
    st.subheader("Five famous events")
    st.dataframe(rd.FAMOUS, hide_index=True)
    st.subheader("Direction tests")
    st.dataframe(rd.ABLATION, hide_index=True)
    st.subheader("Volatility tests")
    pick = st.radio("Data version", ["Original", "Cleaned (stale days removed)"], horizontal=True)
    st.dataframe(rd.VOL_ORIGINAL if pick == "Original" else rd.VOL_CLEAN, hide_index=True)
    st.caption("R2 vs HAR: positive means better than the recent-volatility benchmark. DM p-value: chance the "
               "improvement is luck. About 84 stale days were found after the first run; both versions are reported.")

# ---------------------------------------------------------------- method
with tab_a:
    st.graphviz_chart("""digraph { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#E8ECF0", color="#B7C2CC", fontname="Helvetica"];
      A [label="Yahoo Finance\\nFRED"]; B [label="GDELT events\\n(BigQuery)"]; C [label="Features\\nprice, macro, news"];
      D [label="Walk-forward\\nvalidation"]; E [label="Models\\nlogistic, HAR-ridge"]; F [label="This dashboard"];
      G [label="GDELT news index\\n(live headlines)"];
      A -> C; B -> C; C -> D -> E -> F; G -> F; }""")
    st.markdown("""
**Data.** Gold futures, dollar index, VIX, Brent and S&P 500 from Yahoo Finance; real yields from FRED; news from the GDELT Events
table (2015 onward), grouped into four hotspots.

**Safeguards.** Features use only information available before each prediction. Models are always tested on years they never saw.
The 2025 to 2026 data is locked for one final test. Settings were fixed before results were seen, and the one change made afterward
(removing stale days) is reported alongside the original.

**Limits.** Daily data cannot capture reactions that happen within minutes. Yahoo's futures series has small jumps at contract
rolls. GDELT tone is a noisy measure. The sample is about 2,900 trading days. Backtests assume trading at the close, which is
slightly optimistic. A negative result means these features did not help in this design, not that news never matters.

**Not financial advice.** This is a student research project.
""")

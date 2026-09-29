"""
Weekly put scanner + AI review + order placement.

Run:  streamlit run app.py
Data: Yahoo Finance via yfinance (delayed ~15 min).
AI:   Ollama (free, local), Gemini, or Claude — see ai.py
Orders: Alpaca, paper trading by default — see broker.py
"""
import math
import os
import time
from datetime import date, datetime

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import ai
import broker

st.set_page_config(page_title="Weekly put scanner", page_icon="📉", layout="wide")

# ---------- password gate ----------
# Required when deployed (Render sets RENDER=true). Optional locally.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if os.getenv("RENDER") and not APP_PASSWORD:
    st.error("APP_PASSWORD is not set. Add it in Render → Environment before using this app.")
    st.stop()
if APP_PASSWORD and not st.session_state.get("authed"):
    import hmac

    st.title("Weekly put scanner")
    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw, APP_PASSWORD):
        st.session_state.authed = True
        st.rerun()
    if pw:
        st.error("Wrong password.")
    st.stop()

MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "1"))
MAX_COLLATERAL_PCT = float(os.getenv("MAX_COLLATERAL_PCT", "0.25"))
LIVE_ALLOWED = os.getenv("LIVE_TRADING", "").lower() == "true"
STALE_SECONDS = 300


# ---------- math ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_delta(spot, strike, t_years, iv, rate):
    if iv <= 0 or t_years <= 0 or strike <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv**2) * t_years) / (iv * math.sqrt(t_years))
    return norm_cdf(d1) - 1


# ---------- data ----------
def pick_expiry(expirations, min_dte, max_dte):
    today = date.today()
    for exp in expirations:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if min_dte <= dte <= max_dte:
            return exp, dte
    return None, None


@st.cache_data(ttl=120, show_spinner=False)
def scan_ticker(ticker, target_delta, min_dte, max_dte, rate):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1mo")
    if hist.empty:
        return None, f"{ticker}: no price data"
    spot = float(hist["Close"].iloc[-1])
    trend = hist["Close"].reset_index()
    trend.columns = ["Date", "Close"]

    expiry, dte = pick_expiry(list(tk.options), min_dte, max_dte)
    if expiry is None:
        return None, f"{ticker}: no expiration {min_dte}–{max_dte} days out"

    puts = tk.option_chain(expiry).puts.copy()
    puts = puts[(puts["strike"] < spot) & (puts["bid"] > 0) & (puts["impliedVolatility"] > 0.01)]
    if puts.empty:
        return None, f"{ticker}: no out-of-the-money puts with a bid"

    t_years = max(dte, 1) / 365
    puts["delta"] = puts.apply(
        lambda r: put_delta(spot, r["strike"], t_years, r["impliedVolatility"], rate), axis=1
    )
    puts = puts.dropna(subset=["delta"])
    puts["dist"] = (puts["delta"].abs() - target_delta).abs()
    best = puts.loc[puts["dist"].idxmin()]

    bid, ask, strike = float(best["bid"]), float(best["ask"]), float(best["strike"])
    mid = (bid + ask) / 2 if ask > 0 else bid
    yld = bid / strike
    row = {
        "Ticker": ticker,
        "Contract": best["contractSymbol"],
        "Price": spot,
        "Expiry": expiry,
        "DTE": dte,
        "Strike": strike,
        "Delta": float(best["delta"]),
        "Bid": bid,
        "Mid": mid,
        "IV": float(best["impliedVolatility"]),
        "Premium / contract": bid * 100,
        "Collateral": strike * 100,
        "Yield": yld,
        "Annualized": yld * 365 / max(dte, 1),
        "OTM %": 1 - strike / spot,
        "1M change": spot / float(hist["Close"].iloc[0]) - 1,
    }
    return {"row": row, "trend": trend}, None


# ---------- sidebar ----------
with st.sidebar:
    st.header("Scan settings")
    tickers_raw = st.text_input("Tickers (comma separated)", "NVDA, META, NVDL, SOXL")
    target = st.slider("Target delta", 0.05, 0.40, 0.20, 0.01)
    dte_range = st.slider("Days to expiration", 1, 21, (4, 10))
    rate = st.number_input("Risk-free rate", 0.0, 0.10, 0.04, 0.005, format="%.3f")
    run = st.button("Scan", type="primary", use_container_width=True)

st.title("Weekly put scanner")
st.caption(
    f"Finds the put closest to {target:.2f} delta on the nearest expiration "
    f"{dte_range[0]}–{dte_range[1]} days out, ranked by bid ÷ strike."
)

if run:
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    results, trends, errors = [], {}, []
    with st.spinner("Pulling option chains…"):
        for t in tickers:
            try:
                data, err = scan_ticker(t, target, dte_range[0], dte_range[1], rate)
            except Exception as e:
                data, err = None, f"{t}: {e}"
            if err:
                errors.append(err)
            else:
                results.append(data["row"])
                trends[t] = data["trend"]
    st.session_state.update(
        df=pd.DataFrame(results).sort_values("Yield", ascending=False).reset_index(drop=True)
        if results else None,
        trends=trends,
        errors=errors,
        scanned_at=time.time(),
        analysis=None,
    )

if "df" not in st.session_state:
    st.info("Set your tickers and delta in the sidebar, then select Scan.")
    st.stop()

for e in st.session_state.errors:
    st.warning(e)
df = st.session_state.df
if df is None:
    st.error("No results. Check the tickers or widen the days-to-expiration range.")
    st.stop()

age = time.time() - st.session_state.scanned_at
top = df.iloc[0]

# ---------- results ----------
c1, c2, c3 = st.columns(3)
c1.metric("Highest yield", top["Ticker"], f"{top['Yield']:.2%} this week")
c2.metric("Premium per contract", f"${top['Premium / contract']:,.0f}", f"{top['Strike']:g} strike")
c3.metric("Annualized", f"{top['Annualized']:.0%}", f"{top['DTE']} days to expiry")

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("Ticker:N", sort="-y", title=None),
        y=alt.Y("Yield:Q", axis=alt.Axis(format="%"), title="Premium ÷ strike"),
        tooltip=["Ticker", "Strike", "Bid", alt.Tooltip("Yield:Q", format=".2%"),
                 alt.Tooltip("Delta:Q", format=".2f"), alt.Tooltip("IV:Q", format=".0%")],
    )
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

st.dataframe(
    df.style.format({
        "Price": "${:,.2f}", "Strike": "${:,.2f}", "Delta": "{:.2f}", "Bid": "${:.2f}",
        "Mid": "${:.2f}", "IV": "{:.0%}", "Premium / contract": "${:,.0f}",
        "Collateral": "${:,.0f}", "Yield": "{:.2%}", "Annualized": "{:.0%}",
        "OTM %": "{:.1%}", "1M change": "{:+.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("1 month trend")
cols = st.columns(len(df))
for col, ticker in zip(cols, df["Ticker"]):
    with col:
        st.caption(ticker)
        st.line_chart(st.session_state.trends[ticker], x="Date", y="Close", height=140)

# ---------- AI review ----------
st.divider()
st.subheader("AI review")
a1, a2 = st.columns(2)
provider = a1.selectbox("Provider", list(ai.PROVIDERS))
model = a2.text_input("Model", ai.PROVIDERS[provider][1], key=f"model_{provider}")

if st.button("Analyze with AI"):
    rows = df.drop(columns=["Contract"]).round(4).to_dict("records")
    with st.spinner(f"Asking {provider}…"):
        try:
            st.session_state.analysis = ai.analyze(provider, model, rows)
        except Exception as e:
            st.error(f"AI request failed: {e}")

analysis = st.session_state.get("analysis")
if analysis:
    st.markdown(f"**Pick: {analysis.get('pick')}** · {analysis.get('confidence', '?')} confidence")
    st.write(analysis.get("summary", ""))
    if analysis.get("risks"):
        st.markdown("**Risks**\n" + "\n".join(f"- {r}" for r in analysis["risks"]))
    for t, note in analysis.get("per_ticker", {}).items():
        st.caption(f"{t}: {note}")

# ---------- order ----------
st.divider()
st.subheader("Place order")

tickers_list = list(df["Ticker"])
pick = analysis.get("pick") if analysis else None
default_idx = tickers_list.index(pick) if pick in tickers_list else 0
ticker = st.selectbox("Contract to sell", tickers_list, index=default_idx)
row = df[df["Ticker"] == ticker].iloc[0]

o1, o2, o3 = st.columns(3)
qty = int(o1.number_input("Contracts", 1, MAX_CONTRACTS, 1, key=f"qty_{ticker}"))
limit = o2.number_input(
    "Limit price per share", 0.01, value=broker.round_price(float(row["Mid"])),
    step=0.01, format="%.2f", key=f"limit_{ticker}",
)
mode = o3.radio("Account", ["Paper", "Live"] if LIVE_ALLOWED else ["Paper"], horizontal=True)

collateral = float(row["Strike"]) * 100 * qty
st.write(
    f"Sell to open **{qty} × {row['Contract']}** "
    f"({ticker} ${row['Strike']:g} put, expires {row['Expiry']}) at **${limit:.2f}** limit. "
    f"Collects about **${limit * 100 * qty:,.0f}**, ties up **${collateral:,.0f}** collateral."
)

stale = age > STALE_SECONDS
if stale:
    st.warning("Quotes are over 5 minutes old. Re-scan before placing an order.")

confirm = st.checkbox(f"I've checked this against my broker and want to place it ({mode})")
if st.button("Place order", type="primary", disabled=stale or not confirm):
    try:
        tc = broker.client(paper=(mode == "Paper"))
        bp = broker.buying_power(tc)
        if collateral > bp * MAX_COLLATERAL_PCT:
            st.error(
                f"Blocked: ${collateral:,.0f} collateral is over {MAX_COLLATERAL_PCT:.0%} "
                f"of your ${bp:,.0f} buying power."
            )
        else:
            order = broker.sell_put(tc, row["Contract"], qty, limit)
            st.success(f"Order submitted to {mode}: {order.id} — status {order.status}")
    except KeyError:
        st.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in your .env file.")
    except Exception as e:
        st.error(f"Order failed: {e}")

st.caption(
    "Data from Yahoo Finance, delayed. The AI only reviews; strike, size, and price come "
    "from the scan and your inputs. Not financial advice."
)

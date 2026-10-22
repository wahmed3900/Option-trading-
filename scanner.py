"""
Options scanner — finds the put nearest a target delta for a given ticker.
Called by api.py; also usable standalone.
"""
import math
import time
from datetime import date, datetime


def num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_delta(spot, strike, t_years, iv, rate):
    if iv <= 0 or t_years <= 0 or strike <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv**2) * t_years) / (iv * math.sqrt(t_years))
    return norm_cdf(d1) - 1


def pick_expiry(expirations, min_dte, max_dte):
    today = date.today()
    for exp in expirations:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if min_dte <= dte <= max_dte:
            return exp, dte
    return None, None


def scan_ticker(ticker, target, min_dte, max_dte, rate=0.04):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="1mo")
    if hist.empty:
        raise ValueError(f"{ticker}: no price data")
    closes = [round(num(c), 2) for c in hist["Close"].tolist()]
    spot = closes[-1]

    expiry, dte = pick_expiry(list(tk.options), min_dte, max_dte)
    if expiry is None:
        raise ValueError(f"{ticker}: no expiration {min_dte}-{max_dte} days out")

    puts = tk.option_chain(expiry).puts.copy()
    puts = puts[(puts["strike"] < spot) & (puts["bid"] > 0) & (puts["impliedVolatility"] > 0.01)]
    if puts.empty:
        raise ValueError(f"{ticker}: no out-of-the-money puts with a bid")

    t_years = max(dte, 1) / 365
    puts["delta"] = puts.apply(
        lambda r: put_delta(spot, r["strike"], t_years, r["impliedVolatility"], rate), axis=1
    )
    puts = puts.dropna(subset=["delta"])
    puts["dist"] = (puts["delta"].abs() - target).abs()
    best = puts.loc[puts["dist"].idxmin()]

    bid, ask, strike = num(best["bid"]), num(best["ask"]), num(best["strike"])
    mid = (bid + ask) / 2 if ask > 0 else bid
    yld = bid / strike
    contract = str(best["contractSymbol"])

    return {
        "ticker": ticker, "contract": contract, "price": spot, "expiry": expiry, "dte": dte,
        "strike": strike, "delta": round(num(best["delta"]), 3), "bid": bid, "mid": round(mid, 2),
        "iv": round(num(best["impliedVolatility"]), 4), "premium": round(bid * 100, 2),
        "collateral": round(strike * 100, 2), "yield": yld, "annualized": yld * 365 / max(dte, 1),
        "otm": 1 - strike / spot, "change_1m": spot / closes[0] - 1 if closes[0] else 0.0,
        "trend": closes,
    }

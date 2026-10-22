"""
Order placement — supports Alpaca (paper + live) and Robinhood (live only).
Set BROKER env var to choose:
  BROKER=alpaca    (default) — paper or live via ALPACA_API_KEY / ALPACA_SECRET_KEY
  BROKER=robinhood           — live only via RH_USERNAME / RH_PASSWORD (+ RH_MFA_CODE if 2FA)

Get Alpaca keys at app.alpaca.markets (paper and live keys are separate sets).
"""
import os
import re

BROKER = os.getenv("BROKER", "alpaca").lower()


# ── shared ────────────────────────────────────────────────────────────────────

def round_price(price):
    """Options tick: $0.01 under $3, $0.05 at or above $3."""
    tick = 0.01 if price < 3 else 0.05
    return round(max(tick, round(price / tick) * tick), 2)


# ── Alpaca ────────────────────────────────────────────────────────────────────

def _alpaca_client(paper=True):
    from alpaca.trading.client import TradingClient
    return TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper
    )


def _alpaca_buying_power(tc):
    acct = tc.get_account()
    bp = getattr(acct, "options_buying_power", None) or acct.buying_power
    return float(bp)


def _alpaca_sell_put(tc, contract_symbol, qty, limit_price):
    from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    order = LimitOrderRequest(
        symbol=contract_symbol,
        qty=qty,
        side=OrderSide.SELL,
        position_intent=PositionIntent.SELL_TO_OPEN,
        time_in_force=TimeInForce.DAY,
        limit_price=round_price(limit_price),
    )
    return tc.submit_order(order)


# ── Robinhood ─────────────────────────────────────────────────────────────────

def _rh_login():
    import robin_stocks.robinhood as rh
    username = os.environ["RH_USERNAME"]
    password = os.environ["RH_PASSWORD"]
    mfa_code = os.getenv("RH_MFA_CODE", "")
    kwargs = {"store_session": False}
    if mfa_code:
        kwargs["mfa_code"] = mfa_code
    rh.login(username, password, **kwargs)
    return rh


def _parse_occ(contract_symbol):
    """Parse OCC symbol e.g. AAPL251219P00150000 → (ticker, expiry, strike, type)."""
    m = re.match(r"^([A-Z1-9]+)(\d{6})([CP])(\d{8})$", contract_symbol.upper())
    if not m:
        raise ValueError(f"Cannot parse OCC symbol: {contract_symbol}")
    ticker = m.group(1)
    d = m.group(2)                          # YYMMDD
    opt_type = "put" if m.group(3) == "P" else "call"
    strike = int(m.group(4)) / 1000.0
    expiry = f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
    return ticker, expiry, strike, opt_type


class _RHResult:
    """Thin wrapper so Robinhood order dicts look like Alpaca order objects."""
    def __init__(self, d):
        self.id = d.get("id", "")
        self.status = d.get("state", "unknown")


def _rh_client(paper=False):
    if paper:
        raise ValueError("Robinhood does not support paper trading; use account='live'.")
    return _rh_login()


def _rh_buying_power(rh):
    profile = rh.profiles.load_account_profile()
    bp = (
        profile.get("option_buying_power")
        or profile.get("buying_power")
        or "0"
    )
    return float(bp)


def _rh_sell_put(rh, contract_symbol, qty, limit_price):
    ticker, expiry, strike, opt_type = _parse_occ(contract_symbol)
    result = rh.orders.order_sell_option_limit(
        positionEffect="open",
        creditOrDebit="credit",
        price=round_price(limit_price),
        symbol=ticker,
        quantity=qty,
        expirationDate=expiry,
        strike=strike,
        optionType=opt_type,
    )
    if not isinstance(result, dict) or "id" not in result:
        raise RuntimeError(f"Robinhood rejected the order: {result}")
    return _RHResult(result)


# ── public interface (used by api.py) ─────────────────────────────────────────

def client(paper=True):
    if BROKER == "robinhood":
        return _rh_client(paper)
    return _alpaca_client(paper)


def buying_power(tc):
    if BROKER == "robinhood":
        return _rh_buying_power(tc)
    return _alpaca_buying_power(tc)


def sell_put(tc, contract_symbol, qty, limit_price):
    if BROKER == "robinhood":
        return _rh_sell_put(tc, contract_symbol, qty, limit_price)
    return _alpaca_sell_put(tc, contract_symbol, qty, limit_price)

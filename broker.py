"""
Order placement through Alpaca (free paper trading, supports options).
Keys (in .env): ALPACA_API_KEY, ALPACA_SECRET_KEY
Paper keys and live keys are different; get both at app.alpaca.markets.
"""
import os


def client(paper=True):
    from alpaca.trading.client import TradingClient

    return TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper
    )


def buying_power(tc):
    acct = tc.get_account()
    bp = getattr(acct, "options_buying_power", None) or acct.buying_power
    return float(bp)


def round_price(price):
    """Options tick: $0.01 under $3, $0.05 at or above $3."""
    tick = 0.01 if price < 3 else 0.05
    return round(max(tick, round(price / tick) * tick), 2)


def sell_put(tc, contract_symbol, qty, limit_price):
    """Sell to open a put with a DAY limit order. contract_symbol is OCC format."""
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

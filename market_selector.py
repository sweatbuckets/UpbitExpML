import pandas as pd
import requests


MARKET_ALL_URL = "https://api.upbit.com/v1/market/all"
TICKER_URL = "https://api.upbit.com/v1/ticker"


def get_krw_markets():
    resp = requests.get(MARKET_ALL_URL, timeout=10)
    resp.raise_for_status()
    return [m["market"] for m in resp.json() if m["market"].startswith("KRW-")]


def get_tickers(markets):
    if not markets:
        return []
    resp = requests.get(TICKER_URL, params={"markets": ",".join(markets)}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def select_top_volatile_symbols(n):
    markets = get_krw_markets()
    tickers = get_tickers(markets)
    if not tickers:
        return []

    df = pd.DataFrame(tickers)
    df["volatility_rate"] = df["signed_change_rate"].abs()
    return df.sort_values("volatility_rate", ascending=False).head(n)["market"].tolist()


def select_top_volatile_symbol():
    symbols = select_top_volatile_symbols(1)
    return symbols[0] if symbols else None

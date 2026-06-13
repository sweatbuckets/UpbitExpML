import time

import config
from market_selector import get_krw_markets, get_tickers, select_top_volatile_symbols
from ws_collector import WSTickCollector


CHECK_INTERVAL = config.INTERVAL_SEC
TOP_N = config.SELECT_TOP_N
SPAWN_WS = {}


def get_markets():
    try:
        return get_krw_markets()
    except Exception as exc:
        print(f"get_markets error: {exc}")
        return []


def detect_high_volatility():
    symbols = select_top_volatile_symbols(TOP_N)
    new_symbols = [symbol for symbol in symbols if symbol not in SPAWN_WS]
    if not new_symbols:
        return

    print("High volatility detected:", new_symbols)
    collector = WSTickCollector(new_symbols, ticket="tick_service")
    collector.start()
    SPAWN_WS.update({symbol: collector for symbol in new_symbols})


def detect_price_spike():
    return detect_high_volatility()


if __name__ == "__main__":
    print(f"Starting Upbit high volatility collector. interval={CHECK_INTERVAL}s top_n={TOP_N}")
    try:
        while True:
            detect_high_volatility()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("Stopped by user")

import time

import config
from upbit_client import WSTickCollector, get_krw_markets, select_top_volatile_symbols


CHECK_INTERVAL = config.INTERVAL_SEC
LIQUIDITY_CANDIDATE_N = 30
TOP_N = 5
SPAWN_WS = {}


# REST API 연결 확인용 KRW 마켓 조회 wrapper
def get_markets():
    try:
        return get_krw_markets()
    except Exception as exc:
        print(f"get_markets error: {exc}")
    return []


# 변동성 상위 종목을 찾아 신규 종목만 WebSocket 구독
def detect_high_volatility():
    symbols = select_top_volatile_symbols(
        TOP_N,
        liquidity_candidate_n=LIQUIDITY_CANDIDATE_N,
    )
    new_symbols = [symbol for symbol in symbols if symbol not in SPAWN_WS]
    if not new_symbols:
        return

    print("High volatility detected:", new_symbols)
    collector = WSTickCollector(new_symbols, ticket="tick_service")
    collector.start()
    SPAWN_WS.update({symbol: collector for symbol in new_symbols})


# 예전 함수명 호환용 alias
def detect_price_spike():
    return detect_high_volatility()


if __name__ == "__main__":
    print(
        "Starting Upbit high volatility collector. "
        f"interval={CHECK_INTERVAL}s candidate_n={LIQUIDITY_CANDIDATE_N} top_n={TOP_N}"
    )
    try:
        while True:
            detect_high_volatility()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("Stopped by user")

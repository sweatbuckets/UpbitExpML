import json
import logging
import threading
import time
from collections import deque

import pandas as pd
import requests
import websocket


# Upbit REST endpoint
MARKET_ALL_URL = "https://api.upbit.com/v1/market/all"
TICKER_URL = "https://api.upbit.com/v1/ticker"


# KRW 마켓 전체 조회
def get_krw_markets():
    resp = requests.get(MARKET_ALL_URL, timeout=10)
    resp.raise_for_status()
    return [market["market"] for market in resp.json() if market["market"].startswith("KRW-")]


# 선택된 마켓들의 현재 ticker 정보 조회
def get_tickers(markets):
    if not markets:
        return []
    resp = requests.get(TICKER_URL, params={"markets": ",".join(markets)}, timeout=10)
    resp.raise_for_status()
    return resp.json()


# 24시간 거래대금 상위 후보군 안에서 전일 종가 대비 등락률 절댓값이 큰 종목 N개 선택
# acc_trade_price_24h는 24시간 누적 거래대금, signed_change_rate는 전일 종가 대비 등락률
def select_top_volatile_symbols(n, liquidity_candidate_n=30):
    markets = get_krw_markets()
    tickers = get_tickers(markets)
    if not tickers:
        return []

    df = pd.DataFrame(tickers)
    df = df.sort_values("acc_trade_price_24h", ascending=False).head(liquidity_candidate_n).copy()
    df["absolute_change_rate"] = df["signed_change_rate"].abs()
    display_df = df[["market", "acc_trade_price_24h", "absolute_change_rate"]].copy()
    display_df["trade_value_24h_b_krw"] = display_df["acc_trade_price_24h"] / 100_000_000
    display_df["volatility_pct"] = display_df["absolute_change_rate"] * 100
    display_df = display_df[["market", "trade_value_24h_b_krw", "volatility_pct"]]
    logging.info(
        "Liquidity candidates by 24h trade value:\n%s",
        display_df.to_string(
            index=False,
            formatters={
                "trade_value_24h_b_krw": "{:,.1f}".format,
                "volatility_pct": "{:.2f}%".format,
            },
        ),
    )

    selected = df.sort_values("absolute_change_rate", ascending=False).head(n)
    logging.info("Selected symbols: %s", selected["market"].tolist())
    return selected["market"].tolist()


# 실시간 추론/검증용 단일 종목 선택
def select_top_volatile_symbol(liquidity_candidate_n=30):
    symbols = select_top_volatile_symbols(1, liquidity_candidate_n=liquidity_candidate_n)
    return symbols[0] if symbols else None


# Upbit WebSocket 체결/호가 수집기
class WSTickCollector:
    def __init__(self, markets, maxlen=5000, ticket="ml_collector"):
        # 구독 대상 종목 목록
        self.markets = list(markets)

        # 종목별 체결 tick buffer
        self.ticks = {market: deque(maxlen=maxlen) for market in self.markets}

        # 종목별 orderbook row buffer
        self.orderbooks = {market: deque(maxlen=maxlen) for market in self.markets}

        # WebSocket callback thread와 수집 loop 사이의 buffer 보호용 lock
        self.lock = threading.Lock()

        # Upbit WebSocket subscription ticket
        self.ticket = ticket

        # 현재 WebSocketApp 인스턴스와 종료 이벤트
        self.ws = None
        self._stop = threading.Event()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            market = data.get("code") or data.get("market")
            if not market or market not in self.ticks:
                return

            # 체결 데이터 저장
            if "trade_price" in data:
                tick = {
                    "market": market,
                    "trade_price": float(data.get("trade_price", 0)),
                    "trade_volume": float(data.get("trade_volume", 0)),
                    # Upbit timestamp는 millisecond epoch
                    "timestamp": data.get("timestamp"),
                    "ask_bid": data.get("ask_bid"),
                }
                with self.lock:
                    self.ticks[market].append(tick)
                return

            # 호가 데이터 저장
            if "orderbook_units" in data:
                timestamp = data.get("timestamp")
                orderbook_rows = [
                    {
                        "market": market,
                        # 동일한 orderbook snapshot timestamp를 각 호가 row에 부여
                        "timestamp": timestamp,
                        "bid_price": float(unit["bid_price"]),
                        "bid_size": float(unit["bid_size"]),
                        "ask_price": float(unit["ask_price"]),
                        "ask_size": float(unit["ask_size"]),
                    }
                    for unit in data["orderbook_units"]
                ]
                with self.lock:
                    self.orderbooks[market].extend(orderbook_rows)
        except Exception as exc:
            logging.debug("WS message parse error: %s", exc)

    def on_open(self, ws):
        # 연결이 열릴 때마다 같은 종목을 다시 구독
        payload = [
            {"ticket": self.ticket},
            {"type": "trade", "codes": self.markets, "isOnlyRealtime": True},
            {"type": "orderbook", "codes": self.markets},
        ]
        ws.send(json.dumps(payload))
        logging.info("WebSocket subscription sent for %d symbols", len(self.markets))

    def on_error(self, ws, error):
        logging.warning("WebSocket error: %s", error)

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning(
            "WebSocket closed: code=%s msg=%s",
            close_status_code,
            close_msg,
        )

    def start(self):
        def run_ws():
            while not self._stop.is_set():
                # run_forever가 종료되면 같은 markets/ticket으로 5초 후 자동 재연결
                self.ws = websocket.WebSocketApp(
                    "wss://api.upbit.com/websocket/v1",
                    on_message=self.on_message,
                    on_open=self.on_open,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
                if not self._stop.is_set():
                    logging.info("WebSocket disconnected. Reconnecting in 5 seconds...")
                    time.sleep(5)

        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()

        time.sleep(1)

    def stop(self):
        self._stop.set()
        if self.ws is not None:
            self.ws.close()

    def pop_all(self):
        # 수집 루프에서 interval 단위로 데이터를 꺼내고 buffer를 비운다. deque 비우기 전에 Lock 걸어서 WebSocket 콜백과 충돌 방지
        with self.lock:
            out_ticks = {market: list(self.ticks[market]) for market in self.markets}
            out_orderbooks = {
                market: list(self.orderbooks[market]) for market in self.markets
            }
            for market in self.markets:
                self.ticks[market].clear()
                self.orderbooks[market].clear()
        return out_ticks, out_orderbooks

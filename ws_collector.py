import json
import logging
import threading
import time
from collections import deque

import websocket


class WSTickCollector:
    def __init__(self, markets, maxlen=5000, ticket="ml_collector"):
        self.markets = list(markets)
        self.ticks = {market: deque(maxlen=maxlen) for market in self.markets}
        self.orderbooks = {market: deque(maxlen=maxlen) for market in self.markets}
        self.lock = threading.Lock()
        self.ticket = ticket
        self.ws = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            market = data.get("code") or data.get("market")
            if not market or market not in self.ticks:
                return

            if "trade_price" in data:
                tick = {
                    "market": market,
                    "trade_price": float(data.get("trade_price", 0)),
                    "trade_volume": float(data.get("trade_volume", 0)),
                    "timestamp": data.get("timestamp"),
                    "ask_bid": data.get("ask_bid"),
                }
                with self.lock:
                    self.ticks[market].append(tick)
                return

            if "orderbook_units" in data:
                ts = data.get("timestamp")
                orderbook_rows = [
                    {
                        "market": market,
                        "timestamp": ts,
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
        payload = [
            {"ticket": self.ticket},
            {"type": "trade", "codes": self.markets, "isOnlyRealtime": True},
            {"type": "orderbook", "codes": self.markets},
        ]
        ws.send(json.dumps(payload))
        logging.info("WebSocket subscription sent for %d symbols", len(self.markets))

    def start(self):
        def run_ws():
            self.ws = websocket.WebSocketApp(
                "wss://api.upbit.com/websocket/v1",
                on_message=self.on_message,
                on_open=self.on_open,
            )
            self.ws.run_forever()

        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        time.sleep(1)

    def pop_all(self):
        with self.lock:
            out_ticks = {market: list(self.ticks[market]) for market in self.markets}
            out_orderbooks = {
                market: list(self.orderbooks[market]) for market in self.markets
            }
            for market in self.markets:
                self.ticks[market].clear()
                self.orderbooks[market].clear()
        return out_ticks, out_orderbooks

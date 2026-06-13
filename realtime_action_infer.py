import logging
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch

import config
import feature_engineering as fe
from market_selector import select_top_volatile_symbol
from model import CNNLSTM
from ws_collector import WSTickCollector


ACTION_MAP = {"sell": 0, "hold": 1, "buy": 2}
INV_ACTION_MAP = {value: key for key, value in ACTION_MAP.items()}


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(device):
    model = CNNLSTM(len(fe.FEATURE_COLS))
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    return model.to(device).eval()


def append_closed_intervals(collector, pending_ticks, pending_orderbooks, market_history):
    ticks_by_market, orderbooks_by_market = collector.pop_all()
    market = collector.markets[0]

    ticks, pending_ticks = fe.split_closed_records(
        pending_ticks + ticks_by_market.get(market, []),
        interval_sec=config.INTERVAL_SEC,
    )
    orderbooks, pending_orderbooks = fe.split_closed_records(
        pending_orderbooks + orderbooks_by_market.get(market, []),
        interval_sec=config.INTERVAL_SEC,
    )
    agg = fe.aggregate_interval(ticks, orderbooks, interval_sec=config.INTERVAL_SEC)
    if agg is None or agg.empty:
        return market_history, pending_ticks, pending_orderbooks, None

    market_history = (
        pd.concat([market_history, agg], ignore_index=True)
        .drop_duplicates("interval")
        .sort_values("interval")
        .reset_index(drop=True)
    )
    feat_df = fe.compute_features_one_market(market_history)
    return market_history, pending_ticks, pending_orderbooks, feat_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    market = select_top_volatile_symbol()
    if not market:
        logging.error("No symbol selected. Exiting.")
        raise SystemExit(1)
    logging.info("Selected volatile symbol: %s", market)

    device = get_device()
    model = load_model(device)
    scaler = joblib.load(config.SCALER_PATH)
    logging.info("CNN-LSTM model loaded on device: %s", device)

    collector = WSTickCollector([market], ticket="ml_realtime_infer")
    collector.start()
    logging.info("WebSocket collector started for %s", market)

    market_history = pd.DataFrame()
    history = deque(maxlen=config.SEQ_LEN)
    pending_ticks = []
    pending_orderbooks = []
    last_interval = None

    while True:
        time.sleep(config.INTERVAL_SEC)
        market_history, pending_ticks, pending_orderbooks, feat_df = append_closed_intervals(
            collector=collector,
            pending_ticks=pending_ticks,
            pending_orderbooks=pending_orderbooks,
            market_history=market_history,
        )
        if feat_df is None or feat_df.empty:
            continue

        min_interval = last_interval if last_interval else pd.Timestamp(0, tz="UTC")
        new_rows = feat_df[feat_df["interval"] > min_interval]
        for _, row in new_rows.iterrows():
            feature_vector = [row[col] for col in fe.FEATURE_COLS]
            history.append(scaler.transform([feature_vector])[0])
            last_interval = row["interval"]

            if len(history) < config.SEQ_LEN:
                logging.info("[%s] Warming up (%d/%d)", row["interval"], len(history), config.SEQ_LEN)
                continue

            x_tensor = torch.from_numpy(np.array(history, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_tensor).argmax(1).item()

            logging.info("[%s] Predicted Action: %s", row["interval"], INV_ACTION_MAP[pred])


if __name__ == "__main__":
    main()

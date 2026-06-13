import logging
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import config
import feature_engineering as fe
from market_selector import select_top_volatile_symbol
from realtime_action_infer import ACTION_MAP, INV_ACTION_MAP, append_closed_intervals, get_device, load_model
from ws_collector import WSTickCollector


def label_from_return(return_value):
    if return_value >= config.LABEL_THRESHOLD:
        return ACTION_MAP["buy"]
    if return_value <= -config.LABEL_THRESHOLD:
        return ACTION_MAP["sell"]
    return ACTION_MAP["hold"]


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

    collector = WSTickCollector([market], ticket="ml_realtime_check")
    collector.start()
    logging.info("WebSocket collector started for %s", market)

    market_history = pd.DataFrame()
    history = deque(maxlen=config.SEQ_LEN)
    pending_ticks = []
    pending_orderbooks = []
    last_interval = None
    pending_pred = None
    y_true, y_pred = [], []

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

            if pending_pred is not None:
                pred_interval, pred_class = pending_pred
                true_class = label_from_return(row["last_return"])
                y_true.append(true_class)
                y_pred.append(pred_class)

                logging.info(
                    "[pred@%s eval@%s] Pred=%s True=%s | Acc=%.3f F1=%.3f",
                    pred_interval,
                    row["interval"],
                    INV_ACTION_MAP[pred_class],
                    INV_ACTION_MAP[true_class],
                    accuracy_score(y_true, y_pred),
                    f1_score(y_true, y_pred, average="macro", zero_division=0),
                )

            if len(history) < config.SEQ_LEN:
                logging.info("[%s] Warming up (%d/%d)", row["interval"], len(history), config.SEQ_LEN)
                continue

            x_tensor = torch.from_numpy(np.array(history, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_tensor).argmax(1).item()

            pending_pred = (row["interval"], pred)


if __name__ == "__main__":
    main()

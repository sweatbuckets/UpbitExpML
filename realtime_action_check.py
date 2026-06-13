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
from realtime_action_infer import ACTION_MAP, INV_ACTION_MAP, append_closed_intervals, get_device, load_model
from upbit_client import WSTickCollector, select_top_volatile_symbol


# 다음 30초 실제 수익률을 sell / hold / buy 라벨로 변환
def label_from_return(return_value):
    if return_value >= config.LABEL_THRESHOLD:
        return ACTION_MAP["buy"]
    if return_value <= -config.LABEL_THRESHOLD:
        return ACTION_MAP["sell"]
    return ACTION_MAP["hold"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # 1. 현재 변동성이 가장 큰 KRW 종목 선택
    market = select_top_volatile_symbol()
    if not market:
        logging.error("No symbol selected. Exiting.")
        raise SystemExit(1)
    logging.info("Selected volatile symbol: %s", market)

    # 2. 모델과 scaler 로드
    device = get_device()
    model = load_model(device)
    scaler = joblib.load(config.SCALER_PATH)
    logging.info("CNN-LSTM model loaded on device: %s", device)

    # 3. 선택 종목 WebSocket 수집 시작
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

        # 4. 새로 닫힌 30초봉을 feature로 변환
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
            # 5. 학습 때 저장한 scaler로 실시간 feature 표준화
            feature_vector = [row[col] for col in fe.FEATURE_COLS]
            history.append(scaler.transform([feature_vector])[0])
            last_interval = row["interval"]

            # 6. 직전 interval에서 만든 예측을 현재 interval의 실제 수익률로 평가
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

            # 7. 최근 10개 feature row가 쌓일 때까지 warm-up
            if len(history) < config.SEQ_LEN:
                logging.info("[%s] Warming up (%d/%d)", row["interval"], len(history), config.SEQ_LEN)
                continue

            # 8. 이번 interval 예측은 다음 interval에서 평가
            x_tensor = torch.from_numpy(np.array(history, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_tensor).argmax(1).item()

            pending_pred = (row["interval"], pred)


if __name__ == "__main__":
    main()

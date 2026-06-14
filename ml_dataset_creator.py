import logging
import os
import time

import numpy as np
import pandas as pd

import config
import feature_engineering as fe
from upbit_client import WSTickCollector, select_top_volatile_symbols


FEATURE_COLS = fe.FEATURE_COLS


# Upbit KRW 마켓 중 24시간 거래대금 상위 후보군 개수
LIQUIDITY_CANDIDATE_N = 30

# 거래대금 상위 후보군 안에서 전일 종가 대비 절대 변동률이 큰 상위 N개 종목을 구독
SELECT_TOP_N = 5

# CSV가 무한히 커지지 않도록 최근 MAX_SEQUENCES개 시퀀스만 유지
MAX_SEQUENCES = 2000

# 방금 끝난 30초봉은 WebSocket 지연 체결이 늦게 들어올 수 있어 1개 interval 뒤에 확정
CLOSE_DELAY_INTERVALS = 1


# 단일 종목 feature row를 모델 입력 시퀀스와 다음 30초 라벨로 변환
def create_sequences_one_market(df_feat, features, seq_len, threshold):
    label_len = 1
    if len(df_feat) < seq_len + label_len:
        return None, None, None

    x_rows, y_rows, seq_ids = [], [], []
    returns = df_feat["last_return"].values

    for start in range(len(df_feat) - seq_len - label_len + 1):
        # 입력: 최근 seq_len개 30초 feature
        x_rows.append(df_feat.iloc[start:start + seq_len][features].values)

        # 라벨: 시퀀스 종료 직후 다음 30초 수익률
        future_return = returns[start + seq_len]

        if future_return >= threshold:
            y_rows.append(2)
        elif future_return <= -threshold:
            y_rows.append(0)
        else:
            y_rows.append(1)

        seq_ids.append(df_feat["interval"].iloc[start])

    return np.array(x_rows), np.array(y_rows), seq_ids


# 여러 종목에서 생성된 시퀀스를 하나의 학습 batch 형태로 합침
def create_sequences_all_markets(features_by_market, features, seq_len, threshold):
    x_all, y_all, seq_ids_all = [], [], []

    for market, df_feat in features_by_market.items():
        x_market, y_market, seq_ids = create_sequences_one_market(
            df_feat=df_feat,
            features=features,
            seq_len=seq_len,
            threshold=threshold,
        )
        if x_market is None or len(x_market) == 0:
            continue

        x_all.append(x_market)
        y_all.append(y_market)
        seq_ids_all.extend((market, seq_id) for seq_id in seq_ids)

    if not x_all:
        return None, None, None

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), seq_ids_all


# 3D sequence 배열을 CSV 저장용 2D row로 펼침
def save_sequence_csv(x_all, y_all, feature_dim, seq_len, save_path, sequence_ids=None):
    if x_all is None or len(x_all) == 0:
        logging.info("No sequences to save.")
        return
    if sequence_ids is None:
        raise ValueError("sequence_ids is required for metadata CSV output.")

    num_sequences = x_all.shape[0]
    x_flat = x_all.reshape(num_sequences, seq_len * feature_dim)
    columns = [
        f"feature{feature_idx}_t{step_idx}"
        for step_idx in range(seq_len)
        for feature_idx in range(feature_dim)
    ]
    df = pd.DataFrame(x_flat, columns=columns)
    # 새 CSV 스키마는 모든 row에 추적 가능한 메타데이터를 함께 저장한다.
    df.insert(0, "market", [market for market, _ in sequence_ids])
    df.insert(
        1,
        "sequence_start_interval",
        [pd.to_datetime(interval).isoformat() for _, interval in sequence_ids],
    )
    df["label"] = y_all

    write_header = not os.path.exists(save_path) or os.path.getsize(save_path) == 0
    df.to_csv(save_path, mode="a", header=write_header, index=False)


# 종목별 30초봉 history를 갱신하고 ML-ready feature를 누적
def append_market_features(market, agg, market_history, features_by_market):
    market_history[market] = (
        pd.concat([market_history.get(market, pd.DataFrame()), agg], ignore_index=True)
        .drop_duplicates(subset=["interval"])
        .sort_values("interval")
        .reset_index(drop=True)
    )

    df_feat = fe.compute_features_one_market(market_history[market])
    if df_feat is None or df_feat.empty:
        logging.info("Market %s: initial intervals, waiting for full features...", market)
        return

    previous_len = len(features_by_market.get(market, pd.DataFrame()))
    features_by_market[market] = (
        pd.concat([features_by_market.get(market, pd.DataFrame()), df_feat], ignore_index=True)
        .drop_duplicates(subset=["interval"])
        .sort_values("interval")
        .reset_index(drop=True)
    )
    added_count = len(features_by_market[market]) - previous_len
    logging.info(
        "Market %s: +%d ML-ready intervals (total=%d)",
        market,
        added_count,
        len(features_by_market[market]),
    )


# WebSocket에서 모은 raw tick/orderbook 중 확정된 interval만 집계
def collect_closed_intervals(
    symbols,
    collector,
    pending_ticks,
    pending_orderbooks,
    close_delay_intervals=CLOSE_DELAY_INTERVALS,
):
    ticks_by_market, orderbooks_by_market = collector.pop_all()
    tick_summary = {market: len(ticks_by_market.get(market, [])) for market in symbols}
    logging.info("Interval collected ticks: %s", tick_summary)

    interval_data = {}
    for market in symbols:
        raw_ticks = pending_ticks.get(market, []) + ticks_by_market.get(market, [])
        raw_orderbooks = (
            pending_orderbooks.get(market, []) + orderbooks_by_market.get(market, [])
        )

        # 데이터셋 품질을 위해 방금 닫힌 봉은 한 번 더 보류한다.
        ticks, pending_ticks[market] = fe.split_closed_records(
            raw_ticks,
            interval_sec=config.INTERVAL_SEC,
            close_delay_intervals=close_delay_intervals,
        )
        orderbooks, pending_orderbooks[market] = fe.split_closed_records(
            raw_orderbooks,
            interval_sec=config.INTERVAL_SEC,
            close_delay_intervals=close_delay_intervals,
        )
        agg = fe.aggregate_interval(
            ticks=ticks,
            orderbooks=orderbooks,
            interval_sec=config.INTERVAL_SEC,
        )
        if agg is not None and not agg.empty:
            interval_data[market] = agg

    return interval_data


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Starting Upbit ML dataset collector...")

    config.DATASET_DIR.mkdir(exist_ok=True)

    # 1. 변동성 상위 종목 선택
    symbols = select_top_volatile_symbols(
        SELECT_TOP_N,
        liquidity_candidate_n=LIQUIDITY_CANDIDATE_N,
    )
    if not symbols:
        logging.error("No symbols selected. Exiting.")
        raise SystemExit(1)

    # 2. WebSocket 체결/호가 수집 시작
    collector = WSTickCollector(symbols, ticket="ml_dataset_collector")
    collector.start()
    logging.info("WebSocket collector started for %d symbols", len(symbols))

    seq_len = config.SEQ_LEN
    pending_ticks = {market: [] for market in symbols}
    pending_orderbooks = {market: [] for market in symbols}
    market_history = {}
    features_by_market = {}
    saved_sequence_ids = set()

    try:
        while True:
            time.sleep(config.INTERVAL_SEC)

            # 3. 확정된 30초 interval만 30초봉으로 집계
            interval_data = collect_closed_intervals(
                symbols=symbols,
                collector=collector,
                pending_ticks=pending_ticks,
                pending_orderbooks=pending_orderbooks,
            )

            # 4. 30초봉 history 기반 feature 계산
            for market, agg in interval_data.items():
                append_market_features(market, agg, market_history, features_by_market)

            # 5. 최근 10개 feature + 다음 30초 라벨 시퀀스 생성
            x_all, y_all, seq_ids = create_sequences_all_markets(
                features_by_market=features_by_market,
                features=FEATURE_COLS,
                seq_len=seq_len,
                threshold=config.LABEL_THRESHOLD,
            )
            if x_all is None:
                logging.info("No sequences yet")
                continue

            # 6. 이미 저장한 시퀀스는 제외하고 새 시퀀스만 CSV에 append
            remaining_slots = MAX_SEQUENCES - len(saved_sequence_ids)
            if remaining_slots <= 0:
                logging.info("Reached max sequences (%d). Stopping CSV save.", MAX_SEQUENCES)
                break

            x_new, y_new, sequence_ids_new = [], [], []
            for x_row, y_row, (market, start_interval) in zip(x_all, y_all, seq_ids):
                sequence_id = (market, pd.to_datetime(start_interval))
                if sequence_id in saved_sequence_ids:
                    continue
                if len(x_new) >= remaining_slots:
                    break
                saved_sequence_ids.add(sequence_id)
                x_new.append(x_row)
                y_new.append(y_row)
                sequence_ids_new.append(sequence_id)

            if x_new:
                x_new = np.array(x_new)
                y_new = np.array(y_new)
                save_sequence_csv(
                    x_all=x_new,
                    y_all=y_new,
                    feature_dim=len(FEATURE_COLS),
                    seq_len=seq_len,
                    save_path=config.CSV_PATH,
                    sequence_ids=sequence_ids_new,
                )
                logging.info(
                    "Saved new sequences: %d (total=%d, shape=%s)",
                    len(x_new),
                    len(saved_sequence_ids),
                    x_new.shape,
                )

            if len(saved_sequence_ids) >= MAX_SEQUENCES:
                logging.info("Reached max sequences (%d). Stopping CSV save.", MAX_SEQUENCES)
                break

            for market, df in features_by_market.items():
                features_by_market[market] = df.tail(seq_len + 1).reset_index(drop=True)

    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()

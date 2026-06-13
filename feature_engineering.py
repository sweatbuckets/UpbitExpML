import time

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "slope",
    "accel",
    "last_return",
    "cusum_pos",
    "cusum_neg",
    "volume_ratio",
    "bid_ask_imbalance",
    "spread_ratio",
]


def _records_to_frame(records):
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df.empty:
        return df

    ts_col = "timestamp"
    if ts_col not in df.columns and "trade_timestamp" in df.columns:
        ts_col = "trade_timestamp"

    if ts_col in df.columns and df[ts_col].notnull().any():
        df["timestamp"] = pd.to_datetime(df[ts_col].astype("int64"), unit="ms", utc=True)
    elif "trade_date" in df.columns and "trade_time" in df.columns:
        dt_str = df["trade_date"].astype(str) + df["trade_time"].astype(str)
        df["timestamp"] = pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    else:
        now_ms = int(time.time() * 1000)
        df["timestamp"] = pd.to_datetime([now_ms] * len(df), unit="ms", utc=True)

    return df.dropna(subset=["timestamp"])


def split_closed_records(records, interval_sec, now=None, close_delay_intervals=0):
    """Return records for closed intervals and keep current interval records pending."""
    if not records:
        return [], []

    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")

    cutoff = now.floor(f"{interval_sec}s") - pd.Timedelta(
        seconds=interval_sec * close_delay_intervals
    )
    closed = []
    pending = []

    for record in records:
        ts_value = record.get("timestamp") or record.get("trade_timestamp")
        if ts_value is None:
            closed.append(record)
            continue
        ts = pd.to_datetime(int(ts_value), unit="ms", utc=True)
        if ts < cutoff:
            closed.append(record)
        else:
            pending.append(record)

    return closed, pending


def aggregate_ticks(ticks, interval_sec=30):
    columns = ["interval", "open", "high", "low", "close", "volume", "tick_count"]
    if not ticks:
        return pd.DataFrame(columns=columns)

    df = _records_to_frame(ticks)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.sort_values("timestamp").set_index("timestamp")
    ohlcvt = df.resample(f"{interval_sec}s").agg(
        open=("trade_price", "first"),
        high=("trade_price", "max"),
        low=("trade_price", "min"),
        close=("trade_price", "last"),
        volume=("trade_volume", "sum"),
        tick_count=("trade_price", "count"),
    )

    if ohlcvt.empty:
        return ohlcvt.reset_index().rename(columns={"timestamp": "interval"})

    mask = ohlcvt["tick_count"] == 0
    ohlcvt.loc[mask, ["open", "high", "low", "close"]] = np.nan
    ohlcvt["close"] = ohlcvt["close"].ffill()
    for col in ("open", "high", "low"):
        ohlcvt[col] = ohlcvt[col].fillna(ohlcvt["close"])

    ohlcvt["volume"] = ohlcvt["volume"].fillna(0)
    ohlcvt["tick_count"] = ohlcvt["tick_count"].fillna(0)

    return ohlcvt.reset_index().rename(columns={"timestamp": "interval"})


def aggregate_orderbook(orderbooks, interval_sec=30):
    columns = ["interval", "bid_volume", "ask_volume", "bid_price", "ask_price"]
    if not orderbooks:
        return pd.DataFrame(columns=columns)

    df = _records_to_frame(orderbooks)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["interval"] = df["timestamp"].dt.floor(f"{interval_sec}s")
    agg = df.groupby("interval").agg(
        bid_volume=("bid_size", "sum"),
        ask_volume=("ask_size", "sum"),
        bid_price=("bid_price", "mean"),
        ask_price=("ask_price", "mean"),
    )

    return agg.reset_index()


def aggregate_interval(ticks, orderbooks, interval_sec=30):
    ohlcv = aggregate_ticks(ticks, interval_sec)
    ob = aggregate_orderbook(orderbooks, interval_sec)

    # tick과 orderbook이 모두 관측된 30초 interval만 모델 feature 후보로 사용한다.
    # orderbook 결측을 0으로 채우면 실제 시장 상태가 아닌 결측값을 정상 신호처럼 학습할 수 있다.
    if ohlcv.empty or ob.empty:
        return None

    merged = ohlcv.merge(ob, on="interval", how="inner")
    if merged.empty:
        return None
    return merged


def compute_features_one_market(agg: pd.DataFrame):
    if agg is None or len(agg) == 0:
        return None

    agg = agg.copy().sort_values("interval").reset_index(drop=True)
    agg["last_return"] = agg["close"].pct_change().fillna(0.0)

    slopes = [0.0]
    accels = [0.0]
    for i in range(1, len(agg)):
        prev_close = agg.at[i - 1, "close"]
        slope_i = 0.0 if prev_close == 0 else (agg.at[i, "close"] - prev_close) / prev_close
        slopes.append(slope_i)
        accels.append(slope_i - slopes[i - 1])

    agg["slope"] = slopes
    agg["accel"] = accels

    rolling_vol_mean = agg["volume"].rolling(2, min_periods=1).mean().shift(1)
    agg["volume_ratio"] = (
        (agg["volume"] / rolling_vol_mean)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    cusum_pos = [0.0]
    cusum_neg = [0.0]
    for i in range(1, len(agg)):
        delta = agg.at[i, "last_return"]
        cusum_pos.append(max(0.0, cusum_pos[i - 1] + delta))
        cusum_neg.append(min(0.0, cusum_neg[i - 1] + delta))
    agg["cusum_pos"] = cusum_pos
    agg["cusum_neg"] = cusum_neg

    denom = agg["bid_volume"] + agg["ask_volume"]
    agg["bid_ask_imbalance"] = np.where(
        denom != 0,
        (agg["bid_volume"] - agg["ask_volume"]) / denom,
        0,
    )

    mid = (agg["bid_price"] + agg["ask_price"]) / 2
    agg["spread_ratio"] = np.where(
        mid != 0,
        (agg["ask_price"] - agg["bid_price"]) / mid,
        0,
    )

    df_ml = agg[["interval"] + FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    if len(df_ml) <= 2:
        return pd.DataFrame(columns=["interval"] + FEATURE_COLS)
    return df_ml.iloc[2:].reset_index(drop=True)

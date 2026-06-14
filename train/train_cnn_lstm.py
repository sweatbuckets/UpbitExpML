import copy
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TMP_DIR = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "upbitexp-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_DIR / "upbitexp-cache"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import config
import feature_engineering as fe
from model import CNNLSTM


PER_STEP_FEATURE = len(fe.FEATURE_COLS)
FEATURE_COLUMN_PATTERN = re.compile(r"^feature(\d+)_t(\d+)$")

# 학습 전용 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 3e-5
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5
VAL_RATIO = 0.2
SPLIT_GAP_SIZE = config.SEQ_LEN
BEST_START_EPOCH = 10


# dataset 정의
class SequenceDataset(Dataset):
    def __init__(self, x_values, y_values):
        self.x_values = torch.tensor(x_values, dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.long)

    def __len__(self):
        return len(self.x_values)

    def __getitem__(self, idx):
        return self.x_values[idx], self.y_values[idx]


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# CSV 데이터 로드
def make_feature_columns():
    return [
        f"feature{feature_idx}_t{step_idx}"
        for step_idx in range(config.SEQ_LEN)
        for feature_idx in range(PER_STEP_FEATURE)
    ]


def read_sequence_csv(csv_path):
    df = pd.read_csv(csv_path)
    if any(FEATURE_COLUMN_PATTERN.match(str(col)) for col in df.columns):
        return df

    expected_with_metadata = config.SEQ_LEN * PER_STEP_FEATURE + 3
    expected_legacy = config.SEQ_LEN * PER_STEP_FEATURE + 1
    raw_df = pd.read_csv(csv_path, header=None)
    if raw_df.shape[1] == expected_with_metadata:
        raw_df.columns = ["market", "sequence_start_interval"] + make_feature_columns() + ["label"]
        return raw_df
    if raw_df.shape[1] == expected_legacy:
        raw_df.columns = make_feature_columns() + ["label"]
        return raw_df

    raise ValueError(
        "Could not find sequence feature columns. "
        f"Expected {expected_with_metadata} metadata columns or {expected_legacy} legacy columns, "
        f"got {raw_df.shape[1]} columns."
    )


def get_ordered_feature_columns(df):
    parsed_columns = []
    for col in df.columns:
        match = FEATURE_COLUMN_PATTERN.match(col)
        if match:
            feature_idx = int(match.group(1))
            step_idx = int(match.group(2))
            parsed_columns.append((step_idx, feature_idx, col))

    expected_feature_count = config.SEQ_LEN * PER_STEP_FEATURE
    if len(parsed_columns) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} feature columns, got {len(parsed_columns)}"
        )

    expected_pairs = {
        (step_idx, feature_idx)
        for step_idx in range(config.SEQ_LEN)
        for feature_idx in range(PER_STEP_FEATURE)
    }
    actual_pairs = {(step_idx, feature_idx) for step_idx, feature_idx, _ in parsed_columns}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"Invalid feature columns. missing={missing}, extra={extra}")

    return [col for _, _, col in sorted(parsed_columns)]


def load_sequence_csv(csv_path):
    df = read_sequence_csv(csv_path)
    if "sequence_start_interval" in df.columns:
        df["sequence_start_interval"] = pd.to_datetime(df["sequence_start_interval"])
        sort_cols = ["sequence_start_interval"]
        if "market" in df.columns:
            sort_cols.append("market")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    feature_cols = get_ordered_feature_columns(df)
    x_values = df[feature_cols].values
    y_values = df["label"].values
    num_samples = len(x_values)
    x_values = x_values.reshape(num_samples, config.SEQ_LEN, PER_STEP_FEATURE)
    return x_values, y_values


# train/validation split
# rolling window overlap으로 인한 검증 leakage를 줄이기 위해 split 경계에 gap을 둔다.
def time_split(x_values, y_values, val_ratio=VAL_RATIO, gap_size=SPLIT_GAP_SIZE):
    split_idx = int(len(x_values) * (1 - val_ratio))
    val_start_idx = split_idx + gap_size
    if split_idx <= 0 or val_start_idx >= len(x_values):
        raise ValueError(f"Not enough samples for train/val split: {len(x_values)}")

    return (
        x_values[:split_idx],
        x_values[val_start_idx:],
        y_values[:split_idx],
        y_values[val_start_idx:],
        gap_size,
    )


# 표준화
# scaler는 train 데이터에만 fit하고 validation/실시간 데이터에는 transform만 적용한다.
def scale_train_val(x_train_raw, x_val_raw):
    scaler = StandardScaler()
    x_train_2d = x_train_raw.reshape(-1, PER_STEP_FEATURE)
    x_val_2d = x_val_raw.reshape(-1, PER_STEP_FEATURE)

    x_train = scaler.fit_transform(x_train_2d).reshape(
        len(x_train_raw),
        config.SEQ_LEN,
        PER_STEP_FEATURE,
    )
    x_val = scaler.transform(x_val_2d).reshape(
        len(x_val_raw),
        config.SEQ_LEN,
        PER_STEP_FEATURE,
    )
    return x_train, x_val, scaler


# 평가
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            loss = criterion(out, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


# 모델 학습 1 epoch
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


# 시각화 저장
def save_training_figures(train_losses, val_losses, train_macro_f1s, val_macro_f1s, labels, preds):
    config.FIG_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(config.FIG_DIR / "train_val_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_macro_f1s, label="Train Macro F1")
    plt.plot(val_macro_f1s, label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Macro F1: Train vs Val")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.FIG_DIR / "train_val_macro_f1.png")
    plt.close()

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(config.FIG_DIR / "confusion_matrix.png")
    plt.close()


def main():
    device = get_device()
    print("Using device:", device)

    # CSV는 이미 시퀀스 형태로 저장되어 있으므로 reshape만 수행
    x_values, y_values = load_sequence_csv(config.CSV_PATH)
    print("X shape:", x_values.shape)
    print("y shape:", y_values.shape)
    print("Label distribution:", np.bincount(y_values, minlength=3))

    # 시간 순서 기반 split + gap 적용
    x_train_raw, x_val_raw, y_train, y_val, split_gap = time_split(x_values, y_values)
    x_train, x_val, scaler = scale_train_val(x_train_raw, x_val_raw)

    # scaler 저장
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    print("Train label distribution:", np.bincount(y_train, minlength=3))
    print("Val label distribution:", np.bincount(y_val, minlength=3))
    print(f"Split gap samples dropped: {split_gap}")

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 모델 학습 준비
    # 손실함수에 class weights 도입
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.divide(
        1.0,
        np.sqrt(class_counts),
        out=np.zeros_like(class_counts, dtype=float),
        where=class_counts > 0,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # CNN-LSTM 모델 / 손실함수 / optimizer 정의
    model = CNNLSTM(PER_STEP_FEATURE).to(device)
    # 완화된 class weights를 적용해 hold 편향을 줄이되 소수 클래스 loss 과증폭을 피한다.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )

    train_losses, val_losses = [], []
    train_macro_f1s, val_macro_f1s = [], []
    last_val_labels, last_val_preds = [], []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    best_val_labels, best_val_preds = [], []

    # 모델 학습
    for epoch in range(EPOCHS):
        # 1. train set으로 한 epoch 학습
        train_loss, train_macro_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # 2. validation set으로 현재 epoch 성능 평가
        val_loss, val_macro_f1, last_val_labels, last_val_preds = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # 3. loss / macro F1 로그 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_macro_f1s.append(train_macro_f1)
        val_macro_f1s.append(val_macro_f1)
        val_pred_distribution = np.bincount(last_val_preds, minlength=3)

        # 4. 초반 spike를 피하기 위해 BEST_START_EPOCH 이후부터 best model 갱신
        if epoch + 1 >= BEST_START_EPOCH and val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_labels = list(last_val_labels)
            best_val_preds = list(last_val_preds)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Macro F1: {train_macro_f1:.4f} | Val Macro F1: {val_macro_f1:.4f} | "
            f"Best Val Macro F1: {best_val_macro_f1:.4f} @ Epoch {best_epoch} | "
            f"Val Pred Distribution: {val_pred_distribution.tolist()}"
        )

        # 5. learning rate scheduler 업데이트
        scheduler.step()

    # loss / macro F1 / confusion matrix 저장
    save_training_figures(
        train_losses,
        val_losses,
        train_macro_f1s,
        val_macro_f1s,
        best_val_labels,
        best_val_preds,
    )

    # 모델 저장
    torch.save(best_state_dict, config.MODEL_PATH)
    print(
        f"Best model saved to {config.MODEL_PATH} "
        f"(epoch={best_epoch}, val_macro_f1={best_val_macro_f1:.4f})"
    )


if __name__ == "__main__":
    main()

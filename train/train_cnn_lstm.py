import os
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


def load_sequence_csv(csv_path):
    df = pd.read_csv(csv_path)
    x_values = df.drop(columns=["label"]).values
    y_values = df["label"].values
    num_samples = len(x_values)
    x_values = x_values.reshape(num_samples, config.SEQ_LEN, PER_STEP_FEATURE)
    return x_values, y_values


def time_split(x_values, y_values, val_ratio=0.2):
    split_idx = int(len(x_values) * (1 - val_ratio))
    if split_idx <= 0 or split_idx >= len(x_values):
        raise ValueError(f"Not enough samples for train/val split: {len(x_values)}")

    return (
        x_values[:split_idx],
        x_values[split_idx:],
        y_values[:split_idx],
        y_values[split_idx:],
    )


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


def make_class_weights(y_train, device):
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.divide(
        1.0,
        class_counts,
        out=np.zeros_like(class_counts, dtype=float),
        where=class_counts > 0,
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


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

    x_values, y_values = load_sequence_csv(config.CSV_PATH)
    print("X shape:", x_values.shape)
    print("y shape:", y_values.shape)
    print("Label distribution:", np.bincount(y_values, minlength=3))

    x_train_raw, x_val_raw, y_train, y_val = time_split(x_values, y_values)
    x_train, x_val, scaler = scale_train_val(x_train_raw, x_val_raw)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    print("Train label distribution:", np.bincount(y_train, minlength=3))
    print("Val label distribution:", np.bincount(y_val, minlength=3))

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(x_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    model = CNNLSTM(PER_STEP_FEATURE).to(device)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(y_train, device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses = [], []
    train_macro_f1s, val_macro_f1s = [], []
    last_val_labels, last_val_preds = [], []

    for epoch in range(config.EPOCHS):
        train_loss, train_macro_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_macro_f1, last_val_labels, last_val_preds = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_macro_f1s.append(train_macro_f1)
        val_macro_f1s.append(val_macro_f1)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Macro F1: {train_macro_f1:.4f} | Val Macro F1: {val_macro_f1:.4f}"
        )
        scheduler.step()

    save_training_figures(
        train_losses,
        val_losses,
        train_macro_f1s,
        val_macro_f1s,
        last_val_labels,
        last_val_preds,
    )
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

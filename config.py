from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "fig"

INTERVAL_SEC = 30
SEQ_LEN = 10
SEQ_LEN_SEC = INTERVAL_SEC * SEQ_LEN
LABEL_THRESHOLD = 0.008
SELECT_TOP_N = 5
MAX_SEQUENCES = 2000

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

CSV_PATH = DATASET_DIR / "sequence_dataset.csv"
MODEL_PATH = MODELS_DIR / "cnn_lstm_model.pth"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

from pathlib import Path


# 프로젝트 기준 경로
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "fig"

# 데이터 수집 / 시퀀스 생성 설정
# INTERVAL_SEC: WebSocket tick/orderbook을 몇 초 단위 봉으로 묶을지 결정
INTERVAL_SEC = 30

# SEQ_LEN: 모델 입력에 사용할 과거 30초봉 개수
SEQ_LEN = 10
SEQ_LEN_SEC = INTERVAL_SEC * SEQ_LEN

# LABEL_THRESHOLD: 다음 30초 수익률이 이 값 이상/이하일 때 buy/sell 라벨 부여
LABEL_THRESHOLD = 0.008

# 산출물 경로
CSV_PATH = DATASET_DIR / "sequence_dataset.csv"
MODEL_PATH = MODELS_DIR / "cnn_lstm_model.pth"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

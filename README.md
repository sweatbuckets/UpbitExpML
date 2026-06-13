# UpbitExp

UpbitExp는 업비트 실시간 체결/호가 데이터를 수집해 30초 단위 시계열 feature를 만들고, CNN + LSTM 모델로 다음 30초의 `sell / hold / buy` 액션을 예측하는 머신러닝 프로젝트입니다.

종목은 KRW 마켓 전체 중 `abs(signed_change_rate)`가 큰 변동성 상위 종목을 기준으로 선택합니다. 데이터셋 생성 단계에서는 변동성 상위 5개 종목을 동시에 수집하고, 실시간 추론/검증 단계에서는 현재 변동성이 가장 큰 1개 종목을 대상으로 예측합니다.

기획의 중심은 단순 가격 예측이 아니라, 편차를 메인으로 하는 CUSUM 기반 데이터를 입력 feature로 구성하는 것입니다. 체결 가격의 기울기, 가속도, 직전 수익률, CUSUM 양/음 누적 편차, 거래량 비율, 호가 불균형, 스프레드 비율을 30초 단위로 계산하고 최근 10개 step, 즉 5분 구간을 하나의 입력 시퀀스로 사용합니다.

## 주요 수행

- 기획: 편차를 메인으로 하는 종래의 CUSUM 기반 데이터를 입력 feature로 구성
- 데이터 수집 및 처리: Upbit WebSocket으로 체결/호가 데이터를 수집하고, 30초봉 가공 및 feature 계산이 포함된 시퀀스 데이터 수집 프로세스 개발
- 머신러닝: CNN + LSTM 모델 개발 및 학습, `StandardScaler` 기반 feature scaler 도입
- 검증: 30초 전 예측과 실제 다음 30초 결과를 대조하는 실시간 검증 구현, confusion matrix로 클래스별 성능 모니터링
- 성과: 실시간 검증에서 F1 score 0.48 기록, 실시간 데이터 수집 후 예측까지 이어지는 파이프라인 구축 성공

## 성능 요약

실시간 검증에서 macro F1 score 0.48을 기록했습니다. 학습 단계에서는 train/validation loss, train/validation macro F1, confusion matrix를 저장해 모델이 특정 클래스에 치우치는지 확인합니다.

### Train / Validation Loss

![Train vs Validation Loss](fig/train_val_loss.png)

### Train / Validation Macro F1

![Train vs Validation Macro F1](fig/train_val_macro_f1.png)

### Confusion Matrix

![Confusion Matrix](fig/confusion_matrix.png)

## 예측 방식

입력 데이터는 최근 5분입니다.

- interval: 30초
- sequence length: 10개 step
- feature 수: step당 8개
- 모델 입력 shape: `(batch, 10, 8)`

사용 feature:

- `slope`: 직전 30초봉 대비 가격 변화율
- `accel`: 변화율의 변화량
- `last_return`: 직전 30초 수익률
- `cusum_pos`: 양의 수익률 누적 편차
- `cusum_neg`: 음의 수익률 누적 편차
- `volume_ratio`: 이전 거래량 평균 대비 현재 거래량 비율
- `bid_ask_imbalance`: 매수/매도 호가 잔량 불균형
- `spread_ratio`: 평균 호가 기준 스프레드 비율

라벨은 시퀀스 직후 다음 30초 수익률로 만듭니다.

- `buy`: 다음 30초 수익률 `>= +0.8%`
- `sell`: 다음 30초 수익률 `<= -0.8%`
- `hold`: 그 외 구간

## 파이프라인

1. KRW 마켓에서 변동성 상위 종목 선택
2. Upbit WebSocket으로 체결/호가 데이터 수집
3. 닫힌 30초 interval만 30초봉으로 집계
4. OHLCV와 orderbook 데이터를 결합
5. CUSUM 기반 feature와 호가 feature 계산
6. 최근 10개 feature row를 하나의 시퀀스로 구성
7. 다음 30초 수익률 기준으로 `sell / hold / buy` 라벨 생성
8. CNN + LSTM 모델 학습
9. 저장된 모델과 scaler로 실시간 예측
10. 실시간 검증 스크립트에서 이전 예측과 실제 결과 비교

## 파일 역할

### 실행 스크립트

- `ml_dataset_creator.py`  
  실시간 수집 데이터를 바탕으로 학습용 시퀀스 CSV를 생성합니다. 변동성 상위 5개 종목을 구독하고, 30초봉 집계 및 feature 계산 후 `dataset/sequence_dataset.csv`에 누적 저장합니다.

- `train/train_cnn_lstm.py`  
  `dataset/sequence_dataset.csv`를 읽어 CNN-LSTM 모델을 학습합니다. 시간 순서 기반 train/validation split을 사용하고, scaler는 train 데이터에만 fit합니다. 학습된 모델, scaler, 성능 그래프를 저장합니다.

- `realtime_action_infer.py`  
  현재 변동성이 가장 큰 1개 종목을 선택해 실시간 `sell / hold / buy` 예측을 수행합니다. 최근 10개 feature row가 쌓이기 전에는 warm-up 상태로 대기합니다.

- `realtime_action_check.py`  
  실시간 예측 결과를 다음 30초 실제 수익률과 비교합니다. 30초 전 예측과 현재 실제 라벨을 대조해 accuracy와 macro F1을 로그로 출력합니다.

- `upbit_data_collector.py`  
  변동성 종목 WebSocket 수집을 단독으로 확인하기 위한 보조 유틸입니다.

### 공통 모듈

- `config.py`  
  interval, sequence length, threshold, batch size, epoch, 파일 경로 등 프로젝트 설정을 관리합니다.

- `feature_engineering.py`  
  체결 데이터 30초봉 집계, 호가 데이터 집계, 닫힌 interval 분리, 공통 feature 계산을 담당합니다.

- `market_selector.py`  
  Upbit REST API로 KRW 마켓과 ticker를 조회하고, `abs(signed_change_rate)` 기준 변동성 상위 종목을 선택합니다.

- `ws_collector.py`  
  Upbit WebSocket을 통해 체결 데이터와 orderbook 데이터를 수집합니다. 단일 종목과 다중 종목 모두 같은 collector를 사용합니다.

- `model.py`  
  CNN + LSTM 분류 모델을 정의합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행 방법

### 1. 데이터 수집 및 시퀀스 생성

```bash
python ml_dataset_creator.py
```

생성 파일:

- `dataset/sequence_dataset.csv`

동작:

- 변동성 상위 5개 KRW 종목 선택
- 체결/호가 WebSocket 구독
- 닫힌 30초봉만 집계
- feature 계산
- 최근 10개 feature row + 다음 30초 라벨로 학습 시퀀스 생성

### 2. 모델 학습

```bash
python train/train_cnn_lstm.py
```

입력:

- `dataset/sequence_dataset.csv`

생성 파일:

- `models/cnn_lstm_model.pth`
- `models/feature_scaler.pkl`
- `fig/train_val_loss.png`
- `fig/train_val_macro_f1.png`
- `fig/confusion_matrix.png`

동작:

- CSV를 `(N, 10, 8)` shape으로 변환
- 시간 순서 기준 train/validation split
- train 데이터 기준 scaler fit
- CNN-LSTM 학습
- validation macro F1 및 confusion matrix 저장

### 3. 실시간 예측

```bash
python realtime_action_infer.py
```

필요 파일:

- `models/cnn_lstm_model.pth`
- `models/feature_scaler.pkl`

동작:

- 현재 변동성이 가장 큰 1개 KRW 종목 선택
- 30초봉 feature를 실시간 생성
- feature row 10개가 쌓이면 예측 시작
- 로그로 `sell / hold / buy` 출력

### 4. 실시간 검증

```bash
python realtime_action_check.py
```

필요 파일:

- `models/cnn_lstm_model.pth`
- `models/feature_scaler.pkl`

동작:

- 실시간 예측 수행
- 다음 30초 실제 수익률로 true label 생성
- 30초 전 예측과 현재 실제 라벨 비교
- accuracy와 macro F1 출력

## 산출물

- `dataset/sequence_dataset.csv`: 학습용 시퀀스 데이터셋
- `models/cnn_lstm_model.pth`: 학습된 CNN-LSTM 모델 가중치
- `models/feature_scaler.pkl`: 실시간 추론에서 재사용할 feature scaler
- `fig/train_val_loss.png`: train/validation loss 그래프
- `fig/train_val_macro_f1.png`: train/validation macro F1 그래프
- `fig/confusion_matrix.png`: 클래스별 confusion matrix

## 주의 사항

- 현재 프로젝트는 신호 생성과 검증 중심이며, 실제 주문 API 호출은 포함하지 않습니다.
- 실시간 예측은 모델과 scaler가 먼저 생성되어 있어야 실행할 수 있습니다.
- `dataset/`, `models/`, `fig/`는 실행 산출물입니다.
- 수집 기준 종목은 실행 시점의 `abs(signed_change_rate)` 기준이므로 실행할 때마다 달라질 수 있습니다.
- 실시간 예측은 10개 feature row가 쌓인 뒤 시작합니다. feature 계산에서 초기 2개 interval은 제외되므로 첫 예측까지는 약 12개 닫힌 30초봉이 필요합니다.

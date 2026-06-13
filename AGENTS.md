# Repository Guidelines for AI Agents

이 파일은 Codex 같은 AI 개발 에이전트가 이 저장소에서 작업할 때 참고하는 운영 지침입니다. 사용자용 프로젝트 소개와 실행 설명은 `README.md`를 기준으로 하고, 이 파일은 코드 수정 시 지켜야 할 구조, 검증, 커밋 규칙을 정의합니다.

## 프로젝트 개요

이 저장소는 Upbit 실시간 체결/호가 데이터를 수집해 30초 단위 feature를 만들고, CNN + LSTM 모델로 다음 30초의 `sell / hold / buy` 액션을 예측하는 Python 기반 ML 프로젝트입니다.

종목 선택 기준은 KRW 마켓 중 `abs(signed_change_rate)`가 큰 변동성 상위 종목입니다. 데이터셋 생성은 상위 5개 종목, 실시간 예측/검증은 상위 1개 종목을 대상으로 합니다.

## 파일 책임

- `config.py`: interval, sequence length, label threshold, CSV/model/scaler/fig 경로 등 공통 계약 설정
- `feature_engineering.py`: 닫힌 30초 interval 분리, 체결/호가 집계, OHLCV/orderbook interval 매칭, feature 계산
- `upbit_client.py`: Upbit REST API 조회, 변동성 상위 종목 선택, WebSocket 체결/호가 수집
- `model.py`: CNN-LSTM 모델 정의
- `ml_dataset_creator.py`: 실시간 수집 데이터를 학습용 시퀀스 CSV로 저장
- `train/train_cnn_lstm.py`: CSV 로드, time split, scaler fit, 모델 학습, 성능 그래프 저장
- `realtime_action_infer.py`: 저장된 모델/스케일러로 실시간 액션 예측
- `realtime_action_check.py`: 30초 전 예측과 실제 다음 30초 라벨 비교
- `upbit_data_collector.py`: 변동성 종목 수집 보조 유틸

## 개발 명령

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ml_dataset_creator.py
python train/train_cnn_lstm.py
python realtime_action_infer.py
python realtime_action_check.py
```

## 수정 원칙

- 학습/추론 간 feature 순서와 의미는 반드시 `feature_engineering.FEATURE_COLS`를 기준으로 맞춥니다.
- 모델 구조는 `model.CNNLSTM` 한 곳에서만 정의합니다.
- Upbit REST 조회, 종목 선택, WebSocket 수집 구현은 `upbit_client.py`에만 둡니다.
- 30초봉 집계와 feature 계산은 `feature_engineering.py`에만 둡니다.
- `config.py`에는 공통 입력/출력 계약만 두고, 학습 하이퍼파라미터와 수집 정책값은 각 실행 스크립트에서 관리합니다.
- OHLCV와 orderbook은 interval 기준 `inner join`으로 결합하며, orderbook 결측 interval을 0으로 채워 학습시키지 않습니다.
- 실행 스크립트에는 orchestration 로직만 남기고, 공통 계산 로직을 중복 작성하지 않습니다.
- 실시간 예측은 feature row 10개 warm-up 이후 시작해야 합니다.
- scaler는 train 데이터에만 fit하고 validation/test/실시간 데이터에는 transform만 적용합니다.

## 검증 가이드

정식 `tests/` 스위트는 아직 없습니다. 변경 후 최소한 아래를 확인하세요.

```bash
python3 -m py_compile config.py feature_engineering.py model.py upbit_client.py ml_dataset_creator.py train/train_cnn_lstm.py realtime_action_infer.py realtime_action_check.py upbit_data_collector.py
python3 -c "import config, feature_engineering, model, upbit_client, ml_dataset_creator, realtime_action_infer, realtime_action_check, upbit_data_collector; import train.train_cnn_lstm as t; x,y=t.load_sequence_csv(t.config.CSV_PATH); print(x.shape, y.shape)"
```

모델 관련 변경 시 추가로 확인할 항목:

- CSV reshape 결과가 `(N, 10, 8)`인지
- label 분포가 극단적으로 깨지지 않았는지
- train/validation macro F1 추이
- `models/cnn_lstm_model.pth`, `models/feature_scaler.pkl` 저장 여부
- `fig/confusion_matrix.png`로 클래스별 예측 편향 여부

## 코딩 스타일

- Python PEP 8 기준, 들여쓰기는 스페이스 4칸을 사용합니다.
- 함수/변수는 `snake_case`, 상수는 `UPPER_SNAKE_CASE`를 사용합니다.
- 반복 실행 루프에서는 `print`보다 `logging.info`를 우선합니다.
- 생성 산출물인 `dataset/`, `models/`, `fig/`는 필요할 때만 의도적으로 버전 관리합니다.

## 커밋 및 PR

Conventional Commit을 사용합니다.

- `feat: ...` 기능 추가
- `fix: ...` 버그 수정
- `docs: ...` 문서 수정
- `refactor: ...` 동작 변경 없는 구조 개선

PR에는 변경 목적, 영향 파일, 실행/검증 명령, 모델/추론 변경 시 전후 지표 또는 로그를 포함하세요.

## 보안

- API 키, 비밀값, 인증 정보는 커밋하지 마세요.
- 실제 주문 API 호출은 현재 프로젝트 범위에 포함되어 있지 않습니다.

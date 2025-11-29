# 바이낸스 선물 비트코인 5분봉 가격 예측 시스템

바이낸스 선물 거래소의 비트코인 5분봉 가격을 예측하는 딥러닝 시스템입니다.

## 주요 기능

- **데이터 수집**: 바이낸스 선물 거래소에서 실시간 5분봉 데이터 수집
- **기술적 지표**: RSI, CCI, 볼린저밴드 및 다양한 파생변수 계산
- **딥러닝 모델**: Patch CNN-BiLSTM 구조를 사용한 시계열 예측
- **데이터 누수 방지**: 미래 데이터 컨닝을 방지하는 안전한 전처리
- **평가 시스템**: 예측 결과와 실제 가격을 비교하는 상세한 평가 지표

## 프로젝트 구조

```
python_project/
├── data_fetcher.py          # 바이낸스 API 데이터 수집
├── feature_engineering.py    # 기술적 지표 및 파생변수 계산
├── data_preprocessor.py     # 데이터 전처리 및 슬라이딩 윈도우
├── model.py                 # Patch CNN-BiLSTM 모델 정의
├── trainer.py               # 모델 학습
├── predictor.py             # 예측 수행
├── evaluator.py             # 예측 결과 평가
├── main.py                  # 메인 실행 파일
└── requirements.txt         # 의존성 패키지
```

## 설치 및 실행

### 1. 가상환경 활성화

```bash
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 실행

#### 모델 학습

```bash
python main.py
```

#### 실시간 시그널 생성 (거래 없음)

```bash
python realtime_trading.py
```

#### 실시간 자동 거래 (실거래)

**⚠️ 주의: 실제 자금이 사용됩니다!**

1. 바이낸스 API 키 설정 (환경변수 또는 코드에서 직접 설정)

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

2. 실거래 실행

```bash
python realtime_trading.py --trade
```

**거래 조건:**
- 3분봉 변화율 >= 0.10%
- 5분봉 변화율 >= 0.15%
- 15분봉 변화율 >= 0.15%
- 모두 같은 부호 (상승 예측일 때만 매수)
- 조건 충족 시 100% 자금 사용, 30배 레버리지, 시장가 거래
- Take Profit: ROI 7% 달성 시 포지션 자동 종료
- 5분마다 예측 및 계좌 점검

**옵션:**
- `--trade`: 실거래 모드 활성화
- `--leverage N`: 레버리지 배수 설정 (기본값: 30)
- `--take-profit N`: Take Profit ROI 설정 (기본값: 0.07 = 7%)
- `--interval N`: 실행 간격 (분, 기본값: 5)

## 모델 구조

- **Patch CNN**: 지역적 패턴 추출을 위한 1D Convolution
- **BiLSTM**: 양방향 LSTM으로 시계열 패턴 학습
- **슬라이딩 윈도우**: 60개 시점(5시간)의 과거 데이터로 다음 5분봉 예측

## 특징 (Features)

- OHLCV 데이터 (Open, High, Low, Close, Volume)
- RSI (7, 14, 21 기간)
- CCI (14, 20 기간)
- 볼린저밴드 (10, 20 기간)
- 가격 변화율, 모멘텀, 변동성 등 파생변수

## 결과

실행 후 다음 디렉토리에 결과가 저장됩니다:

- `models/`: 학습된 모델 파일
- `results/`: 평가 결과 및 시각화 그래프
  - `test_comparison.csv`: 테스트 데이터 예측 vs 실제 비교
  - `test_metrics.txt`: 평가 지표
  - `test_comparison.png`: 시각화 그래프

## 가상환경 비활성화

```bash
deactivate
```


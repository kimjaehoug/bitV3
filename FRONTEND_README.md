# 비트코인 실시간 거래 대시보드

React 기반 실시간 비트코인 거래 대시보드 프론트엔드

## 프로젝트 구조

```
bitV3/
├── api_server.py          # Flask 백엔드 API 서버
├── frontend/              # React 프론트엔드
│   ├── src/
│   │   ├── App.js        # 메인 앱 컴포넌트
│   │   ├── components/
│   │   │   ├── TradingChart.js        # 차트 컴포넌트
│   │   │   ├── PredictionPanel.js     # 예측 정보 패널
│   │   │   ├── PositionPanel.js       # 포지션 정보 패널
│   │   │   └── TechnicalIndicators.js # 기술적 지표 패널
│   │   └── ...
│   └── package.json
└── run_api_server.sh     # API 서버 실행 스크립트
```

## 설치 및 실행

### 1. 백엔드 API 서버 실행

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# API 서버 실행
python api_server.py
# 또는
./run_api_server.sh
```

서버는 `http://localhost:5333`에서 실행됩니다.

### 2. 프론트엔드 실행

```bash
# frontend 디렉토리로 이동
cd frontend

# 패키지 설치
npm install

# 개발 서버 실행
npm start
```

브라우저에서 `http://localhost:3000`으로 접속합니다.

## 주요 기능

### 1. 실시간 가격 차트
- TradingView Lightweight Charts 사용
- 캔들스틱 차트
- 실시간 가격 업데이트 (1분마다)

### 2. 기술적 지표
- 이동평균선 (MA5, MA20, MA50)
- RSI (상대강도지수)
- 볼린저 밴드
- 골든크로스/데드크로스 표시

### 3. 지지선/저항선
- 자동 계산 및 표시
- 최근 20개 캔들 기반 계산

### 4. 모델 예측 정보
- 30분 후 예측 가격 및 변화율
- 1시간 후 예측 가격 및 변화율
- 거래 신호 (롱/숏/대기)
- 신뢰도 표시

### 5. 포지션 정보
- 현재 포지션 상태 (롱/숏)
- 진입 가격
- 미실현 손익
- 차트에 진입 위치 마커 표시

## API 엔드포인트

### REST API
- `GET /api/status` - 서버 상태 확인
- `POST /api/init` - 시스템 초기화
- `POST /api/start` - 데이터 업데이트 시작
- `POST /api/stop` - 데이터 업데이트 중지
- `GET /api/history/price` - 가격 히스토리 조회
- `GET /api/history/prediction` - 예측 히스토리 조회
- `GET /api/history/position` - 포지션 히스토리 조회
- `GET /api/current` - 현재 데이터 조회
- `POST /api/gemini/analyze` - Gemini AI를 통한 시장 분석 (관망/롱/숏 유의점 제공)

### WebSocket
- `price_update` - 실시간 가격 업데이트 이벤트

## 환경 변수

### 프론트엔드 `.env` 파일 (선택사항):

```
REACT_APP_API_URL=http://localhost:5333
REACT_APP_SOCKET_URL=http://localhost:5333
```

### 백엔드 환경 변수 (Gemini API 사용 시 필수):

백엔드 API 서버 실행 전에 환경변수 설정:

```bash
# Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key_here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

또는 `.env` 파일 생성 (프로젝트 루트):

```
GEMINI_API_KEY=your_gemini_api_key_here
```

**Gemini API 키 발급 방법:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급
2. 환경변수로 설정하거나 `.env` 파일에 추가

## 기술 스택

### 백엔드
- Flask
- Flask-SocketIO
- Python 3.10+

### 프론트엔드
- React 18
- TradingView Lightweight Charts
- Socket.IO Client
- Axios

## 문제 해결

### API 서버가 시작되지 않는 경우
- 포트 5333이 이미 사용 중인지 확인
- 필요한 Python 패키지가 모두 설치되었는지 확인

### 프론트엔드가 API에 연결되지 않는 경우
- API 서버가 실행 중인지 확인
- CORS 설정 확인
- `.env` 파일의 URL 확인

### 차트가 표시되지 않는 경우
- 브라우저 콘솔에서 오류 확인
- WebSocket 연결 상태 확인
- 데이터가 정상적으로 수신되는지 확인


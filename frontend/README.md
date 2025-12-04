# 비트코인 실시간 거래 대시보드

React 기반 실시간 비트코인 거래 대시보드

## 설치

```bash
npm install
```

## 실행

```bash
npm start
```

브라우저에서 http://localhost:3000 으로 접속

## 환경 변수

`.env` 파일 생성:

```
REACT_APP_API_URL=http://localhost:5333
REACT_APP_SOCKET_URL=http://localhost:5333
REACT_APP_GEMINI_API_KEY=your_gemini_api_key_here
```

### Gemini API 키 설정

1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급
2. `.env` 파일에 `REACT_APP_GEMINI_API_KEY` 환경변수로 추가
3. 프론트엔드 재시작

## 기능

- 실시간 가격 차트 (캔들스틱)
- 모델 예측 정보 표시
- 기술적 지표 (MA, RSI, 볼린저 밴드)
- 지지선/저항선 자동 계산
- 골든크로스/데드크로스 표시
- 포지션 정보 표시
- **Gemini AI 분석**: 관망/롱/숏 포지션 유의점 제공


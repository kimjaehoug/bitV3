import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import TradingChart from './components/TradingChart';
import PredictionPanel from './components/PredictionPanel';
import PositionPanel from './components/PositionPanel';
import TechnicalIndicators from './components/TechnicalIndicators';
import TradingPanel from './components/TradingPanel';
import MarketIndicatorsPanel from './components/MarketIndicatorsPanel';
import ChartControls from './components/ChartControls';
import AlertBanner from './components/AlertBanner';
import GeminiAnalysisPanel from './components/GeminiAnalysisPanel';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5333';
const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:5333';

function App() {
  const [connected, setConnected] = useState(false);
  const [priceData, setPriceData] = useState([]);
  const [predictionData, setPredictionData] = useState(null);
  const [thresholdInfo, setThresholdInfo] = useState(null);
  const [positionData, setPositionData] = useState(null);
  const [technicalIndicators, setTechnicalIndicators] = useState({});
  const [supportResistance, setSupportResistance] = useState({});
  const [fibonacci, setFibonacci] = useState({});
  const [trendLines, setTrendLines] = useState({});
  const [marketIndicators, setMarketIndicators] = useState(null);
  const [socket, setSocket] = useState(null);
  const [systemInitialized, setSystemInitialized] = useState(false);
  const [chartVisibility, setChartVisibility] = useState({
    ma5: true,
    ma20: true,
    ma50: true,
    support: true,
    resistance: true,
    uptrend: true,
    downtrend: true,
    fibonacci: true,
    bollinger: true  // 기본값을 true로 변경
  });

  // 시스템 초기화
  useEffect(() => {
    const initSystem = async () => {
      try {
        console.log('시스템 초기화 시작...');
        const response = await axios.post(`${API_BASE_URL}/api/init`, {
          model_path: 'models/best_model.h5',
          leverage: 10,
          dry_run: true,
          enable_trading: false
        });
        
        if (response.data.success) {
          setSystemInitialized(true);
          console.log('시스템 초기화 완료');
        } else {
          console.error('시스템 초기화 실패:', response.data);
          // 초기화 실패해도 계속 진행 (자동 초기화 시도)
          setSystemInitialized(true);
        }
      } catch (error) {
        console.error('시스템 초기화 오류:', error);
        // 초기화 실패해도 계속 진행 (자동 초기화 시도)
        setSystemInitialized(true);
      }
    };

    initSystem();
  }, []);

  // WebSocket 연결
  useEffect(() => {
    if (!systemInitialized) return;

    const newSocket = io(SOCKET_URL);
    
    newSocket.on('connect', () => {
      console.log('WebSocket 연결됨');
      setConnected(true);
      
      // 데이터 업데이트 시작 (약간의 지연을 두어 초기화 완료 대기)
      setTimeout(() => {
        axios.post(`${API_BASE_URL}/api/start`)
          .then(res => {
            console.log('데이터 업데이트 시작:', res.data);
            if (!res.data.success) {
              console.warn('데이터 업데이트 시작 실패:', res.data.message);
            }
          })
          .catch(err => {
            console.error('데이터 업데이트 시작 오류:', err);
            if (err.response) {
              console.error('응답 데이터:', err.response.data);
            }
          });
      }, 1000); // 1초 대기
    });

    newSocket.on('disconnect', () => {
      console.log('WebSocket 연결 해제됨');
      setConnected(false);
    });

    newSocket.on('price_update', (data) => {
        // 최근 24시간간의 OHLCV 데이터 업데이트
        if (data.ohlcv_data && Array.isArray(data.ohlcv_data)) {
          const ohlcvData = data.ohlcv_data.map(item => ({
            time: new Date(item.timestamp).getTime() / 1000,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.price,
            volume: item.volume
          }));
          // 최근 24시간간의 데이터 유지 (약 288개 캔들)
          setPriceData(ohlcvData);
        } else {
          // 이전 형식 호환성 (ohlcv_data가 없는 경우)
          setPriceData(prev => {
            const newData = [...prev, {
              time: new Date(data.timestamp).getTime() / 1000,
              open: data.open || data.current_price,
              high: data.high || data.current_price,
              low: data.low || data.current_price,
              close: data.current_price || data.price,
              volume: data.volume || 0
            }];
            // 최근 24시간간만 유지 (약 288개 캔들)
            return newData.slice(-288);
          });
        }

      // 예측 데이터 업데이트
        if (data.prediction) {
          setPredictionData(data.prediction);
        }
        
        // 임계값 정보 업데이트
        if (data.threshold_info) {
          setThresholdInfo(data.threshold_info);
        }

      // 포지션 데이터 업데이트
      if (data.position) {
        setPositionData(data.position);
      }

      // 기술적 지표 업데이트
      if (data.technical_indicators) {
        setTechnicalIndicators(data.technical_indicators);
      }

      // 지지선/저항선 업데이트
      if (data.support_resistance) {
        setSupportResistance(data.support_resistance);
      }

      // 피보나치 되돌림 업데이트
      if (data.fibonacci) {
        setFibonacci(data.fibonacci);
      }

      // 추세선 업데이트
      if (data.trend_lines) {
        setTrendLines(data.trend_lines);
      }

      // 시장 지표 업데이트
      if (data.market_indicators) {
        setMarketIndicators(data.market_indicators);
      }
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
      axios.post(`${API_BASE_URL}/api/stop`)
        .catch(err => console.error('데이터 업데이트 중지 실패:', err));
    };
  }, [systemInitialized]);

  // 초기 히스토리 데이터 로드
  useEffect(() => {
    if (!systemInitialized) return;

    const loadHistory = async () => {
      try {
        const [priceRes, predictionRes, positionRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/history/price?limit=288`), // 최근 24시간 (288개 캔들)
          axios.get(`${API_BASE_URL}/api/history/prediction?limit=200`),
          axios.get(`${API_BASE_URL}/api/history/position`)
        ]);

        // 가격 데이터 변환 (최근 24시간간)
        const priceHistory = priceRes.data.slice(-288).map(item => ({
          time: new Date(item.timestamp).getTime() / 1000,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.price,
          volume: item.volume
        }));
        setPriceData(priceHistory);

        // 예측 데이터
        if (predictionRes.data.length > 0) {
          setPredictionData(predictionRes.data[predictionRes.data.length - 1]);
        }

        // 포지션 데이터
        if (positionRes.data.length > 0) {
          setPositionData(positionRes.data[positionRes.data.length - 1]);
        }
      } catch (error) {
        console.error('히스토리 데이터 로드 실패:', error);
      }
    };

    loadHistory();
  }, [systemInitialized]);

  return (
    <div className="App">
      <AlertBanner technicalIndicators={technicalIndicators} />
      <header className="App-header">
        <h1>비트코인 실시간 거래 대시보드</h1>
        <div className="status-indicator">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
          <span>{connected ? '연결됨' : '연결 안 됨'}</span>
        </div>
      </header>

      <div className="dashboard-container">
        <div className="main-chart-section">
          <TradingChart
            priceData={priceData}
            technicalIndicators={technicalIndicators}
            supportResistance={supportResistance}
            positionData={positionData}
            fibonacci={fibonacci}
            trendLines={trendLines}
            marketIndicators={marketIndicators}
            visibility={chartVisibility}
          />
          <div className="indicators-row">
            <MarketIndicatorsPanel marketIndicators={marketIndicators} />
            <TechnicalIndicators indicators={technicalIndicators} />
          </div>
          <GeminiAnalysisPanel
            priceData={priceData}
            predictionData={predictionData}
            technicalIndicators={technicalIndicators}
            supportResistance={supportResistance}
            trendLines={trendLines}
            marketIndicators={marketIndicators}
            fibonacci={fibonacci}
            socket={socket}
          />
        </div>

        <div className="sidebar-section">
          <TradingPanel onUpdate={() => {
            // 포지션 정보 갱신을 위해 소켓 재연결 트리거
            if (socket) {
              socket.emit('refresh');
            }
          }} />
          <ChartControls 
            visibility={chartVisibility}
            onToggle={(key) => {
              setChartVisibility(prev => ({
                ...prev,
                [key]: !prev[key]
              }));
            }}
          />
          <PredictionPanel predictionData={predictionData} thresholdInfo={thresholdInfo} />
          <PositionPanel positionData={positionData} />
        </div>
      </div>
    </div>
  );
}

export default App;


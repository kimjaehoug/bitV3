import React, { useState } from 'react';
import axios from 'axios';
import './TradingPanel.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5333';

const TradingPanel = ({ onUpdate }) => {
  const [balance, setBalance] = useState(null);
  const [position, setPosition] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [leverage, setLeverage] = useState(30);
  const [leverageInput, setLeverageInput] = useState(30);
  const [takeProfitRoi, setTakeProfitRoi] = useState(40);
  const [takeProfitInput, setTakeProfitInput] = useState(40);
  const [stopLossRoi, setStopLossRoi] = useState(5);
  const [stopLossInput, setStopLossInput] = useState(5);
  const [tradingMode, setTradingMode] = useState('normal');
  const [tradingEnabled, setTradingEnabled] = useState(false);

  const [tradingStatus, setTradingStatus] = useState(null);

  const fetchBalance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/balance`);
      if (response.data.success) {
        setBalance(response.data.balance);
        if (response.data.trading_status) {
          setTradingStatus(response.data.trading_status);
        }
      }
    } catch (error) {
      console.error('잔액 조회 실패:', error);
      setMessage('잔액 조회 실패');
    }
  };

  const fetchPosition = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/position`);
      if (response.data.success) {
        setPosition(response.data.position);
      }
    } catch (error) {
      console.error('포지션 조회 실패:', error);
      setMessage('포지션 조회 실패');
    }
  };

  const fetchLeverage = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/leverage`);
      if (response.data.success) {
        setLeverage(response.data.leverage);
        setLeverageInput(response.data.leverage);
      }
    } catch (error) {
      console.error('레버리지 조회 실패:', error);
    }
  };

  const fetchRoiSl = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/roi-sl`);
      if (response.data.success) {
        setTakeProfitRoi(response.data.take_profit_roi * 100);
        setTakeProfitInput(response.data.take_profit_roi * 100);
        setStopLossRoi(response.data.stop_loss_roi * 100);
        setStopLossInput(response.data.stop_loss_roi * 100);
      }
    } catch (error) {
      console.error('ROI/SL 조회 실패:', error);
    }
  };

  const fetchTradingMode = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/mode`);
      if (response.data.success) {
        setTradingMode(response.data.mode);
      }
    } catch (error) {
      console.error('투자 모드 조회 실패:', error);
    }
  };

  const fetchTradingStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/status`);
      if (response.data.success) {
        setTradingEnabled(response.data.trading_enabled);
      }
    } catch (error) {
      console.error('거래 사이클 상태 조회 실패:', error);
    }
  };

  const toggleTradingCycle = async () => {
    setLoading(true);
    setMessage('');
    try {
      if (tradingEnabled) {
        // 비활성화
        const response = await axios.post(`${API_BASE_URL}/api/trading/disable`);
        if (response.data.success) {
          setTradingEnabled(false);
          setMessage('거래 사이클이 비활성화되었습니다.');
        } else {
          setMessage(response.data.error || '거래 사이클 비활성화 실패');
        }
      } else {
        // 활성화
        const response = await axios.post(`${API_BASE_URL}/api/trading/enable`);
        if (response.data.success) {
          setTradingEnabled(true);
          setMessage('거래 사이클이 활성화되었습니다.');
        } else {
          setMessage(response.data.error || '거래 사이클 활성화 실패');
        }
      }
    } catch (error) {
      console.error('거래 사이클 토글 실패:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || '거래 사이클 토글 실패';
      setMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const updateRoiSl = async () => {
    if (takeProfitInput <= 0 || takeProfitInput > 1000) {
      setMessage('Take Profit ROI는 0보다 크고 1000 이하의 값이어야 합니다.');
      return;
    }
    if (stopLossInput <= 0 || stopLossInput > 100) {
      setMessage('Stop Loss ROI는 0보다 크고 100 이하의 값이어야 합니다.');
      return;
    }

    setLoading(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/trading/roi-sl`, {
        take_profit_roi: takeProfitInput,
        stop_loss_roi: stopLossInput
      });
      if (response.data.success) {
        setTakeProfitRoi(takeProfitInput);
        setStopLossRoi(stopLossInput);
        setMessage(`Take Profit ${takeProfitInput}%, Stop Loss ${stopLossInput}%로 설정되었습니다.`);
      } else {
        setMessage(response.data.error || 'ROI/SL 설정 실패');
      }
    } catch (error) {
      console.error('ROI/SL 설정 실패:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || 'ROI/SL 설정 실패';
      setMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const updateTradingMode = async (mode) => {
    setLoading(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/trading/mode`, {
        mode: mode
      });
      if (response.data.success) {
        setTradingMode(mode);
        setMessage(`투자 모드가 ${mode === 'aggressive' ? '공격적' : mode === 'conservative' ? '보수적' : '노말'}로 설정되었습니다.`);
      } else {
        setMessage(response.data.error || '투자 모드 설정 실패');
      }
    } catch (error) {
      console.error('투자 모드 설정 실패:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || '투자 모드 설정 실패';
      setMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const updateLeverage = async () => {
    if (leverageInput < 1 || leverageInput > 125) {
      setMessage('레버리지는 1~125 사이의 값이어야 합니다.');
      return;
    }

    setLoading(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/trading/leverage`, {
        leverage: leverageInput
      });
      if (response.data.success) {
        setLeverage(leverageInput);
        setMessage(`레버리지가 ${leverageInput}배로 설정되었습니다.`);
      } else {
        setMessage(response.data.error || '레버리지 설정 실패');
      }
    } catch (error) {
      console.error('레버리지 설정 실패:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || '레버리지 설정 실패';
      setMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const executeCycle = async () => {
    setLoading(true);
    setMessage('');
    try {
      // 실제 거래 모드로 실행 (dry_run=false)
      const response = await axios.post(`${API_BASE_URL}/api/trading/execute-cycle`, {
        dry_run: false,
        leverage: leverage
      });
      if (response.data.success) {
        setMessage('거래 사이클 실행 완료 (실제 거래 모드)');
        // 잔액과 포지션 정보 갱신
        await fetchBalance();
        await fetchPosition();
        if (onUpdate) onUpdate();
      } else {
        setMessage(response.data.message || '거래 사이클 실행 실패');
      }
    } catch (error) {
      console.error('거래 사이클 실행 실패:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || '거래 사이클 실행 실패';
      setMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const closePosition = async () => {
    if (!window.confirm('정말로 포지션을 닫으시겠습니까?')) {
      return;
    }
    
    setLoading(true);
    setMessage('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/trading/close-position`);
      if (response.data.success) {
        setMessage('포지션 닫기 완료');
        await fetchBalance();
        await fetchPosition();
        if (onUpdate) onUpdate();
      } else {
        setMessage(response.data.message || '포지션 닫기 실패');
      }
    } catch (error) {
      console.error('포지션 닫기 실패:', error);
      setMessage(error.response?.data?.error || '포지션 닫기 실패');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    fetchBalance();
    fetchPosition();
    fetchLeverage();
    fetchRoiSl();
    fetchTradingMode();
    fetchTradingStatus();
    const interval = setInterval(() => {
      fetchBalance();
      fetchPosition();
      fetchLeverage();
      fetchRoiSl();
      fetchTradingMode();
      fetchTradingStatus();
    }, 30000); // 30초마다 갱신
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="trading-panel">
      <h3>거래 제어</h3>
      
      {message && (
        <div className={`message ${message.includes('실패') ? 'error' : 'success'}`}>
          {message}
        </div>
      )}

      <div className="trading-settings">
        <div className="setting-group">
          <h4>레버리지 설정</h4>
          <div className="leverage-input-group">
            <label htmlFor="leverage-input">레버리지:</label>
            <input
              id="leverage-input"
              type="number"
              min="1"
              max="125"
              value={leverageInput}
              onChange={(e) => setLeverageInput(parseInt(e.target.value) || 30)}
              disabled={loading}
              className="leverage-input"
            />
            <span className="leverage-unit">배</span>
            <button
              onClick={updateLeverage}
              disabled={loading || leverageInput === leverage}
              className="btn-secondary"
            >
              설정
            </button>
          </div>
          <div className="current-setting">
            현재: <strong>{leverage}배</strong>
          </div>
        </div>

        <div className="setting-group">
          <h4>ROI/SL 설정</h4>
          <div className="roi-sl-inputs">
            <div className="roi-input-group">
              <label htmlFor="tp-input">Take Profit:</label>
              <input
                id="tp-input"
                type="number"
                min="0.1"
                max="1000"
                step="0.1"
                value={takeProfitInput}
                onChange={(e) => setTakeProfitInput(parseFloat(e.target.value) || 40)}
                disabled={loading}
                className="roi-input"
              />
              <span className="roi-unit">%</span>
            </div>
            <div className="roi-input-group">
              <label htmlFor="sl-input">Stop Loss:</label>
              <input
                id="sl-input"
                type="number"
                min="0.1"
                max="100"
                step="0.1"
                value={stopLossInput}
                onChange={(e) => setStopLossInput(parseFloat(e.target.value) || 5)}
                disabled={loading}
                className="roi-input"
              />
              <span className="roi-unit">%</span>
            </div>
            <button
              onClick={updateRoiSl}
              disabled={loading || (takeProfitInput === takeProfitRoi && stopLossInput === stopLossRoi)}
              className="btn-secondary"
            >
              설정
            </button>
          </div>
          <div className="current-setting">
            현재: TP <strong>{takeProfitRoi}%</strong>, SL <strong>{stopLossRoi}%</strong>
          </div>
        </div>

        <div className="setting-group">
          <h4>투자 모드</h4>
          <div className="mode-buttons">
            <button
              onClick={() => updateTradingMode('aggressive')}
              disabled={loading}
              className={`mode-btn ${tradingMode === 'aggressive' ? 'active aggressive' : ''}`}
            >
              공격적
            </button>
            <button
              onClick={() => updateTradingMode('normal')}
              disabled={loading}
              className={`mode-btn ${tradingMode === 'normal' ? 'active normal' : ''}`}
            >
              노말
            </button>
            <button
              onClick={() => updateTradingMode('conservative')}
              disabled={loading}
              className={`mode-btn ${tradingMode === 'conservative' ? 'active conservative' : ''}`}
            >
              보수적
            </button>
          </div>
          <div className="current-setting">
            현재 모드: <strong>{tradingMode === 'aggressive' ? '공격적' : tradingMode === 'conservative' ? '보수적' : '노말'}</strong>
          </div>
        </div>
      </div>

      <div className="trading-cycle-control">
        <div className="cycle-toggle-section">
          <div className="cycle-status">
            <span className="cycle-label">거래 사이클:</span>
            <span className={`cycle-status-badge ${tradingEnabled ? 'enabled' : 'disabled'}`}>
              {tradingEnabled ? '🟢 활성화' : '🔴 비활성화'}
            </span>
          </div>
          <button
            onClick={toggleTradingCycle}
            disabled={loading}
            className={`cycle-toggle-btn ${tradingEnabled ? 'disable' : 'enable'}`}
          >
            {tradingEnabled ? '⏸️ 비활성화' : '▶️ 활성화'}
          </button>
        </div>
        <div className="cycle-description">
          {tradingEnabled 
            ? '거래 사이클이 활성화되어 있습니다. LLM 응답에 따라 자동으로 거래가 실행됩니다.'
            : '거래 사이클이 비활성화되어 있습니다. 활성화하면 LLM 응답에 따라 자동 거래가 시작됩니다.'}
        </div>
      </div>

      <div className="trading-controls">
        <button 
          onClick={executeCycle} 
          disabled={loading}
          className="btn-primary"
        >
          {loading ? '실행 중...' : '거래 사이클 실행 (수동)'}
        </button>
        
        {position && (
          <button 
            onClick={closePosition} 
            disabled={loading}
            className="btn-danger"
          >
            포지션 닫기
          </button>
        )}
      </div>

      {balance && (
        <div className="balance-info">
          <h4>자산 정보</h4>
          <div className="info-item">
            <span className="label">총 자산:</span>
            <span className="value">${balance.total.toFixed(2)}</span>
          </div>
          <div className="info-item">
            <span className="label">거래 가능:</span>
            <span className="value">${balance.available.toFixed(2)}</span>
            {balance.available_ratio !== undefined && (
              <span className={`ratio ${balance.available_ratio >= 70 ? 'good' : 'warning'}`}>
                ({balance.available_ratio.toFixed(1)}%)
              </span>
            )}
          </div>
          <div className="info-item">
            <span className="label">사용중인 금액:</span>
            <span className={`value ${balance.used > 0 ? 'warning' : ''}`}>
              ${(balance.used || 0).toFixed(2)}
            </span>
          </div>
          {balance.position_value > 0 && (
            <div className="info-item">
              <span className="label">포지션 가치:</span>
              <span className="value">${balance.position_value.toFixed(2)}</span>
            </div>
          )}
          {balance.min_required_amount && (
            <div className="info-item">
              <span className="label">최소 거래 필요:</span>
              <span className="value">${balance.min_required_amount.toFixed(2)} ({balance.min_required_ratio}%)</span>
            </div>
          )}
          
          {/* 거래 상태 표시 */}
          {tradingStatus && (
            <div className={`trading-status ${tradingStatus.can_trade ? 'can-trade' : 'cannot-trade'}`}>
              <span className="status-icon">
                {tradingStatus.can_trade ? '✅' : '⏸️'}
              </span>
              <div className="status-content">
                <span className="status-text">{tradingStatus.reason}</span>
                {tradingStatus.ai_recommendation && (
                  <span className="ai-recommendation">
                    AI 추천: <strong>{tradingStatus.ai_recommendation.toUpperCase()}</strong>
                  </span>
                )}
              </div>
            </div>
          )}
          
          {balance.used > 0 && !tradingStatus && (
            <div className="trading-status">
              <span className="status-icon">⏸️</span>
              <span className="status-text">포지션이 열려있어 새로운 거래를 하지 않습니다. 포지션이 닫히면 자동으로 다시 거래 가능합니다.</span>
            </div>
          )}
        </div>
      )}

      {position && (
        <div className={`position-info ${position.side}`}>
          <div className="position-header">
            <span className="position-side">{position.side.toUpperCase()}</span>
            <span className={`position-pnl ${position.unrealized_pnl >= 0 ? 'profit' : 'loss'}`}>
              ${position.unrealized_pnl.toFixed(2)} ({position.percentage.toFixed(2)}%)
            </span>
          </div>
          <div className="info-item">
            <span className="label">진입 가격:</span>
            <span className="value">${position.entry_price.toFixed(2)}</span>
          </div>
          <div className="info-item">
            <span className="label">현재 가격:</span>
            <span className="value">${position.mark_price.toFixed(2)}</span>
          </div>
          <div className="info-item">
            <span className="label">포지션 크기:</span>
            <span className="value">{position.size.toFixed(4)} BTC</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingPanel;


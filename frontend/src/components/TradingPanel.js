import React, { useState } from 'react';
import axios from 'axios';
import './TradingPanel.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5333';

const TradingPanel = ({ onUpdate }) => {
  const [balance, setBalance] = useState(null);
  const [position, setPosition] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const fetchBalance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/trading/balance`);
      if (response.data.success) {
        setBalance(response.data.balance);
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

  const executeCycle = async () => {
    setLoading(true);
    setMessage('');
    try {
      // 실제 거래 모드로 실행 (dry_run=false)
      const response = await axios.post(`${API_BASE_URL}/api/trading/execute-cycle`, {
        dry_run: false,
        leverage: 10
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
    const interval = setInterval(() => {
      fetchBalance();
      fetchPosition();
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

      <div className="trading-controls">
        <button 
          onClick={executeCycle} 
          disabled={loading}
          className="btn-primary"
        >
          {loading ? '실행 중...' : '거래 사이클 실행'}
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
          <div className="info-item">
            <span className="label">총 자산:</span>
            <span className="value">${balance.total.toFixed(2)}</span>
          </div>
          <div className="info-item">
            <span className="label">거래 가능:</span>
            <span className="value">${balance.available.toFixed(2)}</span>
          </div>
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


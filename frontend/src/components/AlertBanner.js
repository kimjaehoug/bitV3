import React, { useEffect, useState } from 'react';
import './AlertBanner.css';

const AlertBanner = ({ technicalIndicators }) => {
  const [alerts, setAlerts] = useState([]);
  const [previousState, setPreviousState] = useState({
    golden_cross: false,
    dead_cross: false
  });

  useEffect(() => {
    if (!technicalIndicators) return;

    const newAlerts = [];
    const currentGoldenCross = technicalIndicators.golden_cross || false;
    const currentDeadCross = technicalIndicators.dead_cross || false;

    // 골든크로스 감지 (이전에는 없었고 지금은 있는 경우)
    if (currentGoldenCross && !previousState.golden_cross) {
      newAlerts.push({
        id: Date.now(),
        type: 'golden_cross',
        message: '골든크로스 발생! 상승 추세 전환 신호',
        timestamp: new Date()
      });
    }

    // 데드크로스 감지 (이전에는 없었고 지금은 있는 경우)
    if (currentDeadCross && !previousState.dead_cross) {
      newAlerts.push({
        id: Date.now() + 1,
        type: 'dead_cross',
        message: '데드크로스 발생! 하락 추세 전환 신호',
        timestamp: new Date()
      });
    }

    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 5)); // 최대 5개까지만 표시
    }

    // 이전 상태 업데이트
    setPreviousState({
      golden_cross: currentGoldenCross,
      dead_cross: currentDeadCross
    });
  }, [technicalIndicators]);

  // 5초 후 알람 자동 제거
  useEffect(() => {
    if (alerts.length > 0) {
      const timer = setTimeout(() => {
        setAlerts(prev => prev.slice(1));
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [alerts]);

  if (alerts.length === 0) {
    return null;
  }

  return (
    <div className="alert-banner-container">
      {alerts.map(alert => (
        <div
          key={alert.id}
          className={`alert-banner ${alert.type}`}
        >
          <div className="alert-icon">
            {alert.type === 'golden_cross' ? '📈' : '📉'}
          </div>
          <div className="alert-content">
            <div className="alert-title">
              {alert.type === 'golden_cross' ? '골든크로스' : '데드크로스'}
            </div>
            <div className="alert-message">{alert.message}</div>
          </div>
          <button
            className="alert-close"
            onClick={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};

export default AlertBanner;


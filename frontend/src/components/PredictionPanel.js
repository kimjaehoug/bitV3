import React from 'react';
import './PredictionPanel.css';

const PredictionPanel = ({ predictionData, thresholdInfo }) => {
  if (!predictionData) {
    return (
      <div className="prediction-panel">
        <h3>예측 정보</h3>
        <div className="no-data">데이터 대기 중...</div>
      </div>
    );
  }

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'long':
        return '#10b981';
      case 'short':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getSignalText = (signal) => {
    switch (signal) {
      case 'long':
        return '롱 신호';
      case 'short':
        return '숏 신호';
      default:
        return '대기';
    }
  };

  return (
    <div className="prediction-panel">
      <h3>모델 예측</h3>
      
      <div className="current-price">
        <div className="label">현재 가격</div>
        <div className="value">${predictionData.current_price?.toFixed(2) || '0.00'}</div>
      </div>

      <div className="prediction-grid">
        <div className="prediction-item">
          <div className="label">30분 후 예측</div>
          <div className="value">
            ${predictionData.predicted_price_30m?.toFixed(2) || '0.00'}
          </div>
          <div className={`change ${predictionData.change_30m >= 0 ? 'positive' : 'negative'}`}>
            {predictionData.change_30m >= 0 ? '+' : ''}{predictionData.change_30m?.toFixed(2) || '0.00'}%
          </div>
        </div>

        <div className="prediction-item">
          <div className="label">1시간 후 예측</div>
          <div className="value">
            ${predictionData.predicted_price_1h?.toFixed(2) || '0.00'}
          </div>
          <div className={`change ${predictionData.change_1h >= 0 ? 'positive' : 'negative'}`}>
            {predictionData.change_1h >= 0 ? '+' : ''}{predictionData.change_1h?.toFixed(2) || '0.00'}%
          </div>
        </div>
      </div>

      <div className="signal-section">
        <div className="label">거래 신호</div>
        <div 
          className="signal-badge"
          style={{ backgroundColor: getSignalColor(predictionData.signal) }}
        >
          {getSignalText(predictionData.signal)}
        </div>
        <div className="confidence">
          신뢰도: {predictionData.confidence?.toFixed(1) || '0.0'}%
        </div>
      </div>

      {thresholdInfo && thresholdInfo.current_threshold !== null && (
        <div className="threshold-section">
          <div className="label">예측 모델 임계값</div>
          <div className="threshold-info">
            <div className="threshold-value">
              현재: {(thresholdInfo.current_threshold * 100).toFixed(2)}%
              {thresholdInfo.is_ai_adjusted && (
                <span className="ai-adjusted-badge"> (AI 조정됨)</span>
              )}
            </div>
            {thresholdInfo.original_threshold !== null && (
              <div className="threshold-original">
                기본: {(thresholdInfo.original_threshold * 100).toFixed(2)}%
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionPanel;



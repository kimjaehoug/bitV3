import React from 'react';
import './TechnicalIndicators.css';

const TechnicalIndicators = ({ indicators }) => {
  if (!indicators || Object.keys(indicators).length === 0) {
    return (
      <div className="technical-indicators">
        <h3>기술적 지표</h3>
        <div className="no-data">데이터 대기 중...</div>
      </div>
    );
  }

  const getRSIColor = (rsi) => {
    if (rsi >= 70) return '#ef4444'; // 과매수
    if (rsi <= 30) return '#10b981'; // 과매도
    return '#6b7280'; // 중립
  };

  const getRSILabel = (rsi) => {
    if (rsi >= 70) return '과매수';
    if (rsi <= 30) return '과매도';
    return '중립';
  };

  return (
    <div className="technical-indicators">
      <h3>기술적 지표</h3>

      <div className="indicators-grid">
        {indicators.ma5 && (
          <div className="indicator-item">
            <div className="indicator-label">MA5</div>
            <div className="indicator-value">${indicators.ma5.toFixed(2)}</div>
            <div className="indicator-line" style={{ backgroundColor: '#fbbf24' }}></div>
          </div>
        )}

        {indicators.ma20 && (
          <div className="indicator-item">
            <div className="indicator-label">MA20</div>
            <div className="indicator-value">${indicators.ma20.toFixed(2)}</div>
            <div className="indicator-line" style={{ backgroundColor: '#3b82f6' }}></div>
          </div>
        )}

        {indicators.ma50 && (
          <div className="indicator-item">
            <div className="indicator-label">MA50</div>
            <div className="indicator-value">${indicators.ma50.toFixed(2)}</div>
            <div className="indicator-line" style={{ backgroundColor: '#8b5cf6' }}></div>
          </div>
        )}

        {indicators.rsi !== null && indicators.rsi !== undefined && (
          <div className="indicator-item rsi">
            <div className="indicator-label">RSI</div>
            <div className="indicator-value" style={{ color: getRSIColor(indicators.rsi) }}>
              {indicators.rsi.toFixed(1)}
            </div>
            <div className="rsi-label" style={{ color: getRSIColor(indicators.rsi) }}>
              {getRSILabel(indicators.rsi)}
            </div>
          </div>
        )}

        {indicators.golden_cross && (
          <div className="signal-item golden">
            <div className="signal-icon">✨</div>
            <div className="signal-text">골든크로스</div>
          </div>
        )}

        {indicators.dead_cross && (
          <div className="signal-item dead">
            <div className="signal-icon">⚠️</div>
            <div className="signal-text">데드크로스</div>
          </div>
        )}
      </div>

      {(indicators.bollinger_upper || indicators.bollinger_lower) && (
        <div className="bollinger-section">
          <div className="section-title">볼린저 밴드</div>
          <div className="bollinger-values">
            {indicators.bollinger_upper && (
              <div className="bb-item">
                <span className="bb-label">상단</span>
                <span className="bb-value">${indicators.bollinger_upper.toFixed(2)}</span>
              </div>
            )}
            {indicators.bollinger_middle && (
              <div className="bb-item">
                <span className="bb-label">중간</span>
                <span className="bb-value">${indicators.bollinger_middle.toFixed(2)}</span>
              </div>
            )}
            {indicators.bollinger_lower && (
              <div className="bb-item">
                <span className="bb-label">하단</span>
                <span className="bb-value">${indicators.bollinger_lower.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TechnicalIndicators;



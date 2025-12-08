import React from 'react';
import './MarketIndicatorsPanel.css';

const MarketIndicatorsPanel = ({ marketIndicators }) => {
  if (!marketIndicators) {
    return (
      <div className="market-indicators-panel">
        <h3>시장 지표</h3>
        <div className="no-data">데이터 대기 중...</div>
      </div>
    );
  }

  const getStrengthColor = (strength) => {
    if (strength === 'strong_buy' || strength === 'strong') return '#10b981';
    if (strength === 'buy' || strength === 'bullish') return '#34d399';
    if (strength === 'strong_sell' || strength === 'weak') return '#ef4444';
    if (strength === 'sell' || strength === 'bearish') return '#f87171';
    return '#6b7280';
  };

  const getStrengthText = (strength) => {
    const mapping = {
      'strong_buy': '강한 매수',
      'buy': '매수',
      'neutral': '중립',
      'sell': '매도',
      'strong_sell': '강한 매도',
      'strong': '강함',
      'weak': '약함',
      'bullish': '상승',
      'bearish': '하락',
      'normal': '정상',
      'squeeze': '압축',
      'expansion': '확장',
      'surge': '급증',
      'decline': '감소',
      'balanced': '균형',
      'turnover': '전환'
    };
    return mapping[strength] || strength;
  };

  return (
    <div className="market-indicators-panel">
      <h3>시장 지표</h3>
      
      <div className="indicators-list">
        {/* 오더북 불균형 */}
        <div className="indicator-item">
          <div className="indicator-header">
            <span className="indicator-icon">📊</span>
            <span className="indicator-name">오더북 불균형</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.orderbook?.strength) }}
            >
              {getStrengthText(marketIndicators.orderbook?.strength)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">비율:</span>
              <span className="detail-value">{marketIndicators.orderbook?.ratio?.toFixed(2) || '0.00'}%</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">스프레드:</span>
              <span className="detail-value">{marketIndicators.orderbook?.spread_pct?.toFixed(3) || '0.000'}%</span>
            </div>
          </div>
        </div>

        {/* 청산 클러스터 */}
        <div className="indicator-item">
          <div className="indicator-header">
            <span className="indicator-icon">💥</span>
            <span className="indicator-name">청산 클러스터</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.liquidation?.strength) }}
            >
              {getStrengthText(marketIndicators.liquidation?.strength)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">비율:</span>
              <span className="detail-value">{marketIndicators.liquidation?.ratio?.toFixed(2) || '0.00'}%</span>
            </div>
          </div>
        </div>

        {/* 변동성 */}
        <div className="indicator-item">
          <div className="indicator-header">
            <span className="indicator-icon">📉</span>
            <span className="indicator-name">변동성</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.volatility?.status) }}
            >
              {getStrengthText(marketIndicators.volatility?.status)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">폭발 가능성:</span>
              <span className="detail-value">{getStrengthText(marketIndicators.volatility?.expansion_potential)}</span>
            </div>
          </div>
        </div>

        {/* OI (미체결약정) */}
        <div className="indicator-item">
          <div className="indicator-header">
            <span className="indicator-icon">💰</span>
            <span className="indicator-name">OI (미체결약정)</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.oi?.status) }}
            >
              {getStrengthText(marketIndicators.oi?.status)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">방향:</span>
              <span className="detail-value">{getStrengthText(marketIndicators.oi?.direction)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">펀딩:</span>
              <span className="detail-value">{marketIndicators.oi?.funding_rate?.toFixed(4) || '0.0000'}%</span>
            </div>
          </div>
        </div>

        {/* CVD (누적 거래량 차이) */}
        <div className="indicator-item">
          <div className="indicator-header">
            <span className="indicator-icon">🔄</span>
            <span className="indicator-name">CVD</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.cvd?.trend) }}
            >
              {getStrengthText(marketIndicators.cvd?.trend)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">전환:</span>
              <span className="detail-value">{marketIndicators.cvd?.turnover ? '예' : '아니오'}</span>
            </div>
          </div>
        </div>

        {/* 종합 신호 */}
        <div className="indicator-item summary">
          <div className="indicator-header">
            <span className="indicator-icon">🎯</span>
            <span className="indicator-name">종합 신호</span>
            <span 
              className="indicator-strength"
              style={{ color: getStrengthColor(marketIndicators.signal) }}
            >
              {getStrengthText(marketIndicators.signal)}
            </span>
          </div>
          <div className="indicator-details">
            <div className="detail-row">
              <span className="detail-label">신뢰도:</span>
              <span className="detail-value">{marketIndicators.confidence?.toFixed(1) || '0.0'}%</span>
            </div>
            {marketIndicators.reasons && marketIndicators.reasons.length > 0 && (
              <div className="reasons-list">
                <div className="reasons-label">근거:</div>
                {marketIndicators.reasons.map((reason, idx) => (
                  <div key={idx} className="reason-item">• {reason}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketIndicatorsPanel;




import React from 'react';
import './PositionPanel.css';

const PositionPanel = ({ positionData }) => {
  if (!positionData) {
    return (
      <div className="position-panel">
        <h3>포지션 정보</h3>
        <div className="no-position">
          <div className="status-icon">○</div>
          <div>포지션 없음</div>
        </div>
      </div>
    );
  }

  const isLong = positionData.side === 'long';
  const pnlColor = positionData.unrealized_pnl >= 0 ? '#10b981' : '#ef4444';

  return (
    <div className="position-panel">
      <h3>포지션 정보</h3>
      
      <div className={`position-status ${isLong ? 'long' : 'short'}`}>
        <div className="status-badge">
          {isLong ? '🔼 롱' : '🔽 숏'}
        </div>
      </div>

      <div className="position-details">
        <div className="detail-item">
          <div className="label">진입 가격</div>
          <div className="value">${positionData.entry_price?.toFixed(2) || '0.00'}</div>
        </div>

        <div className="detail-item">
          <div className="label">포지션 크기</div>
          <div className="value">{positionData.size?.toFixed(4) || '0.0000'} BTC</div>
        </div>

        <div className="detail-item">
          <div className="label">미실현 손익</div>
          <div className="value" style={{ color: pnlColor }}>
            ${positionData.unrealized_pnl?.toFixed(2) || '0.00'}
          </div>
        </div>

        {positionData.entry_time && (
          <div className="detail-item">
            <div className="label">진입 시간</div>
            <div className="value small">
              {new Date(positionData.entry_time).toLocaleString('ko-KR')}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PositionPanel;



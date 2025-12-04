import React from 'react';
import './ChartControls.css';

const ChartControls = ({ visibility, onToggle }) => {
  return (
    <div className="chart-controls-panel">
      <h3>차트 표시 설정</h3>
      <div className="controls-list">
        <div className="control-group">
          <h4>이동평균선</h4>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.ma5}
              onChange={() => onToggle('ma5')}
            />
            <span>MA5</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.ma20}
              onChange={() => onToggle('ma20')}
            />
            <span>MA20</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.ma50}
              onChange={() => onToggle('ma50')}
            />
            <span>MA50</span>
          </label>
        </div>

        <div className="control-group">
          <h4>지지선/저항선</h4>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.support}
              onChange={() => onToggle('support')}
            />
            <span>지지선</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.resistance}
              onChange={() => onToggle('resistance')}
            />
            <span>저항선</span>
          </label>
        </div>

        <div className="control-group">
          <h4>추세선</h4>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.uptrend}
              onChange={() => onToggle('uptrend')}
            />
            <span>상승 추세선 (빗각)</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.downtrend}
              onChange={() => onToggle('downtrend')}
            />
            <span>하락 추세선 (엇각)</span>
          </label>
        </div>

        <div className="control-group">
          <h4>피보나치</h4>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.fibonacci}
              onChange={() => onToggle('fibonacci')}
            />
            <span>피보나치 되돌림</span>
          </label>
        </div>

        <div className="control-group">
          <h4>볼린저 밴드</h4>
          <label className="control-item">
            <input
              type="checkbox"
              checked={visibility.bollinger}
              onChange={() => onToggle('bollinger')}
            />
            <span>볼린저 밴드</span>
          </label>
        </div>
      </div>
    </div>
  );
};

export default ChartControls;



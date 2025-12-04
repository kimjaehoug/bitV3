import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './GeminiAnalysisPanel.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5333';

const GeminiAnalysisPanel = ({
  priceData,
  predictionData,
  technicalIndicators,
  supportResistance,
  trendLines,
  marketIndicators,
  fibonacci,
  socket
}) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [isAutoUpdate, setIsAutoUpdate] = useState(false); // 자동 업데이트 여부
  const [patternResult, setPatternResult] = useState(null);
  const [patternLoading, setPatternLoading] = useState(false);
  const [patternError, setPatternError] = useState(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [additionalQuestion, setAdditionalQuestion] = useState('');
  const [askingQuestion, setAskingQuestion] = useState(false);
  const [questionHistory, setQuestionHistory] = useState([]);
  const [selectedModel, setSelectedModel] = useState('gemini-2.5-flash');
  const [includePattern, setIncludePattern] = useState(false);
  const [nextUpdateTime, setNextUpdateTime] = useState(null);
  const [updateInterval, setUpdateInterval] = useState(300); // 5분 기본값
  const [timeUntilUpdate, setTimeUntilUpdate] = useState(null);

  // 타이머 업데이트 (1초마다)
  useEffect(() => {
    if (!nextUpdateTime) return;

    const updateTimer = () => {
      const now = new Date();
      const next = new Date(nextUpdateTime);
      const diff = Math.max(0, Math.floor((next - now) / 1000)); // 초 단위
      setTimeUntilUpdate(diff);
    };

    updateTimer(); // 즉시 실행
    const interval = setInterval(updateTimer, 1000);

    return () => clearInterval(interval);
  }, [nextUpdateTime]);

  // WebSocket을 통한 자동 AI 분석 업데이트 구독
  useEffect(() => {
    if (!socket) return;

    const handleAiAnalysisUpdate = (data) => {
      if (data && data.analysis) {
        setAnalysis(data.analysis);
        setLastUpdate(new Date(data.timestamp || new Date()));
        setIsAutoUpdate(true); // 자동 업데이트로 표시
        setError(null); // 오류 초기화
        
        // 다음 업데이트 시간 설정
        if (data.next_update_time) {
          setNextUpdateTime(data.next_update_time);
        } else if (data.update_interval) {
          // next_update_time이 없으면 현재 시간 + interval로 계산
          const next = new Date();
          next.setSeconds(next.getSeconds() + data.update_interval);
          setNextUpdateTime(next.toISOString());
        }
        
        if (data.update_interval) {
          setUpdateInterval(data.update_interval);
        }
        
        console.log('🤖 자동 AI 분석 업데이트:', data.analysis);
      }
    };

    socket.on('ai_analysis_update', handleAiAnalysisUpdate);

    return () => {
      socket.off('ai_analysis_update', handleAiAnalysisUpdate);
    };
  }, [socket]);

  // 수동 분석 시 자동 업데이트 플래그 해제
  useEffect(() => {
    if (analysis && !isAutoUpdate) {
      // 수동 분석으로 설정된 경우
    }
  }, [analysis, isAutoUpdate]);

  // 유사 패턴 찾기 핸들러 (Gemini 없이)
  const handleFindPattern = async () => {
    if (!priceData || priceData.length === 0) {
      setPatternError('가격 데이터가 필요합니다.');
      return;
    }

    setPatternLoading(true);
    setPatternError(null);
    setPatternResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/pattern/find`, {
        priceData
      });

      if (response.data.success) {
        setPatternResult(response.data.pattern);
      } else {
        throw new Error(response.data.error || '패턴 찾기 실패');
      }
    } catch (err) {
      console.error('패턴 찾기 오류:', err);
      if (err.response) {
        setPatternError(err.response.data?.error || `서버 오류: ${err.response.status}`);
      } else if (err.request) {
        setPatternError('서버에 연결할 수 없습니다.');
      } else {
        setPatternError(err.message || '패턴 찾기 중 오류가 발생했습니다.');
      }
    } finally {
      setPatternLoading(false);
    }
  };

  // 수동 분석 버튼 핸들러 (버튼 클릭 시에만 실행)
  const handleManualAnalysis = async () => {
    // 최소한의 데이터가 있는지 확인
    if (!priceData || priceData.length === 0 || !predictionData) {
      setError('분석을 위한 데이터가 충분하지 않습니다. 잠시 후 다시 시도해주세요.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // 백엔드 API를 통해 Gemini 분석 요청
      const requestData = {
        priceData,
        predictionData,
        technicalIndicators,
        supportResistance,
        trendLines,
        marketIndicators,
        fibonacci,
        sessionId,
        modelName: selectedModel,
        includeSimilarPattern: includePattern
      };
      
      const response = await axios.post(`${API_BASE_URL}/api/gemini/analyze`, requestData);

      if (response.data.success) {
        setAnalysis(response.data.analysis);
        setLastUpdate(new Date());
        setIsAutoUpdate(false); // 수동 분석으로 표시
      } else {
        throw new Error(response.data.error || '분석 실패');
      }
    } catch (err) {
      console.error('Gemini 분석 오류:', err);
      console.error('오류 상세:', err.response?.data);
      
      if (err.response) {
        // 서버 응답이 있는 경우
        const errorMessage = err.response.data?.error || err.response.data?.message || `서버 오류: ${err.response.status}`;
        setError(errorMessage);
        
        // 400 오류인 경우 추가 정보 표시
        if (err.response.status === 400) {
          console.error('400 오류 상세:', err.response.data);
        }
      } else if (err.request) {
        // 요청은 보냈지만 응답을 받지 못한 경우
        setError('서버에 연결할 수 없습니다. API 서버가 실행 중인지 확인해주세요.');
      } else {
        // 요청 설정 중 오류
        setError(err.message || '분석 중 오류가 발생했습니다.');
      }
    } finally {
      setLoading(false);
    }
  };

  // 추가 질문 핸들러
  const handleAskQuestion = async () => {
    if (!additionalQuestion.trim() || !analysis) {
      return;
    }

    setAskingQuestion(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/gemini/ask`, {
        sessionId,
        question: additionalQuestion
      });

      if (response.data.success) {
        setQuestionHistory(prev => [...prev, {
          question: additionalQuestion,
          answer: response.data.answer,
          timestamp: new Date()
        }]);
        setAdditionalQuestion('');
      } else {
        throw new Error(response.data.error || '질문 처리 실패');
      }
    } catch (err) {
      console.error('추가 질문 오류:', err);
      setError(err.response?.data?.error || err.message || '질문 처리 중 오류가 발생했습니다.');
    } finally {
      setAskingQuestion(false);
    }
  };

  return (
    <div className="gemini-analysis-panel">
      <div className="panel-header">
        <h3>Gemini AI 분석</h3>
        <div className="controls-section">
          <div className="model-selector">
            <label htmlFor="model-select">모델:</label>
            <select
              id="model-select"
              className="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
              <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
              <option value="gemini-pro">Gemini Pro</option>
              <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
            </select>
          </div>
          <div className="pattern-option">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={includePattern}
                onChange={(e) => setIncludePattern(e.target.checked)}
                disabled={loading}
              />
              <span>유사 패턴 포함</span>
            </label>
          </div>
        </div>
        <div className="button-group">
          <button 
            className="pattern-button" 
            onClick={handleFindPattern}
            disabled={patternLoading || !priceData || priceData.length === 0}
            title="Dataset에서 유사한 차트 패턴 찾기 (무료)"
          >
            {patternLoading ? '패턴 찾는 중...' : '🔍 유사 패턴 찾기'}
          </button>
          <button 
            className="analyze-button" 
            onClick={handleManualAnalysis}
            disabled={loading || !priceData || priceData.length === 0 || !predictionData}
          >
            {loading ? '분석 중...' : '🤖 AI 분석 요청'}
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-message">AI 분석 중... (잠시만 기다려주세요)</div>
      )}

      {!analysis && !loading && (
        <div className="info-message">
          버튼을 클릭하여 현재 시장 데이터를 분석하고 거래 유의점을 확인하세요.
        </div>
      )}

      {error && (
        <div className="error-message">{error}</div>
      )}

      {/* 추가 질문 섹션 */}
      {analysis && (
        <div className="additional-question-section">
          <div className="question-input-wrapper">
            <input
              type="text"
              className="question-input"
              placeholder="추가 질문을 입력하세요 (예: 이 패턴의 위험도는? 목표가는 얼마가 적절한가요?)"
              value={additionalQuestion}
              onChange={(e) => setAdditionalQuestion(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleAskQuestion();
                }
              }}
              disabled={askingQuestion}
            />
            <button
              className="ask-button"
              onClick={handleAskQuestion}
              disabled={askingQuestion || !additionalQuestion.trim()}
            >
              {askingQuestion ? '질문 중...' : '💬 질문하기'}
            </button>
          </div>
        </div>
      )}

      {/* 질문 히스토리 */}
      {questionHistory.length > 0 && (
        <div className="question-history">
          <div className="question-history-header">
            <h4 className="question-history-title">질문 히스토리</h4>
            <button
              className="clear-history-button"
              onClick={() => setQuestionHistory([])}
              title="히스토리 삭제"
            >
              🗑️ 삭제
            </button>
          </div>
          {questionHistory.map((item, idx) => (
            <div key={idx} className="question-item">
              <div className="question-item-header">
                <span className="question-number">질문 {idx + 1}</span>
                <button
                  className="delete-question-button"
                  onClick={() => {
                    setQuestionHistory(prev => prev.filter((_, i) => i !== idx));
                  }}
                  title="이 질문 삭제"
                >
                  ✕
                </button>
              </div>
              <div className="question-text">
                <strong>Q:</strong> {item.question}
              </div>
              <div className="answer-text">
                <strong>A:</strong> {item.answer}
              </div>
            </div>
          ))}
        </div>
      )}

      {analysis && (
        <div className="analysis-content">
          {lastUpdate && (
            <div className="last-update">
              마지막 업데이트: {lastUpdate.toLocaleTimeString('ko-KR')}
              {isAutoUpdate && <span className="auto-update-badge"> (자동 업데이트)</span>}
              {timeUntilUpdate !== null && (
                <span className="timer-badge">
                  {' | '}
                  다음 업데이트: {Math.floor(timeUntilUpdate / 60)}분 {timeUntilUpdate % 60}초
                </span>
              )}
            </div>
          )}

          {/* 최종 추천 */}
          {analysis.recommendation && (
            <div className={`recommendation-badge ${analysis.recommendation}`}>
              <div className="recommendation-icon">
                {analysis.recommendation === 'long' ? '📈' : 
                 analysis.recommendation === 'short' ? '📉' : '👀'}
              </div>
              <div className="recommendation-text">
                <div className="recommendation-label">AI 최종 추천</div>
                <div className="recommendation-value">
                  {analysis.recommendation === 'long' ? '롱 포지션' : 
                   analysis.recommendation === 'short' ? '숏 포지션' : '관망'}
                </div>
              </div>
            </div>
          )}

          {/* 관망일 때: 다음 타이밍 */}
          {analysis.recommendation === 'waiting' && analysis.next_timing && (
            <div className="timing-info">
              <div className="timing-header">
                <span className="timing-icon">⏰</span>
                <span className="timing-title">다음 매수/매도 타이밍</span>
              </div>
              <div className="timing-content">
                {analysis.next_timing}
              </div>
            </div>
          )}

          {/* 매수/매도 추천일 때: 목표가 및 손절가 */}
          {(analysis.recommendation === 'long' || analysis.recommendation === 'short') && (
            <div className="price-targets">
              {analysis.target_price && (
                <div className="price-target target">
                  <div className="price-label">
                    <span className="price-icon">🎯</span>
                    <span>목표가</span>
                  </div>
                  <div className="price-value target-price">
                    ${typeof analysis.target_price === 'number' ? analysis.target_price.toFixed(2) : analysis.target_price}
                  </div>
                </div>
              )}
              {analysis.stop_loss_price && (
                <div className="price-target stop-loss">
                  <div className="price-label">
                    <span className="price-icon">🛑</span>
                    <span>손절가</span>
                  </div>
                  <div className="price-value stop-loss-price">
                    ${typeof analysis.stop_loss_price === 'number' ? analysis.stop_loss_price.toFixed(2) : analysis.stop_loss_price}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 관망 유의점 */}
          {analysis.waiting && (
            <div className="analysis-section waiting">
              <div className="section-header">
                <span className="section-icon">👀</span>
                <span className="section-title">관망 유의점</span>
              </div>
              <div className="section-content">
                {Array.isArray(analysis.waiting) ? (
                  <ul>
                    {analysis.waiting.map((point, idx) => (
                      <li key={idx}>{point}</li>
                    ))}
                  </ul>
                ) : (
                  <p>{analysis.waiting}</p>
                )}
              </div>
            </div>
          )}

          {/* 롱 유의점 */}
          {analysis.long && (
            <div className="analysis-section long">
              <div className="section-header">
                <span className="section-icon">📈</span>
                <span className="section-title">롱 포지션 유의점</span>
              </div>
              <div className="section-content">
                {Array.isArray(analysis.long) ? (
                  <ul>
                    {analysis.long.map((point, idx) => (
                      <li key={idx}>{point}</li>
                    ))}
                  </ul>
                ) : (
                  <p>{analysis.long}</p>
                )}
              </div>
            </div>
          )}

          {/* 숏 유의점 */}
          {analysis.short && (
            <div className="analysis-section short">
              <div className="section-header">
                <span className="section-icon">📉</span>
                <span className="section-title">숏 포지션 유의점</span>
              </div>
              <div className="section-content">
                {Array.isArray(analysis.short) ? (
                  <ul>
                    {analysis.short.map((point, idx) => (
                      <li key={idx}>{point}</li>
                    ))}
                  </ul>
                ) : (
                  <p>{analysis.short}</p>
                )}
              </div>
            </div>
          )}

          {/* 종합 의견 */}
          {analysis.summary && (
            <div className="analysis-section summary">
              <div className="section-header">
                <span className="section-icon">💡</span>
                <span className="section-title">종합 의견</span>
              </div>
              <div className="section-content">
                <p>{analysis.summary}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 유사 패턴 찾기 결과 */}
      {patternLoading && (
        <div className="loading-message">유사 패턴 찾는 중...</div>
      )}

      {patternError && (
        <div className="error-message">{patternError}</div>
      )}

      {patternResult && (
        <div className="pattern-result">
          <div className="pattern-header">
            <span className="pattern-icon">📊</span>
            <span className="pattern-title">유사 패턴 발견</span>
          </div>
          <div className="pattern-content">
            <div className="pattern-type">
              <strong>패턴 유형:</strong> {patternResult.pattern_type}
            </div>
            <div className="pattern-similarity">
              <strong>유사도:</strong> {patternResult.similarity_score?.toFixed(1)}%
            </div>
            <div className="pattern-description">
              <strong>설명:</strong> {patternResult.description}
            </div>
            <div className="pattern-image-section">
              <strong>참고 패턴 이미지:</strong>
              <div className="pattern-image-wrapper">
                <img 
                  src={`${API_BASE_URL}/api/pattern/image/${patternResult.pattern_type}/${patternResult.pattern_file}`}
                  alt={`${patternResult.pattern_type} 패턴`}
                  className="pattern-reference-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'block';
                  }}
                />
                <div className="pattern-image-error" style={{display: 'none'}}>
                  이미지를 불러올 수 없습니다: {patternResult.pattern_file}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GeminiAnalysisPanel;


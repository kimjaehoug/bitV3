import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import './TradingChart.css';

const TradingChart = ({ 
  priceData, 
  technicalIndicators, 
  supportResistance, 
  positionData,
  fibonacci,
  trendLines,
  marketIndicators,
  visibility = {
    ma5: true,
    ma20: true,
    ma50: true,
    support: true,
    resistance: true,
    uptrend: true,
    downtrend: true,
    fibonacci: true,
    bollinger: false
  }
}) => {
  const chartContainerRef = useRef();
  const volumeContainerRef = useRef();
  const chartRef = useRef();
  const volumeChartRef = useRef();
  const candlestickSeriesRef = useRef();
  const volumeSeriesRef = useRef();
  const ma5SeriesRef = useRef();
  const ma20SeriesRef = useRef();
  const ma50SeriesRef = useRef();
  const supportSeriesRef = useRef();
  const resistanceSeriesRef = useRef();
  const bollingerUpperRef = useRef();
  const bollingerMiddleRef = useRef();
  const bollingerLowerRef = useRef();
  const [markers, setMarkers] = useState([]);
  const fibSeriesRefs = {
    fib_0: useRef(),
    fib_236: useRef(),
    fib_382: useRef(),
    fib_500: useRef(),
    fib_618: useRef(),
    fib_786: useRef(),
    fib_100: useRef(),
  };
  const uptrendSeriesRef = useRef();
  const downtrendSeriesRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current || !volumeContainerRef.current) return;

    // 가격 차트 생성
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1f3a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2d3748' },
        horzLines: { color: '#2d3748' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#2d3748',
      },
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 100,  // 우측 여백 대폭 증가
        barSpacing: 3,
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    chartRef.current = chart;

    // 거래량 차트 생성 (별도 차트)
    const volumeChart = createChart(volumeContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1f3a' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2d3748' },
        horzLines: { color: '#2d3748' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#2d3748',
        scaleMargins: {
          top: 0.1,
          bottom: 0,
        },
      },
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 100,  // 우측 여백 대폭 증가
        barSpacing: 3,
      },
      width: volumeContainerRef.current.clientWidth,
      height: 200,
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    volumeChartRef.current = volumeChart;

    // 시간 스케일 동기화 (데이터가 있을 때만)
    let isUpdating = false;
    chart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange && !isUpdating) {
        isUpdating = true;
        try {
          volumeChart.timeScale().setVisibleRange(timeRange);
        } catch (e) {
          console.warn('시간 스케일 동기화 오류:', e);
        } finally {
          isUpdating = false;
        }
      }
    });

    volumeChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange && !isUpdating) {
        isUpdating = true;
        try {
          chart.timeScale().setVisibleRange(timeRange);
        } catch (e) {
          console.warn('시간 스케일 동기화 오류:', e);
        } finally {
          isUpdating = false;
        }
      }
    });

    // 캔들스틱 시리즈
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });
    candlestickSeriesRef.current = candlestickSeries;

    // 거래량 시리즈 (별도 차트에)
    const volumeSeries = volumeChart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'right',
    });
    volumeSeriesRef.current = volumeSeries;

    // 이동평균선 추가
    const ma5Series = chart.addLineSeries({
      color: '#fbbf24',
      lineWidth: 1,
      title: 'MA5',
      priceScaleId: 'right',
    });
    ma5SeriesRef.current = ma5Series;

    const ma20Series = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      title: 'MA20',
      priceScaleId: 'right',
    });
    ma20SeriesRef.current = ma20Series;

    const ma50Series = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 2,
      title: 'MA50',
      priceScaleId: 'right',
    });
    ma50SeriesRef.current = ma50Series;

    // 지지선/저항선 추가
    const supportSeries = chart.addLineSeries({
      color: '#10b981',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      title: '지지선',
      priceScaleId: 'right',
    });
    supportSeriesRef.current = supportSeries;

    const resistanceSeries = chart.addLineSeries({
      color: '#ef4444',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      title: '저항선',
      priceScaleId: 'right',
    });
    resistanceSeriesRef.current = resistanceSeries;

    // 피보나치 되돌림 레벨 추가
    const fibColors = {
      fib_0: '#ffffff',
      fib_236: '#f59e0b',
      fib_382: '#3b82f6',
      fib_500: '#8b5cf6',
      fib_618: '#ec4899',
      fib_786: '#10b981',
      fib_100: '#ffffff',
    };

    Object.keys(fibSeriesRefs).forEach(key => {
      const series = chart.addLineSeries({
        color: fibColors[key] || '#6b7280',
        lineWidth: 1,
        lineStyle: 1, // Dotted
        title: key.replace('fib_', '') + '%',
        priceScaleId: 'right',
      });
      fibSeriesRefs[key].current = series;
    });

    // 추세선 추가
    const uptrendSeries = chart.addLineSeries({
      color: '#10b981',
      lineWidth: 2,
      lineStyle: 0, // Solid
      title: '상승 추세선',
      priceScaleId: 'right',
    });
    uptrendSeriesRef.current = uptrendSeries;

    const downtrendSeries = chart.addLineSeries({
      color: '#ef4444',
      lineWidth: 2,
      lineStyle: 0, // Solid
      title: '하락 추세선',
      priceScaleId: 'right',
    });
    downtrendSeriesRef.current = downtrendSeries;

    // 볼린저 밴드 추가
    const bollingerUpper = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 1,
      lineStyle: 1, // Dotted
      title: '볼린저 상단',
      priceScaleId: 'right',
    });
    bollingerUpperRef.current = bollingerUpper;

    const bollingerMiddle = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      title: '볼린저 중간',
      priceScaleId: 'right',
    });
    bollingerMiddleRef.current = bollingerMiddle;

    const bollingerLower = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 1,
      lineStyle: 1, // Dotted
      title: '볼린저 하단',
      priceScaleId: 'right',
    });
    bollingerLowerRef.current = bollingerLower;

    // 리사이즈 핸들러
    const handleResize = () => {
      if (chartContainerRef.current && volumeContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        volumeChart.applyOptions({ width: volumeContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      volumeChart.remove();
    };
  }, []);

  // 가격 데이터 업데이트
  useEffect(() => {
    if (candlestickSeriesRef.current && priceData.length > 0) {
      // 유효한 데이터만 필터링
      const validData = priceData.filter(d => 
        d && 
        d.time && 
        d.open != null && 
        d.high != null && 
        d.low != null && 
        d.close != null &&
        !isNaN(d.time) &&
        !isNaN(d.open) &&
        !isNaN(d.high) &&
        !isNaN(d.low) &&
        !isNaN(d.close)
      );
      
      if (validData.length > 0) {
        try {
          candlestickSeriesRef.current.setData(validData);
          
          // 차트 자동 스크롤 (최신 데이터로, 우측 여백 유지)
          if (chartRef.current) {
            setTimeout(() => {
              try {
                const timeScale = chartRef.current.timeScale();
                const visibleRange = timeScale.getVisibleRange();
                if (visibleRange) {
                  // 우측에 여백을 두기 위해 약간 왼쪽으로 스크롤
                  const range = visibleRange.to - visibleRange.from;
                  const lastTime = validData[validData.length - 1].time;
                  timeScale.setVisibleRange({
                    from: lastTime - range * 0.85, // 우측에 15% 여백
                    to: lastTime + range * 0.15,   // 우측에 15% 여백
                  });
                } else {
                  // 처음 로드 시 전체 범위 표시 (우측 여백 포함)
                  const firstTime = validData[0].time;
                  const lastTime = validData[validData.length - 1].time;
                  const range = lastTime - firstTime;
                  timeScale.setVisibleRange({
                    from: firstTime - range * 0.1, // 좌측 10% 여백
                    to: lastTime + range * 0.3,   // 우측 30% 여백 (대폭 증가)
                  });
                }
              } catch (e) {
                console.warn('차트 스크롤 오류:', e);
              }
            }, 100);
          }
        } catch (e) {
          console.error('가격 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData]);

  // 볼륨 데이터 업데이트
  useEffect(() => {
    if (volumeSeriesRef.current && priceData.length > 0) {
      // 유효한 데이터만 필터링
      const validData = priceData
        .filter(d => 
          d && 
          d.time && 
          d.volume != null &&
          !isNaN(d.time) &&
          !isNaN(d.volume)
        )
        .map(d => ({
          time: d.time,
          value: d.volume || 0,
          color: (d.close >= d.open) ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)',
        }));
      
      if (validData.length > 0) {
        try {
          volumeSeriesRef.current.setData(validData);
          
          // 거래량 차트 자동 스크롤 (가격 차트와 동기화)
          if (volumeChartRef.current && chartRef.current) {
            setTimeout(() => {
              try {
                const priceTimeScale = chartRef.current.timeScale();
                const visibleRange = priceTimeScale.getVisibleRange();
                if (visibleRange) {
                  volumeChartRef.current.timeScale().setVisibleRange(visibleRange);
                }
              } catch (e) {
                console.warn('거래량 차트 스크롤 오류:', e);
              }
            }, 100);
          }
        } catch (e) {
          console.error('거래량 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData]);

  // 이동평균선 업데이트
  useEffect(() => {
    if (priceData.length > 0 && technicalIndicators) {
      // 유효한 데이터만 필터링
      const validData = priceData.filter(d => d && d.time && !isNaN(d.time));
      
      if (validData.length > 0) {
        try {
          if (ma5SeriesRef.current && visibility.ma5 && technicalIndicators.ma5 != null && !isNaN(technicalIndicators.ma5)) {
            const ma5Data = validData.map(d => ({ time: d.time, value: technicalIndicators.ma5 }));
            ma5SeriesRef.current.setData(ma5Data);
          } else if (ma5SeriesRef.current && !visibility.ma5) {
            ma5SeriesRef.current.setData([]);
          }
          if (ma20SeriesRef.current && visibility.ma20 && technicalIndicators.ma20 != null && !isNaN(technicalIndicators.ma20)) {
            const ma20Data = validData.map(d => ({ time: d.time, value: technicalIndicators.ma20 }));
            ma20SeriesRef.current.setData(ma20Data);
          } else if (ma20SeriesRef.current && !visibility.ma20) {
            ma20SeriesRef.current.setData([]);
          }
          if (ma50SeriesRef.current && visibility.ma50 && technicalIndicators.ma50 != null && !isNaN(technicalIndicators.ma50)) {
            const ma50Data = validData.map(d => ({ time: d.time, value: technicalIndicators.ma50 }));
            ma50SeriesRef.current.setData(ma50Data);
          } else if (ma50SeriesRef.current && !visibility.ma50) {
            ma50SeriesRef.current.setData([]);
          }
        } catch (e) {
          console.error('이동평균선 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData, technicalIndicators, visibility]);

  // 지지선/저항선 업데이트 (시간별로 변동)
  useEffect(() => {
    if (priceData.length > 0 && supportResistance) {
      // 유효한 데이터만 필터링
      const validData = priceData.filter(d => d && d.time && !isNaN(d.time));
      
      if (validData.length > 0) {
        try {
          // 시간별로 변동하는 지지선/저항선
          if (supportSeriesRef.current) {
            if (visibility.support) {
              if (supportResistance.support_levels && 
                  Array.isArray(supportResistance.support_levels) &&
                  supportResistance.support_levels.length === validData.length) {
                // 시간별 지지선 레벨이 있는 경우
                const supportData = validData.map((d, idx) => ({
                  time: d.time,
                  value: supportResistance.support_levels[idx]
                }));
                supportSeriesRef.current.setData(supportData);
              } else if (supportResistance.current_support != null && 
                         !isNaN(supportResistance.current_support)) {
                // 고정 지지선 (하위 호환성)
                const supportData = validData.map(d => ({
                  time: d.time,
                  value: supportResistance.current_support
                }));
                supportSeriesRef.current.setData(supportData);
              }
            } else {
              supportSeriesRef.current.setData([]);
            }
          }
          
          if (resistanceSeriesRef.current) {
            if (visibility.resistance) {
              if (supportResistance.resistance_levels && 
                  Array.isArray(supportResistance.resistance_levels) &&
                  supportResistance.resistance_levels.length === validData.length) {
                // 시간별 저항선 레벨이 있는 경우
                const resistanceData = validData.map((d, idx) => ({
                  time: d.time,
                  value: supportResistance.resistance_levels[idx]
                }));
                resistanceSeriesRef.current.setData(resistanceData);
              } else if (supportResistance.current_resistance != null && 
                         !isNaN(supportResistance.current_resistance)) {
                // 고정 저항선 (하위 호환성)
                const resistanceData = validData.map(d => ({
                  time: d.time,
                  value: supportResistance.current_resistance
                }));
                resistanceSeriesRef.current.setData(resistanceData);
              }
            } else {
              resistanceSeriesRef.current.setData([]);
            }
          }
        } catch (e) {
          console.error('지지선/저항선 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData, supportResistance, visibility]);

  // 피보나치 되돌림 업데이트
  useEffect(() => {
    if (priceData.length > 0 && fibonacci && Object.keys(fibonacci).length > 0) {
      const validData = priceData.filter(d => d && d.time && !isNaN(d.time));
      
      if (validData.length > 0) {
        try {
          // 피보나치 레벨 매핑
          const fibLevels = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100'];
          
          fibLevels.forEach(key => {
            const fibValue = fibonacci[key];
            if (fibSeriesRefs[key]?.current) {
              if (visibility.fibonacci && fibValue != null && !isNaN(fibValue)) {
                const fibData = validData.map(d => ({ time: d.time, value: fibValue }));
                fibSeriesRefs[key].current.setData(fibData);
              } else {
                fibSeriesRefs[key].current.setData([]);
              }
            }
          });
        } catch (e) {
          console.error('피보나치 되돌림 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData, fibonacci, visibility]);

  // 추세선 업데이트 (시간별로 변동)
  useEffect(() => {
    if (priceData.length > 0 && trendLines && Object.keys(trendLines).length > 0) {
      const validData = priceData.filter(d => d && d.time && !isNaN(d.time));
      
      if (validData.length >= 2) {
        try {
          // 상승 추세선 (빗각) - 시간별 가격 배열 사용
          if (uptrendSeriesRef.current) {
            if (visibility.uptrend && trendLines.uptrend) {
              if (trendLines.uptrend.prices && 
                  Array.isArray(trendLines.uptrend.prices) &&
                  trendLines.uptrend.prices.length === validData.length) {
                // 시간별 추세선 가격이 있는 경우 - None 값 필터링
                const uptrendData = validData
                  .map((d, idx) => {
                    const price = trendLines.uptrend.prices[idx];
                    if (price != null && !isNaN(price)) {
                      return { time: d.time, value: price };
                    }
                    return null;
                  })
                  .filter(d => d !== null);
                
                if (uptrendData.length > 0) {
                  uptrendSeriesRef.current.setData(uptrendData);
                } else {
                  uptrendSeriesRef.current.setData([]);
                }
              } else if (trendLines.uptrend.start_price != null && 
                         trendLines.uptrend.end_price != null) {
                // 하위 호환성: 시작점과 끝점만 있는 경우
                const { start_price, end_price } = trendLines.uptrend;
                const uptrendData = validData.map((d, idx) => {
                  const ratio = idx / (validData.length - 1);
                  const price = start_price + (end_price - start_price) * ratio;
                  return { time: d.time, value: price };
                });
                uptrendSeriesRef.current.setData(uptrendData);
              } else {
                uptrendSeriesRef.current.setData([]);
              }
            } else {
              uptrendSeriesRef.current.setData([]);
            }
          }
          
          // 하락 추세선 (엇각) - 시간별 가격 배열 사용
          if (downtrendSeriesRef.current) {
            if (visibility.downtrend && trendLines.downtrend) {
              if (trendLines.downtrend.prices && 
                  Array.isArray(trendLines.downtrend.prices) &&
                  trendLines.downtrend.prices.length === validData.length) {
                // 시간별 추세선 가격이 있는 경우 - None 값 필터링
                const downtrendData = validData
                  .map((d, idx) => {
                    const price = trendLines.downtrend.prices[idx];
                    if (price != null && !isNaN(price)) {
                      return { time: d.time, value: price };
                    }
                    return null;
                  })
                  .filter(d => d !== null);
                
                if (downtrendData.length > 0) {
                  downtrendSeriesRef.current.setData(downtrendData);
                } else {
                  downtrendSeriesRef.current.setData([]);
                }
              } else if (trendLines.downtrend.start_price != null && 
                         trendLines.downtrend.end_price != null) {
                // 하위 호환성: 시작점과 끝점만 있는 경우
                const { start_price, end_price } = trendLines.downtrend;
                const downtrendData = validData.map((d, idx) => {
                  const ratio = idx / (validData.length - 1);
                  const price = start_price + (end_price - start_price) * ratio;
                  return { time: d.time, value: price };
                });
                downtrendSeriesRef.current.setData(downtrendData);
              } else {
                downtrendSeriesRef.current.setData([]);
              }
            } else {
              downtrendSeriesRef.current.setData([]);
            }
          }
        } catch (e) {
          console.error('추세선 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData, trendLines, visibility]);

  // 볼린저 밴드 업데이트
  useEffect(() => {
    if (priceData.length > 0 && technicalIndicators) {
      const validData = priceData.filter(d => d && d.time && !isNaN(d.time));
      
      if (validData.length > 0) {
        try {
          if (visibility.bollinger) {
            // 볼린저 상단
            if (bollingerUpperRef.current && 
                technicalIndicators.bollinger_upper != null && 
                !isNaN(technicalIndicators.bollinger_upper)) {
              const upperData = validData.map(d => ({
                time: d.time,
                value: technicalIndicators.bollinger_upper
              }));
              bollingerUpperRef.current.setData(upperData);
            }
            
            // 볼린저 중간
            if (bollingerMiddleRef.current && 
                technicalIndicators.bollinger_middle != null && 
                !isNaN(technicalIndicators.bollinger_middle)) {
              const middleData = validData.map(d => ({
                time: d.time,
                value: technicalIndicators.bollinger_middle
              }));
              bollingerMiddleRef.current.setData(middleData);
            }
            
            // 볼린저 하단
            if (bollingerLowerRef.current && 
                technicalIndicators.bollinger_lower != null && 
                !isNaN(technicalIndicators.bollinger_lower)) {
              const lowerData = validData.map(d => ({
                time: d.time,
                value: technicalIndicators.bollinger_lower
              }));
              bollingerLowerRef.current.setData(lowerData);
            }
          } else {
            // 볼린저 밴드 숨김
            if (bollingerUpperRef.current) {
              bollingerUpperRef.current.setData([]);
            }
            if (bollingerMiddleRef.current) {
              bollingerMiddleRef.current.setData([]);
            }
            if (bollingerLowerRef.current) {
              bollingerLowerRef.current.setData([]);
            }
          }
        } catch (e) {
          console.error('볼린저 밴드 데이터 설정 오류:', e);
        }
      }
    }
  }, [priceData, technicalIndicators, visibility]);

  // 포지션 마커 추가
  useEffect(() => {
    if (positionData && candlestickSeriesRef.current && priceData.length > 0) {
      const newMarkers = [];
      
      if (positionData.entry_price) {
        // 포지션 진입 시점 찾기 (가장 가까운 시간)
        const entryTime = new Date(positionData.timestamp).getTime() / 1000;
        const closestData = priceData.reduce((prev, curr) => 
          Math.abs(curr.time - entryTime) < Math.abs(prev.time - entryTime) ? curr : prev
        );
        
        newMarkers.push({
          time: closestData.time,
          position: positionData.side === 'long' ? 'belowBar' : 'aboveBar',
          color: positionData.side === 'long' ? '#10b981' : '#ef4444',
          shape: positionData.side === 'long' ? 'arrowUp' : 'arrowDown',
          text: `${positionData.side.toUpperCase()} @ $${positionData.entry_price.toFixed(2)}`,
        });
      }

      setMarkers(newMarkers);
      candlestickSeriesRef.current.setMarkers(newMarkers);
    }
  }, [positionData, priceData]);

  const handleZoomIn = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const visibleRange = timeScale.getVisibleRange();
      if (visibleRange) {
        const range = visibleRange.to - visibleRange.from;
        const center = (visibleRange.from + visibleRange.to) / 2;
        const newRange = range * 0.7; // 30% 축소
        timeScale.setVisibleRange({
          from: center - newRange / 2,
          to: center + newRange / 2,
        });
      }
    }
  };

  const handleZoomOut = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const visibleRange = timeScale.getVisibleRange();
      if (visibleRange) {
        const range = visibleRange.to - visibleRange.from;
        const center = (visibleRange.from + visibleRange.to) / 2;
        const newRange = range * 1.4; // 40% 확대
        timeScale.setVisibleRange({
          from: center - newRange / 2,
          to: center + newRange / 2,
        });
      }
    }
  };

  const handleResetZoom = () => {
    if (chartRef.current && priceData.length > 0) {
      const timeScale = chartRef.current.timeScale();
      const firstTime = priceData[0].time;
      const lastTime = priceData[priceData.length - 1].time;
      // 우측에 여백을 두기 위해 약간 더 넓게 설정
                  const range = lastTime - firstTime;
                  timeScale.setVisibleRange({
                    from: firstTime - range * 0.1, // 좌측 10% 여백
                    to: lastTime + range * 0.3,   // 우측 30% 여백 (대폭 증가)
                  });
    }
  };

  return (
    <div className="trading-chart">
      <div className="chart-header">
        <h2>BTC/USDT 실시간 차트 (24시간)</h2>
        <div className="chart-controls">
          {technicalIndicators?.golden_cross && (
            <span className="signal-badge golden-cross">골든크로스</span>
          )}
          {technicalIndicators?.dead_cross && (
            <span className="signal-badge dead-cross">데드크로스</span>
          )}
          <div className="zoom-controls">
            <button onClick={handleZoomIn} className="zoom-btn" title="확대">
              <span>+</span>
            </button>
            <button onClick={handleZoomOut} className="zoom-btn" title="축소">
              <span>−</span>
            </button>
            <button onClick={handleResetZoom} className="zoom-btn" title="리셋">
              <span>⟲</span>
            </button>
          </div>
        </div>
      </div>
      <div className="price-chart-wrapper">
        <div ref={chartContainerRef} className="chart-container price-chart" />
      </div>
      <div className="volume-chart-wrapper">
        <div className="volume-chart-label">거래량</div>
        <div ref={volumeContainerRef} className="chart-container volume-chart" />
      </div>
      
      {/* 시장지표 및 기술적지표 - App.js의 차트 하단으로 이동하여 제거 */}
      {/* <div className="indicators-section">
        <div className="indicators-row">
          {marketIndicators && (
            <div className="indicator-panel market-panel">
              <div className="panel-header">
                <span className="panel-icon">📊</span>
                <h4>시장 지표</h4>
              </div>
              <div className="indicator-content">
                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">📈</span>
                    <span className="indicator-label">오더북</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value" style={{ 
                      color: marketIndicators.orderbook?.strength === 'strong_buy' ? '#10b981' : 
                             marketIndicators.orderbook?.strength === 'buy' ? '#34d399' :
                             marketIndicators.orderbook?.strength === 'strong_sell' ? '#ef4444' : 
                             marketIndicators.orderbook?.strength === 'sell' ? '#f87171' : '#6b7280',
                      fontWeight: '600'
                    }}>
                      {marketIndicators.orderbook?.strength === 'strong_buy' ? '강한 매수' :
                       marketIndicators.orderbook?.strength === 'buy' ? '매수' :
                       marketIndicators.orderbook?.strength === 'strong_sell' ? '강한 매도' :
                       marketIndicators.orderbook?.strength === 'sell' ? '매도' : '중립'}
                    </span>
                    <span className="indicator-detail">
                      {marketIndicators.orderbook?.ratio?.toFixed(2) || '0.00'}%
                    </span>
                  </div>
                </div>

                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">💥</span>
                    <span className="indicator-label">청산 클러스터</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value" style={{ 
                      color: marketIndicators.liquidation?.strength === 'strong' ? '#ef4444' : '#6b7280',
                      fontWeight: '600'
                    }}>
                      {marketIndicators.liquidation?.strength === 'strong' ? '강함' : '중립'}
                    </span>
                    <span className="indicator-detail">
                      {marketIndicators.liquidation?.ratio?.toFixed(2) || '0.00'}%
                    </span>
                  </div>
                </div>

                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">📉</span>
                    <span className="indicator-label">변동성</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value">
                      {marketIndicators.volatility?.status === 'squeeze' ? '압축' :
                       marketIndicators.volatility?.status === 'expansion' ? '확장' : '정상'}
                    </span>
                    <span className="indicator-detail">
                      {marketIndicators.volatility?.expansion_potential || 'low'}
                    </span>
                  </div>
                </div>

                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">💰</span>
                    <span className="indicator-label">OI (미체결약정)</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value">
                      {marketIndicators.oi?.status === 'surge' ? '급증' :
                       marketIndicators.oi?.status === 'decline' ? '감소' : '정상'}
                    </span>
                    <span className="indicator-detail">
                      펀딩: {marketIndicators.oi?.funding_rate?.toFixed(4) || '0.0000'}%
                    </span>
                  </div>
                </div>

                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">🔄</span>
                    <span className="indicator-label">CVD</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value" style={{ 
                      color: marketIndicators.cvd?.trend === 'bullish' ? '#10b981' : 
                             marketIndicators.cvd?.trend === 'bearish' ? '#ef4444' : '#6b7280',
                      fontWeight: '600'
                    }}>
                      {marketIndicators.cvd?.trend === 'bullish' ? '상승' :
                       marketIndicators.cvd?.trend === 'bearish' ? '하락' : '중립'}
                    </span>
                    <span className="indicator-detail">
                      {marketIndicators.cvd?.turnover ? '전환 ✓' : '전환 ✗'}
                    </span>
                  </div>
                </div>

                <div className="indicator-item summary-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">🎯</span>
                    <span className="indicator-label">종합 신호</span>
                  </div>
                  <div className="indicator-value-group">
                    <span className="indicator-value summary-value" style={{ 
                      color: marketIndicators.signal === 'buy' ? '#10b981' : 
                             marketIndicators.signal === 'sell' ? '#ef4444' : '#6b7280',
                      fontWeight: 'bold',
                      fontSize: '16px'
                    }}>
                      {marketIndicators.signal === 'buy' ? '매수' :
                       marketIndicators.signal === 'sell' ? '매도' : '중립'}
                    </span>
                    <span className="indicator-detail summary-confidence">
                      신뢰도: {marketIndicators.confidence?.toFixed(1) || '0.0'}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {technicalIndicators && Object.keys(technicalIndicators).length > 0 && (
            <div className="indicator-panel technical-panel">
              <div className="panel-header">
                <span className="panel-icon">📈</span>
                <h4>기술적 지표</h4>
              </div>
              <div className="indicator-content">
                <div className="indicator-item">
                  <div className="indicator-header">
                    <span className="indicator-icon">📊</span>
                    <span className="indicator-label">이동평균선</span>
                  </div>
                  <div className="indicator-grid">
                    {technicalIndicators.ma5 && (
                      <div className="indicator-sub-item">
                        <span className="sub-label">MA5</span>
                        <span className="sub-value" style={{ color: '#fbbf24' }}>
                          ${technicalIndicators.ma5.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {technicalIndicators.ma20 && (
                      <div className="indicator-sub-item">
                        <span className="sub-label">MA20</span>
                        <span className="sub-value" style={{ color: '#3b82f6' }}>
                          ${technicalIndicators.ma20.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {technicalIndicators.ma50 && (
                      <div className="indicator-sub-item">
                        <span className="sub-label">MA50</span>
                        <span className="sub-value" style={{ color: '#8b5cf6' }}>
                          ${technicalIndicators.ma50.toFixed(2)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {technicalIndicators.rsi != null && (
                  <div className="indicator-item">
                    <div className="indicator-header">
                      <span className="indicator-icon">📉</span>
                      <span className="indicator-label">RSI</span>
                    </div>
                    <div className="indicator-value-group">
                      <span className="indicator-value" style={{ 
                        color: technicalIndicators.rsi >= 70 ? '#ef4444' : 
                               technicalIndicators.rsi <= 30 ? '#10b981' : '#6b7280',
                        fontWeight: 'bold',
                        fontSize: '18px'
                      }}>
                        {technicalIndicators.rsi.toFixed(2)}
                      </span>
                      <span className="indicator-detail" style={{ 
                        color: technicalIndicators.rsi >= 70 ? '#ef4444' : 
                               technicalIndicators.rsi <= 30 ? '#10b981' : '#6b7280'
                      }}>
                        {technicalIndicators.rsi >= 70 ? '과매수' : 
                         technicalIndicators.rsi <= 30 ? '과매도' : '중립'}
                      </span>
                    </div>
                  </div>
                )}

                {technicalIndicators.bollinger_upper && technicalIndicators.bollinger_lower && (
                  <div className="indicator-item">
                    <div className="indicator-header">
                      <span className="indicator-icon">📊</span>
                      <span className="indicator-label">볼린저 밴드</span>
                    </div>
                    <div className="indicator-grid">
                      <div className="indicator-sub-item">
                        <span className="sub-label">상단</span>
                        <span className="sub-value" style={{ color: '#3b82f6' }}>
                          ${technicalIndicators.bollinger_upper.toFixed(2)}
                        </span>
                      </div>
                      {technicalIndicators.bollinger_middle && (
                        <div className="indicator-sub-item">
                          <span className="sub-label">중간</span>
                          <span className="sub-value" style={{ color: '#8b5cf6' }}>
                            ${technicalIndicators.bollinger_middle.toFixed(2)}
                          </span>
                        </div>
                      )}
                      <div className="indicator-sub-item">
                        <span className="sub-label">하단</span>
                        <span className="sub-value" style={{ color: '#3b82f6' }}>
                          ${technicalIndicators.bollinger_lower.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {(technicalIndicators.golden_cross || technicalIndicators.dead_cross) && (
                  <div className="indicator-item signal-item">
                    <div className="indicator-header">
                      <span className="indicator-icon">⚡</span>
                      <span className="indicator-label">신호</span>
                    </div>
                    <div className="signal-badges">
                      {technicalIndicators.golden_cross && (
                        <div className="signal-badge golden-cross-badge">
                          <span className="signal-icon">📈</span>
                          <span className="signal-text">골든크로스</span>
                        </div>
                      )}
                      {technicalIndicators.dead_cross && (
                        <div className="signal-badge dead-cross-badge">
                          <span className="signal-icon">📉</span>
                          <span className="signal-text">데드크로스</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div> */}
    </div>
  );
};

export default TradingChart;

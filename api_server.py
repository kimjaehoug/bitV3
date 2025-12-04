"""
백엔드 API 서버
실시간 거래 데이터를 프론트엔드에 제공
"""
import os
import time
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import requests
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Google Generative AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠️ google-generativeai가 설치되지 않았습니다. 'pip install google-generativeai'를 실행하세요.")
    GENAI_AVAILABLE = False

# 환경변수 로드 (.env 파일 지원)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않았습니다. .env 파일을 사용하려면 'pip install python-dotenv'를 실행하세요.")

from realtime_trading import RealtimeTradingSignal, RealtimeTrader
from data_fetcher import BinanceDataFetcher

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 변수
signal_generator = None
trader = None
price_history = []  # 가격 히스토리
prediction_history = []  # 예측 히스토리
position_history = []  # 포지션 히스토리
is_running = False
update_thread = None
gemini_conversations = {}  # Gemini 대화 히스토리 (세션별)
last_broadcasted_ai_analysis = None  # 마지막으로 브로드캐스트한 AI 분석 결과

# 최대 히스토리 크기
MAX_HISTORY_SIZE = 1000
# 차트에 표시할 데이터 기간 (24시간 = 288개 5분봉)
CHART_DATA_HOURS = 24


def calculate_support_resistance(df: pd.DataFrame, df_1h: pd.DataFrame = None, window: int = 20) -> Dict:
    """지지선/저항선 계산 (1시간 추세 기반, 시간에 따라 변동)"""
    try:
        # 1시간봉 데이터가 있으면 사용, 없으면 5분봉 데이터 사용
        if df_1h is not None and len(df_1h) >= 12:  # 최소 12시간 데이터 필요
            # 1시간봉 데이터로 지지선/저항선 계산 (12시간 = 12개 1시간봉)
            trend_df = df_1h
            window_1h = min(12, len(trend_df))  # 12시간 윈도우
        else:
            # 5분봉 데이터 사용 (더 긴 기간)
            trend_df = df
            window_1h = min(144, len(trend_df))  # 12시간 = 144개 5분봉
        
        if len(trend_df) < window_1h:
            return {'support_levels': None, 'resistance_levels': None, 'current_support': None, 'current_resistance': None}
        
        # 1시간 추세 기반으로 지지선/저항선 계산
        recent_data = trend_df.tail(window_1h)
        lows = recent_data['low'].values
        highs = recent_data['high'].values
        
        # 지지선: 최근 저점들의 평균 (최근 저점 5개)
        if len(lows) >= 5:
            support_base = float(np.mean(sorted(lows)[:5]))
        else:
            support_base = float(np.mean(lows))
        
        # 저항선: 최근 고점들의 평균 (최근 고점 5개)
        if len(highs) >= 5:
            resistance_base = float(np.mean(sorted(highs, reverse=True)[:5]))
        else:
            resistance_base = float(np.mean(highs))
        
        # 5분봉 차트에 맞춰 각 시점마다 지지선/저항선 값 생성
        # 1시간 추세는 상대적으로 안정적이므로, 각 5분봉 시점에 동일한 값 사용
        support_levels = [support_base] * len(df)
        resistance_levels = [resistance_base] * len(df)
        
        # 최근 값들 반환 (시간별로 변동하는 값)
        return {
            'support_levels': [float(x) if x is not None else None for x in support_levels],
            'resistance_levels': [float(x) if x is not None else None for x in resistance_levels],
            'current_support': float(support_base) if support_base is not None else None,
            'current_resistance': float(resistance_base) if resistance_base is not None else None
        }
    except Exception as e:
        print(f"지지선/저항선 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'support_levels': None, 'resistance_levels': None, 'current_support': None, 'current_resistance': None}


def calculate_fibonacci_retracement(df: pd.DataFrame) -> Dict:
    """피보나치 되돌림 계산"""
    try:
        if len(df) < 20:
            return {}
        
        recent_data = df.tail(50)  # 최근 50개 캔들 사용
        
        # 최고가와 최저가 찾기
        high_price = float(recent_data['high'].max())
        low_price = float(recent_data['low'].min())
        
        # 가격 차이
        price_range = high_price - low_price
        
        # 피보나치 되돌림 레벨 (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%)
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # 현재 가격
        current_price = float(recent_data['close'].iloc[-1])
        
        # 추세 방향 판단 (최근 가격이 상승 추세인지 하락 추세인지)
        is_uptrend = current_price > recent_data['close'].iloc[-10:].mean()
        
        if is_uptrend:
            # 상승 추세: 최저가에서 최고가로
            base_price = low_price
            fib_prices = {f'fib_{int(level*100)}': float(base_price + price_range * level) 
                         for level in fib_levels}
        else:
            # 하락 추세: 최고가에서 최저가로
            base_price = high_price
            fib_prices = {f'fib_{int(level*100)}': float(base_price - price_range * level) 
                         for level in fib_levels}
        
        return {
            'high': high_price,
            'low': low_price,
            'current': current_price,
            'trend': 'up' if is_uptrend else 'down',
            **fib_prices
        }
    except Exception as e:
        print(f"피보나치 되돌림 계산 오류: {e}")
        return {}


def calculate_trend_lines(df: pd.DataFrame, df_1h: pd.DataFrame = None) -> Dict:
    """추세선 계산 (1시간 추세 기반, 빗각/엇각) - 시간별로 변동"""
    try:
        # 1시간봉 데이터가 있으면 사용, 없으면 5분봉 데이터 사용
        if df_1h is not None and len(df_1h) >= 24:  # 최소 24시간 데이터 필요
            # 1시간봉 데이터로 추세선 계산
            trend_df = df_1h
            window_size = min(24, len(trend_df))  # 24시간 윈도우
        else:
            # 5분봉 데이터 사용 (더 긴 기간)
            trend_df = df
            window_size = min(288, len(trend_df))  # 24시간 = 288개 5분봉
        
        if len(trend_df) < 20:
            return {}
        
        # 1시간 추세 기반으로 추세선 계산
        recent_data = trend_df.tail(window_size).copy()
        high_prices = recent_data['high'].values
        low_prices = recent_data['low'].values
        
        # 최근 고점과 저점 찾기 (더 정확한 방법)
        # 최소 3개 이상의 캔들에서 고점/저점 확인
        recent_highs = []
        recent_lows = []
        
        # 고점/저점 찾기 (조건 완화)
        lookback = 2  # 전후 2개 캔들과 비교 (3 -> 2로 완화)
        for i in range(lookback, len(recent_data) - lookback):
            # 고점: 전후 lookback개보다 모두 높은 경우
            is_high = True
            is_low = True
            for j in range(1, lookback + 1):
                if high_prices[i] < high_prices[i-j] or high_prices[i] < high_prices[i+j]:
                    is_high = False
                if low_prices[i] > low_prices[i-j] or low_prices[i] > low_prices[i+j]:
                    is_low = False
                if not is_high and not is_low:
                    break
            
            if is_high:
                recent_highs.append((i, high_prices[i]))
            if is_low:
                recent_lows.append((i, low_prices[i]))
        
        print(f"🔍 추세선 계산: 고점 {len(recent_highs)}개, 저점 {len(recent_lows)}개 발견")
        
        # 상승 추세선 (저점들을 연결) - 유효성 검증 포함
        uptrend_line = None
        print(f"📊 상승 추세선 계산 시작: 저점 {len(recent_lows)}개")
        if len(recent_lows) >= 2:
            # 여러 저점 중에서 가장 의미있는 추세선 찾기
            # 최근 저점들이 상승 추세를 보이는지 확인
            valid_trends = []
            
            # 최근 3-4개의 저점을 조합하여 추세선 후보 생성
            for i in range(max(0, len(recent_lows) - 4), len(recent_lows) - 1):
                for j in range(i + 1, len(recent_lows)):
                    point1 = recent_lows[i]
                    point2 = recent_lows[j]
                    
                    if point2[0] > point1[0]:  # 시간 순서 확인
                        # 상승 추세선: 두 번째 저점이 첫 번째 저점보다 높아야 함
                        if point2[1] > point1[1]:
                            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                            
                            # 추세선의 각도가 너무 가파르거나 평평하지 않은지 확인
                            # 각도가 -45도 ~ 45도 사이여야 의미있음 (너무 가파르면 무의미)
                            price_range = recent_data['close'].max() - recent_data['close'].min()
                            if price_range > 0:
                                slope_ratio = abs(slope) * (point2[0] - point1[0]) / price_range
                                # slope_ratio 조건 완화: 0.01 ~ 5.0 사이면 합리적
                                if 0.01 <= slope_ratio <= 5.0:
                                    # 추세선이 다른 저점들과도 잘 맞는지 확인 (터치 횟수)
                                    touch_count = 2  # point1, point2
                                    for k in range(len(recent_lows)):
                                        if k != i and k != j:
                                            low_idx, low_price = recent_lows[k]
                                            # 추세선에서 예상되는 가격
                                            expected_price = point1[1] + slope * (low_idx - point1[0])
                                            # 실제 저점과의 차이가 3% 이내면 터치로 간주 (2% -> 3%로 완화)
                                            if abs(low_price - expected_price) / expected_price < 0.03:
                                                touch_count += 1
                                    
                                    # 최소 2개 터치 (항상 만족)
                                    valid_trends.append({
                                        'point1': point1,
                                        'point2': point2,
                                        'slope': slope,
                                        'touch_count': touch_count,
                                        'slope_ratio': slope_ratio
                                    })
            
            # 가장 많은 터치를 가진 추세선 선택, 동일하면 최근 것 선택
            if valid_trends:
                best_trend = max(valid_trends, key=lambda x: (x['touch_count'], x['point2'][0]))
                point1 = best_trend['point1']
                point2 = best_trend['point2']
                slope = best_trend['slope']
                # 5분봉 차트에 맞춰 각 시점마다 추세선 가격 계산
                uptrend_prices = []
                
                # 1시간봉 데이터를 사용하는 경우 시간 기반 매핑
                if df_1h is not None and len(df_1h) >= 24:
                    # 1시간봉 인덱스를 시간으로 변환
                    point1_time = recent_data.index[point1[0]]
                    point2_time = recent_data.index[point2[0]]
                    
                    # 각 5분봉 시점에 대해 추세선 가격 계산
                    for i in range(len(df)):
                        current_time = df.index[i]
                        
                        if current_time < point1_time:
                            # 추세선 시작 전
                            uptrend_prices.append(None)
                        else:
                            # 추세선 범위 내 또는 연장
                            # 1시간봉 인덱스 기준으로 계산
                            # point1_time부터 current_time까지의 시간 차이를 1시간봉 단위로 변환
                            time_diff = (current_time - point1_time).total_seconds() / 3600  # 시간 단위
                            # point1의 인덱스에서 time_diff만큼 더한 인덱스
                            trend_idx = point1[0] + time_diff
                            price = point1[1] + slope * (trend_idx - point1[0])
                            uptrend_prices.append(float(price))
                else:
                    # 5분봉 데이터 사용
                    recent_start_idx = len(df) - len(recent_data) if len(recent_data) <= len(df) else 0
                    
                    for i in range(len(df)):
                        if i < recent_start_idx + point1[0]:
                            # 추세선 시작 전에는 None
                            uptrend_prices.append(None)
                        else:
                            # 추세선 범위 내에서는 계산된 가격
                            relative_idx = i - recent_start_idx
                            if relative_idx < len(recent_data):
                                price = point1[1] + slope * (relative_idx - point1[0])
                                uptrend_prices.append(float(price))
                            else:
                                # 추세선 연장
                                price = point1[1] + slope * (len(recent_data) - 1 - point1[0])
                                uptrend_prices.append(float(price))
                
                uptrend_line = {
                    'prices': uptrend_prices,
                    'start_price': float(point1[1]),
                    'end_price': float(uptrend_prices[-1]) if uptrend_prices[-1] is not None else None,
                    'slope': float(slope),
                    'touch_count': best_trend['touch_count'],
                    'validity': 'high' if best_trend['touch_count'] >= 3 else 'medium'
                }
            else:
                # 유효한 추세선을 찾지 못함
                print(f"⚠️ 유효한 상승 추세선을 찾지 못함 (후보 {len(valid_trends)}개)")
                uptrend_line = None
        
        # 하락 추세선 (고점들을 연결) - 유효성 검증 포함
        downtrend_line = None
        print(f"📊 하락 추세선 계산 시작: 고점 {len(recent_highs)}개")
        if len(recent_highs) >= 2:
            # 여러 고점 중에서 가장 의미있는 추세선 찾기
            # 최근 고점들이 하락 추세를 보이는지 확인
            valid_trends = []
            
            # 최근 3-4개의 고점을 조합하여 추세선 후보 생성
            for i in range(max(0, len(recent_highs) - 4), len(recent_highs) - 1):
                for j in range(i + 1, len(recent_highs)):
                    point1 = recent_highs[i]
                    point2 = recent_highs[j]
                    
                    if point2[0] > point1[0]:  # 시간 순서 확인
                        # 하락 추세선: 두 번째 고점이 첫 번째 고점보다 낮아야 함
                        if point2[1] < point1[1]:
                            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                            
                            # 추세선의 각도가 너무 가파르거나 평평하지 않은지 확인
                            price_range = recent_data['close'].max() - recent_data['close'].min()
                            if price_range > 0:
                                slope_ratio = abs(slope) * (point2[0] - point1[0]) / price_range
                                # slope_ratio 조건 완화: 0.01 ~ 5.0 사이면 합리적
                                if 0.01 <= slope_ratio <= 5.0:
                                    # 추세선이 다른 고점들과도 잘 맞는지 확인 (터치 횟수)
                                    touch_count = 2  # point1, point2
                                    for k in range(len(recent_highs)):
                                        if k != i and k != j:
                                            high_idx, high_price = recent_highs[k]
                                            # 추세선에서 예상되는 가격
                                            expected_price = point1[1] + slope * (high_idx - point1[0])
                                            # 실제 고점과의 차이가 3% 이내면 터치로 간주 (2% -> 3%로 완화)
                                            if abs(high_price - expected_price) / expected_price < 0.03:
                                                touch_count += 1
                                    
                                    # 최소 2개 터치 (항상 만족)
                                    valid_trends.append({
                                        'point1': point1,
                                        'point2': point2,
                                        'slope': slope,
                                        'touch_count': touch_count,
                                        'slope_ratio': slope_ratio
                                    })
            
            # 가장 많은 터치를 가진 추세선 선택, 동일하면 최근 것 선택
            print(f"✅ 유효한 하락 추세선 후보: {len(valid_trends)}개")
            if valid_trends:
                best_trend = max(valid_trends, key=lambda x: (x['touch_count'], x['point2'][0]))
                point1 = best_trend['point1']
                point2 = best_trend['point2']
                slope = best_trend['slope']
                print(f"✅ 최적 하락 추세선 선택: 터치 {best_trend['touch_count']}개, 기울기 {slope:.6f}")
                # 5분봉 차트에 맞춰 각 시점마다 추세선 가격 계산
                downtrend_prices = []
                
                # 1시간봉 데이터를 사용하는 경우 시간 기반 매핑
                if df_1h is not None and len(df_1h) >= 24:
                    # 1시간봉 인덱스를 시간으로 변환
                    point1_time = recent_data.index[point1[0]]
                    point2_time = recent_data.index[point2[0]]
                    
                    # 각 5분봉 시점에 대해 추세선 가격 계산
                    for i in range(len(df)):
                        current_time = df.index[i]
                        
                        if current_time < point1_time:
                            # 추세선 시작 전
                            downtrend_prices.append(None)
                        else:
                            # 추세선 범위 내 또는 연장
                            # 1시간봉 인덱스 기준으로 계산
                            # point1_time부터 current_time까지의 시간 차이를 1시간봉 단위로 변환
                            time_diff = (current_time - point1_time).total_seconds() / 3600  # 시간 단위
                            # point1의 인덱스에서 time_diff만큼 더한 인덱스
                            trend_idx = point1[0] + time_diff
                            price = point1[1] + slope * (trend_idx - point1[0])
                            downtrend_prices.append(float(price))
                else:
                    # 5분봉 데이터 사용
                    recent_start_idx = len(df) - len(recent_data) if len(recent_data) <= len(df) else 0
                    
                    for i in range(len(df)):
                        if i < recent_start_idx + point1[0]:
                            # 추세선 시작 전에는 None
                            downtrend_prices.append(None)
                        else:
                            # 추세선 범위 내에서는 계산된 가격
                            relative_idx = i - recent_start_idx
                            if relative_idx < len(recent_data):
                                price = point1[1] + slope * (relative_idx - point1[0])
                                downtrend_prices.append(float(price))
                            else:
                                # 추세선 연장
                                price = point1[1] + slope * (len(recent_data) - 1 - point1[0])
                                downtrend_prices.append(float(price))
                
                downtrend_line = {
                    'prices': downtrend_prices,
                    'start_price': float(point1[1]),
                    'end_price': float(downtrend_prices[-1]) if downtrend_prices[-1] is not None else None,
                    'slope': float(slope),
                    'touch_count': best_trend['touch_count'],
                    'validity': 'high' if best_trend['touch_count'] >= 3 else 'medium'
                }
            else:
                # 유효한 추세선을 찾지 못함
                print(f"⚠️ 유효한 하락 추세선을 찾지 못함 (후보 {len(valid_trends)}개)")
                downtrend_line = None
        
        # Fallback: 유효한 추세선을 찾지 못한 경우 최소한의 추세선이라도 표시
        print(f"🔄 Fallback 체크: 상승 추세선={uptrend_line is not None}, 하락 추세선={downtrend_line is not None}")
        if uptrend_line is None and len(recent_lows) >= 2:
            print(f"🔄 Fallback: 상승 추세선 생성 시도 (저점 {len(recent_lows)}개)")
            point1 = recent_lows[-2]
            point2 = recent_lows[-1]
            if point2[0] > point1[0] and point2[1] > point1[1]:
                slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                uptrend_prices = []
                recent_start_idx = len(df) - len(recent_data) if len(recent_data) <= len(df) else 0
                for i in range(len(df)):
                    if i >= recent_start_idx + point1[0]:
                        relative_idx = i - recent_start_idx
                        if relative_idx < len(recent_data):
                            price = point1[1] + slope * (relative_idx - point1[0])
                            uptrend_prices.append(float(price))
                        else:
                            price = point1[1] + slope * (len(recent_data) - 1 - point1[0])
                            uptrend_prices.append(float(price))
                    else:
                        uptrend_prices.append(None)
                uptrend_line = {
                    'prices': uptrend_prices,
                    'start_price': float(point1[1]),
                    'end_price': float(uptrend_prices[-1]) if uptrend_prices[-1] is not None else None,
                    'slope': float(slope),
                    'touch_count': 2,
                    'validity': 'low'
                }
                print(f"✅ Fallback 상승 추세선 생성 완료")
        
        if downtrend_line is None and len(recent_highs) >= 2:
            print(f"🔄 Fallback: 하락 추세선 생성 시도 (고점 {len(recent_highs)}개)")
            point1 = recent_highs[-2]
            point2 = recent_highs[-1]
            if point2[0] > point1[0] and point2[1] < point1[1]:
                slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                downtrend_prices = []
                recent_start_idx = len(df) - len(recent_data) if len(recent_data) <= len(df) else 0
                for i in range(len(df)):
                    if i >= recent_start_idx + point1[0]:
                        relative_idx = i - recent_start_idx
                        if relative_idx < len(recent_data):
                            price = point1[1] + slope * (relative_idx - point1[0])
                            downtrend_prices.append(float(price))
                        else:
                            price = point1[1] + slope * (len(recent_data) - 1 - point1[0])
                            downtrend_prices.append(float(price))
                    else:
                        downtrend_prices.append(None)
                downtrend_line = {
                    'prices': downtrend_prices,
                    'start_price': float(point1[1]),
                    'end_price': float(downtrend_prices[-1]) if downtrend_prices[-1] is not None else None,
                    'slope': float(slope),
                    'touch_count': 2,
                    'validity': 'low'
                }
                print(f"✅ Fallback 하락 추세선 생성 완료")
        
        result = {
            'uptrend': uptrend_line,
            'downtrend': downtrend_line
        }
        print(f"📈 최종 추세선 결과: 상승={uptrend_line is not None}, 하락={downtrend_line is not None}")
        return result
    except Exception as e:
        print(f"추세선 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """기술적 지표 계산 (MA, 골든크로스 등)"""
    try:
        if len(df) < 50:
            return {}
        
        close = df['close']
        
        # 이동평균선
        ma5 = close.rolling(window=5).mean().iloc[-1] if len(df) >= 5 else None
        ma10 = close.rolling(window=10).mean().iloc[-1] if len(df) >= 10 else None
        ma20 = close.rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        ma50 = close.rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        
        # 골든크로스/데드크로스
        golden_cross = False
        dead_cross = False
        if ma5 is not None and ma20 is not None:
            # 골든크로스: 단기선이 장기선을 상향 돌파
            if len(df) >= 2:
                prev_ma5 = close.rolling(window=5).mean().iloc[-2]
                prev_ma20 = close.rolling(window=20).mean().iloc[-2]
                if prev_ma5 <= prev_ma20 and ma5 > ma20:
                    golden_cross = True
                elif prev_ma5 >= prev_ma20 and ma5 < ma20:
                    dead_cross = True
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        
        # 볼린저 밴드
        ma20_bb = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper_band = ma20_bb + (std20 * 2)
        lower_band = ma20_bb - (std20 * 2)
        
        return {
            'ma5': float(ma5) if ma5 is not None and not pd.isna(ma5) else None,
            'ma10': float(ma10) if ma10 is not None and not pd.isna(ma10) else None,
            'ma20': float(ma20) if ma20 is not None and not pd.isna(ma20) else None,
            'ma50': float(ma50) if ma50 is not None and not pd.isna(ma50) else None,
            'golden_cross': golden_cross,
            'dead_cross': dead_cross,
            'rsi': current_rsi,
            'bollinger_upper': float(upper_band.iloc[-1]) if len(upper_band) > 0 and not pd.isna(upper_band.iloc[-1]) else None,
            'bollinger_lower': float(lower_band.iloc[-1]) if len(lower_band) > 0 and not pd.isna(lower_band.iloc[-1]) else None,
            'bollinger_middle': float(ma20_bb.iloc[-1]) if len(ma20_bb) > 0 and not pd.isna(ma20_bb.iloc[-1]) else None,
        }
    except Exception as e:
        print(f"기술적 지표 계산 오류: {e}")
        return {}


def update_data_loop():
    """1분마다 데이터 업데이트 및 전송"""
    global is_running, signal_generator, price_history, prediction_history, position_history, trader, last_broadcasted_ai_analysis
    
    fetcher = BinanceDataFetcher()
    
    while is_running:
        try:
            timestamp = datetime.now()
            
            # 최근 24시간간의 5분봉 OHLCV 데이터 가져오기 (288개 캔들)
            ohlcv_data = fetcher.fetch_recent_data(hours=24, timeframe='5m')
            
            if len(ohlcv_data) == 0:
                print("OHLCV 데이터를 가져올 수 없습니다.")
                time.sleep(60)
                continue
            
            # 가격 히스토리를 OHLCV 데이터로 업데이트
            price_history = []
            for idx, row in ohlcv_data.iterrows():
                price_history.append({
                    'timestamp': idx.isoformat(),
                    'price': float(row['close']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'open': float(row['open']),
                    'volume': float(row['volume'])
                })
            
            # 현재 가격 (가장 최근 캔들의 종가)
            current_price = float(ohlcv_data.iloc[-1]['close'])
            
            # 예측 수행
            prediction_data = None
            if signal_generator is not None:
                try:
                    result = signal_generator.predict_and_signal()
                    
                    if result.get('success'):
                        # numpy 타입을 Python 기본 타입으로 변환
                        change_30m = result.get('change_30m', 0)
                        change_1h = result.get('change_1h', result.get('price_change_pct', 0))
                        confidence = result.get('confidence', 0)
                        
                        prediction_data = {
                            'timestamp': timestamp.isoformat(),
                            'current_price': float(result.get('current_price', current_price)),
                            'predicted_price_30m': float(result.get('predicted_price_30m', current_price)),
                            'predicted_price_1h': float(result.get('predicted_price', current_price)),
                            'change_30m': float(change_30m) * 100,
                            'change_1h': float(change_1h) * 100,
                            'signal': result.get('signal', 'hold'),
                            'confidence': float(confidence)
                        }
                        
                        prediction_history.append(prediction_data)
                        if len(prediction_history) > MAX_HISTORY_SIZE:
                            prediction_history.pop(0)
                except Exception as e:
                    print(f"예측 오류: {e}")
            
            # 포지션 정보 가져오기
            position_data = None
            if trader is not None:
                try:
                    position = trader.get_current_position()
                    if position:
                        position_data = {
                            'timestamp': timestamp.isoformat(),
                            'side': position.get('side'),
                            'entry_price': float(position.get('entry_price', 0)),
                            'size': float(position.get('size', 0)),
                            'unrealized_pnl': float(position.get('unrealized_pnl', 0)),
                            'mark_price': float(position.get('mark_price', 0)),
                            'percentage': float(position.get('percentage', 0))
                        }
                        position_history.append(position_data)
                        if len(position_history) > 100:
                            position_history.pop(0)
                except Exception as e:
                    print(f"포지션 정보 가져오기 오류: {e}")
            
            # 시장 지표 정보 가져오기
            market_indicators_data = None
            if trader is not None:
                try:
                    market_signal = trader.market_indicators.get_trading_signal_from_indicators()
                    indicators = market_signal.get('indicators', {})
                    
                    # 각 지표 정보 추출
                    ob = indicators.get('orderbook_imbalance', {})
                    lc = indicators.get('liquidation_clusters', {})
                    vs = indicators.get('volatility_squeeze', {})
                    oi = indicators.get('oi_surge', {})
                    cvd = indicators.get('cvd_turnover', {})
                    
                    market_indicators_data = {
                        'orderbook': {
                            'strength': ob.get('imbalance_strength', 'neutral'),
                            'ratio': float(ob.get('imbalance_ratio', 0)) * 100,
                            'spread_pct': float(ob.get('spread_pct', 0))
                        },
                        'liquidation': {
                            'strength': lc.get('liquidation_strength', 'neutral'),
                            'ratio': float(lc.get('liquidation_ratio', 0)) * 100
                        },
                        'volatility': {
                            'status': vs.get('squeeze_status', 'normal'),
                            'expansion_potential': vs.get('expansion_potential', 'low')
                        },
                        'oi': {
                            'status': oi.get('oi_surge_status', 'normal'),
                            'direction': oi.get('oi_direction', 'balanced'),
                            'funding_rate': float(oi.get('funding_rate_pct', 0))
                        },
                        'cvd': {
                            'trend': cvd.get('cvd_trend', 'neutral'),
                            'turnover': cvd.get('cvd_turnover', False)
                        },
                        'signal': market_signal.get('signal', 'neutral'),
                        'confidence': float(market_signal.get('confidence', 0)) * 100,
                        'reasons': market_signal.get('reasons', [])
                    }
                except Exception as e:
                    print(f"시장 지표 정보 가져오기 오류: {e}")
                    market_indicators_data = None
            
            # 기술적 지표 계산용으로 더 많은 데이터 필요 (24시간)
            ohlcv_24h = ohlcv_data.copy()
            
            # 1시간봉 데이터 가져오기 (1시간 추세 계산용)
            ohlcv_1h = None
            try:
                ohlcv_1h = fetcher.fetch_recent_data(hours=24, timeframe='1h')
                if len(ohlcv_1h) > 0:
                    print(f"1시간봉 데이터 수집 완료: {len(ohlcv_1h)}개")
            except Exception as e:
                print(f"1시간봉 데이터 수집 실패: {e}")
                ohlcv_1h = None
            
            # 기술적 지표 계산
            technical_indicators = calculate_technical_indicators(ohlcv_24h)
            support_resistance = calculate_support_resistance(ohlcv_24h, df_1h=ohlcv_1h)
            fibonacci = calculate_fibonacci_retracement(ohlcv_24h)
            trend_lines = calculate_trend_lines(ohlcv_24h, df_1h=ohlcv_1h)
            
            # 예측 모델 임계값 정보 추가
            threshold_info = {}
            if signal_generator:
                threshold_info = {
                    'current_threshold': float(signal_generator.min_confidence) if hasattr(signal_generator, 'min_confidence') else None,
                    'original_threshold': float(signal_generator.original_min_confidence) if hasattr(signal_generator, 'original_min_confidence') else None,
                    'is_ai_adjusted': False
                }
                if threshold_info['current_threshold'] and threshold_info['original_threshold']:
                    threshold_info['is_ai_adjusted'] = abs(threshold_info['current_threshold'] - threshold_info['original_threshold']) > 0.0001
            
            # WebSocket으로 데이터 전송 (모든 숫자 값을 float로 변환)
            emit_data = {
                'timestamp': timestamp.isoformat(),
                'current_price': float(current_price),
                'ohlcv_data': price_history,  # 최근 24시간간의 OHLCV 데이터
                'prediction': prediction_data,
                'position': position_data,
                'technical_indicators': technical_indicators,
                'support_resistance': support_resistance,
                'fibonacci': fibonacci,
                'trend_lines': trend_lines,
                'market_indicators': market_indicators_data,
                'threshold_info': threshold_info
            }
            
            # 기술적 지표의 모든 값도 float로 변환
            if technical_indicators:
                cleaned_indicators = {}
                for key, value in technical_indicators.items():
                    if value is not None and not isinstance(value, (str, bool)):
                        cleaned_indicators[key] = float(value)
                    else:
                        cleaned_indicators[key] = value
                emit_data['technical_indicators'] = cleaned_indicators
            
            socketio.emit('price_update', emit_data)
            
            # AI 분석 결과도 함께 브로드캐스트 (변경된 경우에만)
            if signal_generator and hasattr(signal_generator, 'ai_analysis') and signal_generator.ai_analysis:
                # 이전 분석과 비교하여 변경된 경우에만 브로드캐스트
                current_analysis = signal_generator.ai_analysis
                current_recommendation = current_analysis.get('recommendation', '')
                
                # 이전 분석이 없거나 추천이 변경된 경우 브로드캐스트
                if (last_broadcasted_ai_analysis is None or 
                    last_broadcasted_ai_analysis.get('recommendation') != current_recommendation):
                    # 다음 업데이트까지 남은 시간 계산
                    next_update_time = None
                    if hasattr(signal_generator, 'ai_analysis_time') and signal_generator.ai_analysis_time:
                        next_update_time = (signal_generator.ai_analysis_time + 
                                           timedelta(seconds=signal_generator.ai_analysis_interval)).isoformat()
                    
                    ai_analysis_data = {
                        'timestamp': datetime.now().isoformat(),
                        'analysis': current_analysis,
                        'next_update_time': next_update_time,
                        'update_interval': signal_generator.ai_analysis_interval if hasattr(signal_generator, 'ai_analysis_interval') else 300
                    }
                    socketio.emit('ai_analysis_update', ai_analysis_data)
                    last_broadcasted_ai_analysis = current_analysis.copy()
                    print(f"📡 AI 분석 결과 브로드캐스트: {current_recommendation}")
            
        except Exception as e:
            print(f"데이터 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(60)  # 1분 대기


@app.route('/api/status', methods=['GET'])
def get_status():
    """서버 상태 확인"""
    return jsonify({
        'status': 'running' if is_running else 'stopped',
        'signal_generator_loaded': signal_generator is not None,
        'trader_loaded': trader is not None
    })


@app.route('/api/init', methods=['POST'])
def init_system():
    """시스템 초기화"""
    global signal_generator, trader
    
    try:
        data = request.json or {}
        model_path = data.get('model_path', 'models/best_model.h5')
        leverage = data.get('leverage', 10)
        dry_run = data.get('dry_run', True)
        enable_trading = data.get('enable_trading', False)
        
        # 시그널 생성기 초기화
        signal_generator = RealtimeTradingSignal(model_path=model_path)
        
        # 트레이더 초기화 (선택적)
        if enable_trading:
            trader = RealtimeTrader(
                model_path=model_path,
                leverage=leverage,
                dry_run=dry_run
            )
        
        return jsonify({'success': True, 'message': '시스템 초기화 완료'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/start', methods=['POST'])
def start_updates():
    """데이터 업데이트 시작"""
    global is_running, update_thread, signal_generator
    
    if is_running:
        return jsonify({'success': False, 'message': '이미 실행 중입니다'})
    
    # 초기화가 안 되어 있으면 자동으로 초기화 시도
    if signal_generator is None:
        try:
            print("시그널 생성기 자동 초기화 중...")
            signal_generator = RealtimeTradingSignal(model_path='models/best_model.h5')
            print("자동 초기화 완료")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'시스템 초기화 실패: {str(e)}. /api/init을 먼저 호출하세요.'
            }), 400
    
    is_running = True
    update_thread = threading.Thread(target=update_data_loop, daemon=True)
    update_thread.start()
    
    return jsonify({'success': True, 'message': '데이터 업데이트 시작'})


@app.route('/api/stop', methods=['POST'])
def stop_updates():
    """데이터 업데이트 중지"""
    global is_running
    
    is_running = False
    return jsonify({'success': True, 'message': '데이터 업데이트 중지'})


@app.route('/api/trading/execute-cycle', methods=['POST'])
def execute_trading_cycle():
    """거래 사이클 실행 (realtime_trading.py의 execute_trading_cycle)"""
    global trader, signal_generator
    
    # 트레이더가 없으면 자동으로 초기화 시도
    if trader is None:
        try:
            print("트레이더 자동 초기화 중...")
            data = request.json or {}
            model_path = data.get('model_path', 'models/best_model.h5')
            leverage = data.get('leverage', 10)
            dry_run = data.get('dry_run', False)  # 기본값은 실제 거래 모드
            
            trader = RealtimeTrader(
                model_path=model_path,
                leverage=leverage,
                dry_run=dry_run
            )
            print("트레이더 자동 초기화 완료")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'트레이더 초기화 실패: {str(e)}. /api/init에서 enable_trading=true로 설정하세요.'
            }), 400
    
    try:
        trader.execute_trading_cycle()
        return jsonify({'success': True, 'message': '거래 사이클 실행 완료'})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"거래 사이클 실행 오류: {error_trace}")
        return jsonify({'success': False, 'error': str(e), 'trace': error_trace}), 500


@app.route('/api/trading/position', methods=['GET'])
def get_position():
    """현재 포지션 조회"""
    global trader
    
    # 트레이더가 없으면 자동으로 초기화 시도
    if trader is None:
        try:
            print("트레이더 자동 초기화 중 (포지션 조회)...")
            trader = RealtimeTrader(
                model_path='models/best_model.h5',
                leverage=10,
                dry_run=False  # 실제 거래 모드
            )
            print("트레이더 자동 초기화 완료")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'트레이더 초기화 실패: {str(e)}'
            }), 400
    
    try:
        position = trader.get_current_position()
        if position:
            return jsonify({
                'success': True,
                'position': {
                    'side': position.get('side'),
                    'entry_price': float(position.get('entry_price', 0)),
                    'size': float(position.get('size', 0)),
                    'unrealized_pnl': float(position.get('unrealized_pnl', 0)),
                    'mark_price': float(position.get('mark_price', 0)),
                    'percentage': float(position.get('percentage', 0))
                }
            })
        else:
            return jsonify({'success': True, 'position': None})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading/close-position', methods=['POST'])
def close_position():
    """포지션 닫기"""
    global trader
    
    # 트레이더가 없으면 자동으로 초기화 시도
    if trader is None:
        try:
            print("트레이더 자동 초기화 중 (포지션 닫기)...")
            trader = RealtimeTrader(
                model_path='models/best_model.h5',
                leverage=10,
                dry_run=False  # 실제 거래 모드
            )
            print("트레이더 자동 초기화 완료")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'트레이더 초기화 실패: {str(e)}'
            }), 400
    
    try:
        success = trader.close_position()
        return jsonify({'success': success, 'message': '포지션 닫기 완료' if success else '포지션 닫기 실패'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading/balance', methods=['GET'])
def get_balance():
    """계좌 잔액 조회"""
    global trader
    
    # 트레이더가 없으면 자동으로 초기화 시도
    if trader is None:
        try:
            print("트레이더 자동 초기화 중 (잔액 조회)...")
            trader = RealtimeTrader(
                model_path='models/best_model.h5',
                leverage=10,
                dry_run=False  # 실제 거래 모드
            )
            print("트레이더 자동 초기화 완료")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'트레이더 초기화 실패: {str(e)}'
            }), 400
    
    try:
        balance = trader.get_account_balance()
        return jsonify({
            'success': True,
            'balance': {
                'free': float(balance.get('free', 0)),
                'total': float(balance.get('total', 0)),
                'available': float(balance.get('available', 0))
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history/price', methods=['GET'])
def get_price_history():
    """가격 히스토리 조회"""
    limit = request.args.get('limit', 288, type=int)
    return jsonify(price_history[-limit:])


@app.route('/api/history/prediction', methods=['GET'])
def get_prediction_history():
    """예측 히스토리 조회"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify(prediction_history[-limit:])


@app.route('/api/history/position', methods=['GET'])
def get_position_history():
    """포지션 히스토리 조회"""
    return jsonify(position_history)


@app.route('/api/current', methods=['GET'])
def get_current_data():
    """현재 데이터 조회"""
    return jsonify({
        'price': price_history[-1] if price_history else None,
        'prediction': prediction_history[-1] if prediction_history else None,
        'position': position_history[-1] if position_history else None
    })


@app.route('/api/pattern/image/<path:filename>')
def get_pattern_image(filename):
    """패턴 이미지 파일 제공"""
    try:
        # 파일 경로에서 패턴 폴더와 파일명 추출
        # filename 형식: "Ascending_Triangle/AT_01.jpg"
        parts = filename.split('/')
        if len(parts) == 2:
            pattern_folder, image_file = parts
            dataset_path = os.path.join('data', 'DATASET', pattern_folder)
            return send_from_directory(dataset_path, image_file)
        else:
            return jsonify({'error': 'Invalid filename format'}), 400
    except Exception as e:
        print(f"⚠️ 이미지 제공 오류: {e}")
        return jsonify({'error': str(e)}), 404


@app.route('/api/pattern/find', methods=['POST', 'OPTIONS'])
def find_similar_pattern():
    """유사 패턴 찾기 (Gemini 없이)"""
    print("🔍 find_similar_pattern 함수 시작")
    
    # OPTIONS 요청 처리 (CORS preflight)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS 요청 처리")
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        data = request.get_json()
        print(f"📋 패턴 찾기 요청 수신: priceData={len(data.get('priceData', []))}개")
        
        # 필수 데이터 확인
        if not data or not data.get('priceData'):
            return jsonify({'success': False, 'error': '가격 데이터가 필요합니다.'}), 400
        
        price_data = data.get('priceData', [])
        
        # 현재 차트를 이미지로 생성
        print("📊 차트 이미지 생성 중...")
        current_chart_image = _create_chart_image(price_data)
        
        if not current_chart_image:
            return jsonify({'success': False, 'error': '차트 이미지 생성 실패'}), 400
        
        # dataset 폴더에서 유사 패턴 찾기
        print("🔍 유사 패턴 찾기 시작...")
        similar_pattern = _find_similar_pattern_from_dataset(current_chart_image)
        
        if similar_pattern:
            return jsonify({
                'success': True,
                'pattern': similar_pattern
            })
        else:
            return jsonify({
                'success': False,
                'error': '유사한 패턴을 찾지 못했습니다. (임계값: 40%)'
            })
            
    except Exception as e:
        print(f"❌ 패턴 찾기 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'패턴 찾기 중 오류가 발생했습니다: {str(e)}'}), 500


@app.route('/api/gemini/ask', methods=['POST', 'OPTIONS'])
def ask_gemini():
    """Gemini API에 추가 질문하기"""
    print("💬 ask_gemini 함수 시작")
    
    # OPTIONS 요청 처리 (CORS preflight)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS 요청 처리")
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId', 'default')
        question = data.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'error': '질문이 필요합니다.'}), 400
        
        print(f"💬 추가 질문 수신: sessionId={session_id}, question={question[:50]}...")
        
        if not GENAI_AVAILABLE:
            return jsonify({'success': False, 'error': 'Gemini API를 사용할 수 없습니다.'}), 500
        
        gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        if not gemini_api_key:
            return jsonify({'success': False, 'error': 'GEMINI_API_KEY가 설정되지 않았습니다.'}), 500
        
        genai.configure(api_key=gemini_api_key)
        
        # 대화 히스토리 가져오기 또는 초기화
        if session_id not in gemini_conversations:
            return jsonify({'success': False, 'error': '먼저 초기 분석을 진행해주세요.'}), 400
        
        conversation = gemini_conversations[session_id]
        
        # 추가 질문 추가
        conversation.send_message(question)
        response = conversation.last
        
        if not response or not response.text:
            return jsonify({'success': False, 'error': 'Gemini API 응답을 받지 못했습니다.'}), 500
        
        return jsonify({
            'success': True,
            'answer': response.text
        })
        
    except Exception as e:
        print(f"❌ 추가 질문 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'질문 처리 중 오류가 발생했습니다: {str(e)}'}), 500


@app.route('/api/gemini/analyze', methods=['POST', 'OPTIONS'])
def analyze_with_gemini():
    """Gemini API를 통한 시장 분석"""
    print("🔍 analyze_with_gemini 함수 시작")
    
    # OPTIONS 요청 처리 (CORS preflight)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS 요청 처리")
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        data = request.get_json()
        print(f"📋 요청 데이터 수신: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        session_id = data.get('sessionId', 'default')
        requested_model = data.get('modelName', 'gemini-2.5-flash')
        include_similar_pattern = data.get('includeSimilarPattern', False)
        
        print(f"📋 요청 모델: {requested_model}, 유사 패턴 포함: {include_similar_pattern}")
        
        # 필수 데이터 확인
        if not data:
            print("❌ Gemini 분석 요청: 요청 데이터가 없습니다.")
            return jsonify({'success': False, 'error': '요청 데이터가 없습니다.'}), 400
        
        print(f"📥 Gemini 분석 요청 수신: priceData={len(data.get('priceData', []))}개, predictionData={'있음' if data.get('predictionData') else '없음'}")
        print(f"📊 데이터 확인: trendLines={'있음' if data.get('trendLines') else '없음'}, supportResistance={'있음' if data.get('supportResistance') else '없음'}")
        
        # Gemini API 키 확인
        gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        print(f"🔑 API 키 확인: {'설정됨' if gemini_api_key else '설정 안 됨'}")
        if not gemini_api_key:
            print("❌ Gemini API 키가 설정되지 않았습니다.")
            result = jsonify({'success': False, 'error': 'Gemini API 키가 설정되지 않았습니다. GEMINI_API_KEY 환경변수를 설정해주세요.'})
            print(f"📤 응답 반환: {result.status_code if hasattr(result, 'status_code') else 'N/A'}")
            return result, 400
        
        # 프롬프트 생성
        print("📝 프롬프트 생성 시작...")
        try:
            prompt = _build_gemini_prompt(data, include_similar_pattern=include_similar_pattern)
            print(f"✅ 프롬프트 생성 완료 (길이: {len(prompt)} 문자, 유사 패턴 포함: {include_similar_pattern})")
        except Exception as prompt_error:
            print(f"❌ 프롬프트 생성 오류: {prompt_error}")
            import traceback
            traceback.print_exc()
            result = jsonify({'success': False, 'error': f'프롬프트 생성 중 오류가 발생했습니다: {str(prompt_error)}'})
            print(f"📤 응답 반환 준비 완료")
            return result, 400
        
        print("🌐 Gemini API 호출 시작...")
        
        # Google Generative AI SDK 사용
        if not GENAI_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.'
            }), 500
        
        # API 키 설정
        genai.configure(api_key=gemini_api_key)
        
        # 요청된 모델을 우선 시도, 없으면 폴백 (gemini-2.5-flash 우선)
        models_to_try = [requested_model]
        fallback_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.5-flash']
        # 요청된 모델이 폴백 목록에 없으면 추가
        if requested_model not in fallback_models:
            models_to_try.extend(fallback_models)
        else:
            # 요청된 모델을 제외한 나머지 추가
            models_to_try.extend([m for m in fallback_models if m != requested_model])
        
        print(f"🔧 사용할 모델 목록: {models_to_try}")
        
        response_text = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                print(f"🔗 시도 중: {model_name}...")
                
                # Gemini API 호출 (대화 히스토리 지원)
                model = genai.GenerativeModel(model_name)
                
                # 세션별 대화 히스토리 초기화 또는 생성
                if session_id not in gemini_conversations:
                    gemini_conversations[session_id] = model.start_chat(history=[])
                    print(f"💬 새로운 대화 세션 생성: {session_id}")
                
                conversation = gemini_conversations[session_id]
                response = conversation.send_message(prompt)
                
                if response and response.text:
                    print(f"✅ {model_name} 모델 사용 성공!")
                    response_text = response.text
                    break
                else:
                    print(f"⚠️ {model_name} 응답이 비어있습니다. 다음 모델 시도...")
                    last_error = f'{model_name} 응답이 비어있습니다.'
                    continue
                    
            except Exception as model_error:
                error_msg = str(model_error)
                print(f"❌ {model_name} 오류: {error_msg}")
                last_error = error_msg
                
                # 404 오류면 다음 모델 시도
                if '404' in error_msg or 'not found' in error_msg.lower():
                    print(f"⚠️ {model_name} 모델을 찾을 수 없습니다. 다음 모델 시도...")
                    continue
                else:
                    # 다른 오류면 중단
                    break
        
        if not response_text:
            print(f"❌ 모든 모델 시도 실패: {last_error}")
            
            if '401' in str(last_error) or 'unauthorized' in str(last_error).lower():
                error_message = 'API 키가 유효하지 않습니다. API 키를 확인해주세요.'
            elif '403' in str(last_error) or 'forbidden' in str(last_error).lower():
                error_message = 'API 접근이 거부되었습니다. API 키 권한을 확인해주세요.'
            elif '429' in str(last_error) or 'quota' in str(last_error).lower() or 'exceeded' in str(last_error).lower():
                error_message = (
                    'Gemini API 사용량 한도를 초과했습니다.\n\n'
                    '해결 방법:\n'
                    '1. 잠시 후 다시 시도해주세요 (일일/분당 한도)\n'
                    '2. Google AI Studio에서 사용량 확인: https://ai.dev/usage?tab=rate-limit\n'
                    '3. 필요시 유료 플랜으로 업그레이드하거나 다른 API 키 사용\n'
                    '4. 한도 정보: https://ai.google.dev/gemini-api/docs/rate-limits'
                )
            elif '404' in str(last_error) or 'not found' in str(last_error).lower():
                error_message = f'모델을 찾을 수 없습니다. 사용 가능한 모델을 확인해주세요: {last_error}'
            else:
                error_message = f'Gemini API 호출 실패: {last_error}'
            
            result = jsonify({
                'success': False,
                'error': error_message
            })
            print(f"📤 오류 응답 반환: {error_message[:100]}...")
            return result, 429 if '429' in str(last_error) or 'quota' in str(last_error).lower() else 500
        
        if not response_text:
            finish_reason = result.get('candidates', [{}])[0].get('finishReason', '')
            if finish_reason == 'SAFETY':
                return jsonify({'success': False, 'error': '콘텐츠가 안전 필터에 의해 차단되었습니다.'}), 400
            return jsonify({'success': False, 'error': 'Gemini API 응답 형식이 올바르지 않습니다.'}), 500
        
        # JSON 파싱
        print("📊 응답 파싱 시작...")
        analysis_result = _parse_gemini_response(response_text)
        print(f"✅ 분석 결과 파싱 완료: {list(analysis_result.keys())}")
        
        result = jsonify({
            'success': True,
            'analysis': analysis_result
        })
        print(f"📤 최종 응답 반환: 200 OK")
        return result
        
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API 요청 시간 초과입니다.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'네트워크 오류: {str(e)}'}), 500
    except Exception as e:
        print(f"❌ Gemini 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        result = jsonify({'success': False, 'error': f'분석 중 오류가 발생했습니다: {str(e)}'})
        print(f"📤 오류 응답 반환: 500")
        return result, 500
    finally:
        print("🏁 analyze_with_gemini 함수 종료")


def _create_chart_image(price_data: List[Dict], save_path: str = None) -> Optional[str]:
    """현재 차트를 이미지로 생성 (최근 40~60개 캔들만 사용)"""
    try:
        if not price_data or len(price_data) < 10:
            return None
        
        # 최근 40~60개 캔들만 사용 (패턴 인식을 위해 적절한 범위)
        # 전체 캔들을 사용하지 않고 최근 일부만 사용하여 패턴 매칭 정확도 향상
        if len(price_data) >= 60:
            # 60개 이상이면 최근 50개 사용 (40~60 범위의 중간값)
            recent_data = price_data[-50:]
        elif len(price_data) >= 40:
            # 40~59개면 모두 사용
            recent_data = price_data
        else:
            # 40개 미만이면 사용 불가
            print(f"⚠️ 캔들 개수 부족: {len(price_data)}개 (최소 40개 필요)")
            return None
        
        print(f"📊 차트 이미지 생성: 최근 {len(recent_data)}개 캔들 사용 (전체: {len(price_data)}개 중)")
        
        # 데이터 추출
        closes = [d.get('close', 0) for d in recent_data]
        opens = [d.get('open', closes[i] if i < len(closes) else 0) for i, d in enumerate(recent_data)]
        highs = [d.get('high', 0) for d in recent_data]
        lows = [d.get('low', 0) for d in recent_data]
        volumes = [d.get('volume', 0) for d in recent_data]
        
        if not closes or all(c == 0 for c in closes):
            return None
        
        # 차트 생성 (dataset 스타일에 맞춤: 흰색 배경, 양봉 초록색, 음봉 빨간색)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # 캔들스틱 차트 그리기
        x = range(len(closes))
        for i in range(len(closes)):
            open_price = recent_data[i].get('open', closes[i])
            close_price = closes[i]
            high_price = highs[i]
            low_price = lows[i]
            
            # 양봉(상승)은 초록색, 음봉(하락)은 빨간색
            if close_price >= open_price:
                color = '#10b981'  # 초록색 (양봉)
                body_color = '#10b981'
            else:
                color = '#ef4444'  # 빨간색 (음봉)
                body_color = '#ef4444'
            
            # 캔들 몸통 (시가-종가)
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            body_height = body_top - body_bottom
            
            # 몸통 그리기
            if body_height > 0:
                ax1.bar(i, body_height, bottom=body_bottom, color=body_color, width=0.6, edgecolor=color, linewidth=0.5)
            else:
                # 도지 (시가=종가)
                ax1.plot([i-0.3, i+0.3], [close_price, close_price], color=color, linewidth=1.5)
            
            # 꼬리 그리기 (상단 꼬리)
            ax1.plot([i, i], [body_top, high_price], color=color, linewidth=1)
            # 꼬리 그리기 (하단 꼬리)
            ax1.plot([i, i], [body_bottom, low_price], color=color, linewidth=1)
        
        ax1.set_ylabel('Price (USDT)', color='black', fontsize=12)
        ax1.tick_params(colors='black')
        ax1.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax1.spines['bottom'].set_color('black')
        ax1.spines['top'].set_color('black')
        ax1.spines['right'].set_color('black')
        ax1.spines['left'].set_color('black')
        
        # 거래량 차트 (양봉/음봉에 따라 색상 구분)
        volume_colors = []
        for i in range(len(closes)):
            open_price = recent_data[i].get('open', closes[i])
            close_price = closes[i]
            if close_price >= open_price:
                volume_colors.append('#10b981')  # 초록색
            else:
                volume_colors.append('#ef4444')  # 빨간색
        
        ax2.bar(x, volumes, color=volume_colors, alpha=0.6, width=0.6)
        ax2.set_ylabel('Volume', color='black', fontsize=12)
        ax2.set_xlabel('Candles', color='black', fontsize=12)
        ax2.tick_params(colors='black')
        ax2.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax2.spines['bottom'].set_color('black')
        ax2.spines['top'].set_color('black')
        ax2.spines['right'].set_color('black')
        ax2.spines['left'].set_color('black')
        
        plt.tight_layout()
        
        # 이미지를 base64로 변환 (흰색 배경)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='white', dpi=100, bbox_inches='tight', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ 차트 이미지 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_chart_features(img_gray):
    """차트 이미지에서 패턴 특성 추출"""
    import cv2
    import numpy as np
    
    features = {}
    
    # 1. 추세 방향 분석 (상단/하단 영역의 밝기 차이)
    h, w = img_gray.shape
    top_region = img_gray[:h//3, :].mean()
    middle_region = img_gray[h//3:2*h//3, :].mean()
    bottom_region = img_gray[2*h//3:, :].mean()
    
    # 상승 추세: 상단이 하단보다 밝음 (차트가 위로 올라감)
    # 하락 추세: 하단이 상단보다 밝음 (차트가 아래로 내려감)
    trend_score = (top_region - bottom_region) / 255.0
    features['trend_direction'] = trend_score
    
    # 2. 패턴 형태 분석 (수평선/대각선 비율)
    edges = cv2.Canny(img_gray, 50, 150)
    h_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    v_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    horizontal_ratio = 0.0
    vertical_ratio = 0.0
    
    if h_lines is not None:
        horizontal_count = len([l for l in h_lines if abs(l[0][1] - l[0][3]) < 5])  # 수평선
        horizontal_ratio = horizontal_count / max(len(h_lines), 1)
    
    if v_lines is not None:
        vertical_count = len([l for l in v_lines if abs(l[0][0] - l[0][2]) < 5])  # 수직선
        vertical_ratio = vertical_count / max(len(v_lines), 1)
    
    features['horizontal_lines'] = horizontal_ratio
    features['vertical_lines'] = vertical_ratio
    
    # 3. 대칭성 분석 (삼각형 패턴 감지)
    left_half = img_gray[:, :w//2]
    right_half = img_gray[:, w//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    # 크기 맞추기
    min_w = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_w]
    right_half_flipped = right_half_flipped[:, :min_w]
    
    symmetry_score = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
    features['symmetry'] = symmetry_score if not np.isnan(symmetry_score) else 0.0
    
    # 4. 밀도 분석 (차트가 차지하는 영역)
    non_zero_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    density = non_zero_pixels / total_pixels
    features['density'] = density
    
    return features


def _calculate_image_similarity(img1_path_or_bytes, img2_path_or_bytes) -> float:
    """이미지 유사도 계산 (차트 패턴 특화 알고리즘 - 개선 버전)"""
    try:
        from PIL import Image
        import cv2
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # 이미지 로드
        if isinstance(img1_path_or_bytes, bytes):
            img1 = Image.open(BytesIO(img1_path_or_bytes))
        else:
            img1 = Image.open(img1_path_or_bytes)
        
        if isinstance(img2_path_or_bytes, bytes):
            img2 = Image.open(BytesIO(img2_path_or_bytes))
        else:
            img2 = Image.open(img2_path_or_bytes)
        
        # 차트 이미지는 가로가 긴 형태이므로 비율 유지하며 리사이즈
        target_width = 800
        target_height = 600
        
        img1 = img1.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img2 = img2.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 그레이스케일로 변환
        img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        
        # 차트 특성 추출
        features1 = _extract_chart_features(img1_gray)
        features2 = _extract_chart_features(img2_gray)
        
        # 특성 기반 유사도 계산
        feature_similarity = 0.0
        feature_count = 0
        
        for key in features1.keys():
            if key in features2:
                # 각 특성의 차이를 계산 (0~1 범위로 정규화)
                diff = abs(features1[key] - features2[key])
                similarity = 1.0 - min(1.0, diff)
                feature_similarity += similarity
                feature_count += 1
        
        if feature_count > 0:
            feature_similarity = feature_similarity / feature_count
        
        # 엣지 검출 (차트 패턴의 형태를 더 잘 인식)
        img1_edges = cv2.Canny(img1_gray, 50, 150)
        img2_edges = cv2.Canny(img2_gray, 50, 150)
        
        # 엣지 이미지의 SSIM 계산 (차트 라인 패턴 비교)
        edge_ssim = ssim(img1_edges, img2_edges, data_range=255)
        
        # 원본 그레이스케일 SSIM
        gray_ssim = ssim(img1_gray, img2_gray, data_range=255)
        
        # 템플릿 매칭 (차트의 주요 영역 비교)
        # 차트의 상단, 중간, 하단 영역을 각각 비교
        h, w = img1_gray.shape
        template_scores = []
        
        for y_offset in [0, h//3, 2*h//3]:
            template_h = h // 3
            template1 = img1_gray[y_offset:y_offset+template_h, :]
            template2 = img2_gray[y_offset:y_offset+template_h, :]
            
            if template1.shape == template2.shape:
                result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)
                template_scores.append(result[0][0] if not np.isnan(result[0][0]) else 0.0)
        
        template_score = np.mean(template_scores) if template_scores else 0.0
        
        # 종합 점수 (차트 패턴 특화 가중치 - 개선)
        # 특성 기반: 35% (추세, 형태 등 패턴 특성)
        # 엣지 SSIM: 25% (차트 라인 패턴)
        # 템플릿 매칭: 25% (주요 영역 비교)
        # 그레이스케일 SSIM: 15% (전체적인 유사도)
        final_score = (
            feature_similarity * 0.35 +
            edge_ssim * 0.25 +
            template_score * 0.25 +
            gray_ssim * 0.15
        ) * 100
        
        return final_score
    except Exception as e:
        print(f"⚠️ 이미지 유사도 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def _find_similar_pattern_from_dataset(current_chart_image: str) -> Optional[Dict]:
    """dataset 폴더에서 유사한 차트 패턴 찾기 (무료 이미지 비교 사용)"""
    try:
        dataset_path = 'data/DATASET'
        print(f"🔍 유사 패턴 찾기 시작: dataset_path={dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset 폴더를 찾을 수 없습니다: {dataset_path}")
            return None
        
        # 패턴 폴더 목록
        pattern_folders = [
            'Ascending_Triangle',
            'Descending_Triangle',
            'Double_Bottom',
            'Double_Top',
            'Falling_Wedge',
            'Rising_Wedge',
            'Symmetrical_Triangle'
        ]
        
        # 현재 차트 이미지를 bytes로 변환
        try:
            current_img_bytes = base64.b64decode(current_chart_image)
            print(f"✅ 차트 이미지 디코딩 완료: {len(current_img_bytes)} bytes")
        except Exception as e:
            print(f"❌ 차트 이미지 디코딩 실패: {e}")
            return None
        
        # 각 패턴 폴더에서 샘플 이미지 선택 (각 폴더당 5개씩)
        best_match = None
        best_score = 0.0
        total_comparisons = 0
        comparison_results = []
        
        for pattern_folder in pattern_folders:
            folder_path = os.path.join(dataset_path, pattern_folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ 패턴 폴더 없음: {folder_path}")
                continue
            
            # 폴더 내 이미지 파일 찾기
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"📁 {pattern_folder}: {len(image_files)}개 이미지 파일 발견")
            
            if not image_files:
                continue
            
            # 샘플 이미지 선택 (최대 5개)
            sample_images = image_files[:5]
            
            for sample_file in sample_images:
                sample_path = os.path.join(folder_path, sample_file)
                try:
                    total_comparisons += 1
                    print(f"  🔄 비교 중: {sample_file}...", end=' ')
                    
                    # 이미지 유사도 계산
                    similarity_score = _calculate_image_similarity(current_img_bytes, sample_path)
                    print(f"유사도: {similarity_score:.2f}%")
                    
                    comparison_results.append({
                        'pattern': pattern_folder,
                        'file': sample_file,
                        'score': similarity_score
                    })
                    
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_match = {
                            'pattern_type': pattern_folder,
                            'pattern_file': sample_file,
                            'similarity_score': similarity_score,
                            'description': f"{pattern_folder} 패턴과 유사도 {similarity_score:.1f}%로 일치합니다."
                        }
                        
                except Exception as e:
                    print(f"❌ 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"📊 총 비교 횟수: {total_comparisons}개")
        if comparison_results:
            # 상위 5개 결과 출력
            top_results = sorted(comparison_results, key=lambda x: x['score'], reverse=True)[:5]
            print(f"🏆 상위 5개 결과:")
            for i, result in enumerate(top_results, 1):
                print(f"   {i}. {result['pattern']}/{result['file']}: {result['score']:.2f}%")
        
        # 상대적 비교: 최고 점수가 두 번째 점수보다 충분히 높으면 유사하다고 판단
        if comparison_results and len(comparison_results) >= 2:
            sorted_results = sorted(comparison_results, key=lambda x: x['score'], reverse=True)
            best_score = sorted_results[0]['score']
            second_score = sorted_results[1]['score']
            score_diff = best_score - second_score
            
            # 최고 점수가 10% 이상이고, 두 번째 점수보다 0.5% 이상 높으면 유사하다고 판단
            if best_score >= 10.0 and score_diff >= 0.5:
                print(f"✅ 유사 패턴 발견: {best_match['pattern_type']} (유사도: {best_score:.1f}%, 차이: {score_diff:.2f}%)")
                return best_match
            else:
                print(f"⚠️ 유사 패턴 없음 (최고 점수: {best_score:.1f}%, 두 번째: {second_score:.1f}%, 차이: {score_diff:.2f}%)")
                return None
        elif best_match and best_score >= 10.0:
            # 비교 결과가 적을 때는 절대값 기준
            print(f"✅ 유사 패턴 발견: {best_match['pattern_type']} (유사도: {best_score:.1f}%)")
            return best_match
        else:
            print(f"⚠️ 유사 패턴 없음 (최고 점수: {best_score:.1f}%)")
            return None
    except Exception as e:
        print(f"❌ Dataset 패턴 찾기 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def _find_similar_pattern(current_prices: List[Dict], price_history: List[Dict]) -> Optional[Dict]:
    """현재 패턴과 유사한 과거 패턴 찾기"""
    if not current_prices or len(current_prices) < 10 or not price_history or len(price_history) < 50:
        return None
    
    try:
        # 현재 패턴의 특징 추출 (최근 20개 캔들)
        current_pattern = current_prices[-20:] if len(current_prices) >= 20 else current_prices
        
        # 현재 패턴의 변화율 계산
        current_changes = []
        for i in range(1, len(current_pattern)):
            prev_close = current_pattern[i-1].get('close', 0)
            curr_close = current_pattern[i].get('close', 0)
            if prev_close > 0:
                change = (curr_close - prev_close) / prev_close
                current_changes.append(change)
        
        if len(current_changes) < 5:
            return None
        
        # 과거 데이터에서 유사한 패턴 찾기
        best_match = None
        best_score = float('inf')
        
        # 과거 데이터를 20개씩 슬라이딩 윈도우로 비교
        for i in range(20, len(price_history) - 20):
            past_pattern = price_history[i-20:i]
            
            # 과거 패턴의 변화율 계산
            past_changes = []
            for j in range(1, len(past_pattern)):
                prev_close = past_pattern[j-1].get('price', past_pattern[j-1].get('close', 0))
                curr_close = past_pattern[j].get('price', past_pattern[j].get('close', 0))
                if prev_close > 0:
                    change = (curr_close - prev_close) / prev_close
                    past_changes.append(change)
            
            if len(past_changes) != len(current_changes):
                continue
            
            # 패턴 유사도 계산 (MSE)
            mse = sum((c - p) ** 2 for c, p in zip(current_changes, past_changes)) / len(current_changes)
            
            if mse < best_score:
                best_score = mse
                best_match = {
                    'start_idx': i - 20,
                    'end_idx': i,
                    'pattern_start_price': past_pattern[0].get('price', past_pattern[0].get('close', 0)),
                    'pattern_end_price': past_pattern[-1].get('price', past_pattern[-1].get('close', 0)),
                    'pattern_timestamp': past_pattern[-1].get('timestamp'),
                    'similarity_score': mse,
                    'future_prices': price_history[i:i+10] if i + 10 < len(price_history) else price_history[i:]
                }
        
        # 유사도가 충분히 높은 경우만 반환 (임계값: 0.0001)
        if best_match and best_match['similarity_score'] < 0.0001:
            return best_match
        
        return None
    except Exception as e:
        print(f"⚠️ 유사 패턴 찾기 오류: {e}")
        return None


def _find_similar_pattern(current_prices: List[Dict], price_history: List[Dict]) -> Optional[Dict]:
    """현재 패턴과 유사한 과거 패턴 찾기"""
    if not current_prices or len(current_prices) < 10 or not price_history or len(price_history) < 50:
        return None
    
    try:
        # 현재 패턴의 특징 추출 (최근 20개 캔들)
        current_pattern = current_prices[-20:] if len(current_prices) >= 20 else current_prices
        
        # 현재 패턴의 변화율 계산
        current_changes = []
        for i in range(1, len(current_pattern)):
            prev_close = current_pattern[i-1].get('close', 0)
            curr_close = current_pattern[i].get('close', 0)
            if prev_close > 0:
                change = (curr_close - prev_close) / prev_close
                current_changes.append(change)
        
        if len(current_changes) < 5:
            return None
        
        # 과거 데이터에서 유사한 패턴 찾기
        best_match = None
        best_score = float('inf')
        
        # 과거 데이터를 20개씩 슬라이딩 윈도우로 비교
        for i in range(20, len(price_history) - 20):
            past_pattern = price_history[i-20:i]
            
            # 과거 패턴의 변화율 계산
            past_changes = []
            for j in range(1, len(past_pattern)):
                prev_close = past_pattern[j-1].get('price', past_pattern[j-1].get('close', 0))
                curr_close = past_pattern[j].get('price', past_pattern[j].get('close', 0))
                if prev_close > 0:
                    change = (curr_close - prev_close) / prev_close
                    past_changes.append(change)
            
            if len(past_changes) != len(current_changes):
                continue
            
            # 패턴 유사도 계산 (MSE)
            mse = sum((c - p) ** 2 for c, p in zip(current_changes, past_changes)) / len(current_changes)
            
            if mse < best_score:
                best_score = mse
                # 패턴 이후 10개 캔들의 가격 변화 확인
                future_prices = price_history_list[i:i+10] if i + 10 < len(price_history_list) else price_history_list[i:]
                future_changes = []
                if len(future_prices) > 1:
                    pattern_end_price = past_pattern[-1].get('price', past_pattern[-1].get('close', 0))
                    for fp in future_prices[1:]:
                        fp_price = fp.get('price', fp.get('close', 0))
                        if pattern_end_price > 0:
                            future_changes.append((fp_price - pattern_end_price) / pattern_end_price)
                            pattern_end_price = fp_price
                
                best_match = {
                    'start_idx': i - 20,
                    'end_idx': i,
                    'pattern_start_price': past_pattern[0].get('price', past_pattern[0].get('close', 0)),
                    'pattern_end_price': past_pattern[-1].get('price', past_pattern[-1].get('close', 0)),
                    'pattern_timestamp': past_pattern[-1].get('timestamp'),
                    'similarity_score': mse,
                    'future_prices': future_prices[:10] if len(future_prices) >= 10 else future_prices,
                    'future_changes': future_changes[:10] if len(future_changes) >= 10 else future_changes
                }
        
        # 유사도가 충분히 높은 경우만 반환 (임계값: 0.0001)
        if best_match and best_match['similarity_score'] < 0.0001:
            return best_match
        
        return None
    except Exception as e:
        print(f"⚠️ 유사 패턴 찾기 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def _build_gemini_prompt(data: Dict, include_similar_pattern: bool = False) -> str:
    """Gemini API용 프롬프트 생성"""
    price_data = data.get('priceData', []) or []
    prediction_data = data.get('predictionData', {}) or {}
    technical_indicators = data.get('technicalIndicators', {}) or {}
    support_resistance = data.get('supportResistance', {}) or {}
    trend_lines = data.get('trendLines', {}) or {}
    market_indicators = data.get('marketIndicators', {}) or {}
    fibonacci = data.get('fibonacci', {}) or {}
    
    # dataset 폴더에서 유사한 차트 패턴 찾기 (옵션)
    similar_pattern = None
    if include_similar_pattern and price_data and len(price_data) >= 10:
        # 현재 차트를 이미지로 생성
        current_chart_image = _create_chart_image(price_data)
        if current_chart_image:
            # dataset 폴더에서 유사 패턴 찾기
            similar_pattern = _find_similar_pattern_from_dataset(current_chart_image)
            if similar_pattern:
                print(f"✅ Dataset에서 유사 패턴 발견: {similar_pattern['pattern_type']} (유사도: {similar_pattern['similarity_score']:.1f}%)")
            else:
                print("⚠️ Dataset에서 유사 패턴을 찾지 못했습니다.")
    
    # 최근 가격 데이터 요약
    recent_prices = price_data[-20:] if price_data and len(price_data) > 0 else []
    current_price = recent_prices[-1].get('close') if recent_prices and len(recent_prices) > 0 else None
    
    # 안전한 숫자 변환 헬퍼 함수
    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_format(value, format_str='.2f', default='N/A'):
        try:
            if value is None:
                return default
            num = safe_float(value)
            return f"{num:{format_str}}"
        except:
            return default
    
    # 거래량 계산
    total_volume = sum(safe_float(p.get('volume', 0)) for p in recent_prices)
    avg_volume = total_volume / len(recent_prices) if recent_prices else 0.0
    
    # 가격 범위
    if recent_prices:
        try:
            lows = [safe_float(p.get('low', 0)) for p in recent_prices]
            highs = [safe_float(p.get('high', 0)) for p in recent_prices]
            price_range = f"${min(lows):.2f} - ${max(highs):.2f}"
        except:
            price_range = 'N/A'
    else:
        price_range = 'N/A'
    
    # 추세선 요약 (키 이름 확인: uptrend/downtrend 또는 uptrend_line/downtrend_line)
    uptrend = trend_lines.get('uptrend', {}) or trend_lines.get('uptrend_line', {}) if trend_lines else {}
    downtrend = trend_lines.get('downtrend', {}) or trend_lines.get('downtrend_line', {}) if trend_lines else {}
    
    # 추세선 데이터 상세 확인
    uptrend_exists = uptrend and (uptrend.get('start_price') is not None or uptrend.get('validity') == 'valid')
    downtrend_exists = downtrend and (downtrend.get('start_price') is not None or downtrend.get('validity') == 'valid')
    print(f"📈 추세선 데이터 확인:")
    print(f"   - trend_lines 전체: {list(trend_lines.keys()) if trend_lines else 'None'}")
    print(f"   - uptrend_line 존재: {bool(uptrend)}, 키: {list(uptrend.keys()) if uptrend else 'None'}")
    print(f"   - downtrend_line 존재: {bool(downtrend)}, 키: {list(downtrend.keys()) if downtrend else 'None'}")
    print(f"   - uptrend 유효: {uptrend_exists}, downtrend 유효: {downtrend_exists}")
    
    # 시장 지표 요약
    market_summary = {}
    if market_indicators:
        market_summary = {
            'orderbook_imbalance': market_indicators.get('orderbook_imbalance', {}).get('imbalance_strength', 'neutral'),
            'liquidation_clusters': market_indicators.get('liquidation_clusters', {}).get('liquidation_strength', 'neutral'),
            'volatility_squeeze': market_indicators.get('volatility_squeeze', {}).get('squeeze_status', 'normal'),
            'oi_surge': market_indicators.get('oi_surge', {}).get('oi_surge_status', 'normal'),
            'cvd_trend': market_indicators.get('cvd_turnover', {}).get('cvd_trend', 'neutral')
        }
    
    # 시장 지표 문자열 생성 (f-string 내부에서 백슬래시 사용 불가하므로 미리 생성)
    if market_summary:
        market_indicators_text = (
            f"- 오더북 불균형: {market_summary.get('orderbook_imbalance', 'N/A')}\n"
            f"- 청산 클러스터: {market_summary.get('liquidation_clusters', 'N/A')}\n"
            f"- 변동성 압축: {market_summary.get('volatility_squeeze', 'N/A')}\n"
            f"- OI 급증: {market_summary.get('oi_surge', 'N/A')}\n"
            f"- CVD 추세: {market_summary.get('cvd_trend', 'N/A')}"
        )
    else:
        market_indicators_text = "- 시장 지표: 데이터 없음"
    
    # 추세선 문자열 생성
    if uptrend_exists and uptrend:
        uptrend_start = safe_format(uptrend.get('start_price'))
        uptrend_end = safe_format(uptrend.get('end_price'))
        uptrend_validity = uptrend.get('validity', 'unknown')
        uptrend_slope = safe_format(uptrend.get('slope'), '.6f') if uptrend.get('slope') is not None else 'N/A'
        uptrend_touches = uptrend.get('touch_count', 'N/A')
        uptrend_text = f"- 상승 추세선: ${uptrend_start} → ${uptrend_end} (유효성: {uptrend_validity}, 기울기: {uptrend_slope}, 터치: {uptrend_touches})"
    else:
        uptrend_text = "- 상승 추세선: 없음"
    
    if downtrend_exists and downtrend:
        downtrend_start = safe_format(downtrend.get('start_price'))
        downtrend_end = safe_format(downtrend.get('end_price'))
        downtrend_validity = downtrend.get('validity', 'unknown')
        downtrend_slope = safe_format(downtrend.get('slope'), '.6f') if downtrend.get('slope') is not None else 'N/A'
        downtrend_touches = downtrend.get('touch_count', 'N/A')
        downtrend_text = f"- 하락 추세선: ${downtrend_start} → ${downtrend_end} (유효성: {downtrend_validity}, 기울기: {downtrend_slope}, 터치: {downtrend_touches})"
    else:
        downtrend_text = "- 하락 추세선: 없음"
    
    # 안전한 데이터 추출
    pred_30m = safe_float(prediction_data.get('predicted_price_30m'), 0)
    change_30m = safe_float(prediction_data.get('change_30m'), 0)
    pred_1h = safe_float(prediction_data.get('predicted_price_1h'), 0)
    change_1h = safe_float(prediction_data.get('change_1h'), 0)
    confidence = safe_float(prediction_data.get('confidence'), 0)
    
    # 모든 이동평균선 추출
    ma5 = safe_float(technical_indicators.get('ma5'), 0)
    ma10 = safe_float(technical_indicators.get('ma10'), 0)
    ma20 = safe_float(technical_indicators.get('ma20'), 0)
    ma50 = safe_float(technical_indicators.get('ma50'), 0)
    ma100 = safe_float(technical_indicators.get('ma100'), 0)
    ma200 = safe_float(technical_indicators.get('ma200'), 0)
    
    # 기술적 지표 추출
    rsi = safe_float(technical_indicators.get('rsi'), 0)
    cci = safe_float(technical_indicators.get('cci'), 0)
    macd = safe_float(technical_indicators.get('macd'), 0)
    macd_signal = safe_float(technical_indicators.get('macd_signal'), 0)
    macd_histogram = safe_float(technical_indicators.get('macd_histogram'), 0)
    
    # 볼린저 밴드 추출
    bb_upper = safe_float(technical_indicators.get('bollinger_upper'), 0)
    bb_middle = safe_float(technical_indicators.get('bollinger_middle'), 0)
    bb_lower = safe_float(technical_indicators.get('bollinger_lower'), 0)
    bb_width = safe_float(technical_indicators.get('bollinger_width'), 0)
    bb_position = safe_float(technical_indicators.get('bollinger_position'), 0)
    
    # 지지/저항선 추출
    support_level = safe_float(support_resistance.get('current_support'))
    resistance_level = safe_float(support_resistance.get('current_resistance'))
    
    # 최근 가격 데이터 상세 추출
    if recent_prices:
        latest = recent_prices[-1]
        open_price = safe_float(latest.get('open'))
        high_price = safe_float(latest.get('high'))
        low_price = safe_float(latest.get('low'))
        close_price = safe_float(latest.get('close'))
        volume = safe_float(latest.get('volume'))
        
        # 최근 5개 캔들 거래량
        recent_volumes = [safe_float(p.get('volume', 0)) for p in recent_prices[-5:]]
        volume_trend = '증가' if len(recent_volumes) >= 2 and recent_volumes[-1] > recent_volumes[-2] else '감소' if len(recent_volumes) >= 2 and recent_volumes[-1] < recent_volumes[-2] else '유지'
    else:
        open_price = high_price = low_price = close_price = volume = 0
        volume_trend = 'N/A'
    
    # 피보나치 되돌림 레벨 추출
    fib_high = safe_float(fibonacci.get('high'))
    fib_low = safe_float(fibonacci.get('low'))
    fib_current = safe_float(fibonacci.get('current'))
    fib_trend = fibonacci.get('trend', 'N/A')
    fib_0 = safe_float(fibonacci.get('fib_0'))
    fib_24 = safe_float(fibonacci.get('fib_24'))
    fib_38 = safe_float(fibonacci.get('fib_38'))
    fib_50 = safe_float(fibonacci.get('fib_50'))
    fib_62 = safe_float(fibonacci.get('fib_62'))
    fib_79 = safe_float(fibonacci.get('fib_79'))
    fib_100 = safe_float(fibonacci.get('fib_100'))
    
    # 유사 패턴 텍스트 미리 생성
    similar_pattern_text = ""
    if similar_pattern:
        pattern_type = similar_pattern['pattern_type']
        similarity_score = safe_format(similar_pattern.get('similarity_score', 0), '.1f')
        pattern_desc = similar_pattern.get('description', 'N/A')[:300]
        similar_pattern_text = f"""
### 유사한 차트 패턴 분석 (Dataset)

Dataset에서 현재 차트와 유사한 패턴을 발견했습니다:

- 패턴 유형: {pattern_type}
- 유사도 점수: {similarity_score}% (높을수록 유사함)
- 패턴 설명: {pattern_desc}

**패턴 유형 설명**:
- Ascending_Triangle (상승 삼각형): 일반적으로 상승 추세에서 나타나며, 돌파 시 상승 가능성이 높습니다.
- Descending_Triangle (하락 삼각형): 일반적으로 하락 추세에서 나타나며, 하향 돌파 시 하락 가능성이 높습니다.
- Double_Bottom (이중 바닥): 강세 반전 패턴으로, 두 번째 바닥 형성 후 상승 가능성이 높습니다.
- Double_Top (이중 천장): 약세 반전 패턴으로, 두 번째 천장 형성 후 하락 가능성이 높습니다.
- Falling_Wedge (하락 쐐기): 일반적으로 상승 반전 패턴으로, 하락 쐐기 형성 후 상승 가능성이 높습니다.
- Rising_Wedge (상승 쐐기): 일반적으로 하락 반전 패턴으로, 상승 쐐기 형성 후 하락 가능성이 높습니다.
- Symmetrical_Triangle (대칭 삼각형): 돌파 방향에 따라 상승 또는 하락 가능성이 있습니다.

**중요**: 위 유사 패턴 정보를 참고하여 해당 패턴의 일반적인 특성과 예상 움직임을 분석하고, 이를 바탕으로 현재 추천을 제공해주세요.
"""
    
    prompt = f"""당신은 20년 경력의 전문 암호화폐 트레이딩 분석가입니다. 매우 신중하고 체계적인 분석을 수행해야 합니다.

## ⚠️ 분석 원칙 (반드시 준수)

1. **균형잡힌 접근**: 불확실성이 크면 "waiting"을 선택하되, 명확한 신호가 3개 이상 일치하면 "long" 또는 "short"를 추천하세요.
2. **단계별 검증**: 아래 제시된 5단계 검증 절차를 반드시 순서대로 수행하세요.
3. **신호 일치도**: 최소 3개 이상의 지표가 같은 방향을 가리킬 때 거래를 추천하세요.
4. **리스크 우선**: 손실 가능성이 수익 가능성보다 현저히 크면 "waiting"을 선택하세요.
5. **데이터 신뢰도**: 데이터가 심각하게 부족하거나 대부분의 지표가 모순되면 "waiting"을 선택하세요.
6. **유의점 필수 제공**: recommendation 값과 관계없이 항상 롱/숏/관망 세 가지 모두에 대한 유의점을 제공하세요.

## 현재 시장 데이터 (모든 정보 포함)

### 📊 가격 정보 (OHLCV)
- 현재 가격 (Close): ${safe_format(current_price)}
- 시가 (Open): ${safe_format(open_price)}
- 고가 (High): ${safe_format(high_price)}
- 저가 (Low): ${safe_format(low_price)}
- 최근 20개 캔들 가격 범위: {price_range}
- 현재 캔들 거래량: {safe_format(volume)}
- 최근 평균 거래량: {safe_format(avg_volume)}
- 거래량 추세: {volume_trend}

### 🔮 예측 데이터
- 30분 후 예측 가격: ${safe_format(pred_30m)} ({change_30m:+.2f}%)
- 1시간 후 예측 가격: ${safe_format(pred_1h)} ({change_1h:+.2f}%)
- 거래 신호: {prediction_data.get('signal', 'neutral')}
- 신뢰도: {safe_format(confidence, '.2f')}

### 📈 이동평균선 (MA) - 모든 기간 포함
- MA5: ${safe_format(ma5)} {"(현재가 위)" if current_price and ma5 and current_price > ma5 else "(현재가 아래)" if current_price and ma5 else ""}
- MA10: ${safe_format(ma10)} {"(현재가 위)" if current_price and ma10 and current_price > ma10 else "(현재가 아래)" if current_price and ma10 else ""}
- MA20: ${safe_format(ma20)} {"(현재가 위)" if current_price and ma20 and current_price > ma20 else "(현재가 아래)" if current_price and ma20 else ""}
- MA50: ${safe_format(ma50)} {"(현재가 위)" if current_price and ma50 and current_price > ma50 else "(현재가 아래)" if current_price and ma50 else ""}
- MA100: ${safe_format(ma100)} {"(현재가 위)" if current_price and ma100 and current_price > ma100 else "(현재가 아래)" if current_price and ma100 else ""}
- MA200: ${safe_format(ma200)} {"(현재가 위)" if current_price and ma200 and current_price > ma200 else "(현재가 아래)" if current_price and ma200 else ""}
- 골든크로스: {'예 (강세 신호)' if technical_indicators.get('golden_cross') else '아니오'}
- 데드크로스: {'예 (약세 신호)' if technical_indicators.get('dead_cross') else '아니오'}

### 📊 기술적 지표 (모든 지표 포함)
- RSI (14): {safe_format(rsi, '.1f')} {"(과매수: 70 이상)" if rsi >= 70 else "(과매도: 30 이하)" if rsi <= 30 else "(중립: 30-70)"}
- CCI (20): {safe_format(cci, '.1f')} {"(과매수: 100 이상)" if cci >= 100 else "(과매도: -100 이하)" if cci <= -100 else "(중립: -100~100)"}
- MACD: {safe_format(macd, '.4f')}
- MACD Signal: {safe_format(macd_signal, '.4f')}
- MACD Histogram: {safe_format(macd_histogram, '.4f')} {"(상승 전환)" if macd_histogram > 0 else "(하락 전환)" if macd_histogram < 0 else ""}

### 📉 볼린저 밴드 (모든 정보 포함)
- 볼린저 밴드 상단: ${safe_format(bb_upper)} {"(현재가 근접: 상단 돌파 가능)" if current_price and bb_upper and (current_price / bb_upper) > 0.98 else ""}
- 볼린저 밴드 중간선: ${safe_format(bb_middle)}
- 볼린저 밴드 하단: ${safe_format(bb_lower)} {"(현재가 근접: 하단 돌파 가능)" if current_price and bb_lower and (current_price / bb_lower) < 1.02 else ""}
- 볼린저 밴드 폭: {safe_format(bb_width, '.4f')} {"(압축 상태)" if bb_width < 0.01 else "(확장 상태)" if bb_width > 0.05 else ""}
- 현재가 위치: {safe_format(bb_position, '.2f')} (0=하단, 0.5=중간, 1=상단)

### 🎯 시장 지표 (상세)
{market_indicators_text}
- 오더북 불균형 상세: {market_indicators.get('orderbook_imbalance', {}).get('imbalance_ratio', 'N/A') if market_indicators.get('orderbook_imbalance') else 'N/A'}
- 청산 클러스터 상세: {market_indicators.get('liquidation_clusters', {}).get('liquidation_amount', 'N/A') if market_indicators.get('liquidation_clusters') else 'N/A'}
- CVD 추세 상세: {market_indicators.get('cvd_turnover', {}).get('cvd_value', 'N/A') if market_indicators.get('cvd_turnover') else 'N/A'}

### 🏛️ 지지선/저항선 (상세)
- 현재 지지선: ${safe_format(support_level) if support_level else 'N/A'} {"(현재가 대비: " + f"{((current_price - support_level) / support_level * 100):.2f}% 위)" if support_level and current_price else ""}
- 현재 저항선: ${safe_format(resistance_level) if resistance_level else 'N/A'} {"(현재가 대비: " + f"{((resistance_level - current_price) / current_price * 100):.2f}% 위)" if resistance_level and current_price else ""}
- 지지선 강도: {support_resistance.get('support_strength', 'N/A') if support_resistance else 'N/A'}
- 저항선 강도: {support_resistance.get('resistance_strength', 'N/A') if support_resistance else 'N/A'}

### 📈 추세선 데이터 (상세)
{uptrend_text}
{downtrend_text}
{f"- 상승 추세선 유효성: {uptrend.get('validity', 'unknown')} (터치 {uptrend.get('touch_count', 'N/A')}회)" if uptrend else ""}
{f"- 하락 추세선 유효성: {downtrend.get('validity', 'unknown')} (터치 {downtrend.get('touch_count', 'N/A')}회)" if downtrend else ""}
{f"- 상승 추세선 기울기: {safe_format(uptrend.get('slope'), '.6f')}" if uptrend and uptrend.get('slope') is not None else ""}
{f"- 하락 추세선 기울기: {safe_format(downtrend.get('slope'), '.6f')}" if downtrend and downtrend.get('slope') is not None else ""}

### 🔢 피보나치 되돌림 레벨 (모든 레벨 포함)
- 최고가: ${safe_format(fib_high)}
- 최저가: ${safe_format(fib_low)}
- 현재가: ${safe_format(fib_current)}
- 추세 방향: {fib_trend}
- 피보나치 레벨:
  - 0% (기준선): ${safe_format(fib_0)}
  - 23.6%: ${safe_format(fib_24)}
  - 38.2%: ${safe_format(fib_38)}
  - 50%: ${safe_format(fib_50)}
  - 61.8%: ${safe_format(fib_62)}
  - 78.6%: ${safe_format(fib_79)}
  - 100%: ${safe_format(fib_100)}

{similar_pattern_text}

## 📊 5단계 검증 절차 (반드시 순서대로 수행)

### 1단계: 예측 모델 신호 확인
- 30분 예측: {change_30m:+.2f}% ({'상승' if change_30m > 0 else '하락' if change_30m < 0 else '중립'})
- 1시간 예측: {change_1h:+.2f}% ({'상승' if change_1h > 0 else '하락' if change_1h < 0 else '중립'})
- **판단**: 두 예측이 같은 방향이고 절댓값이 0.5% 이상이어야 신뢰 가능. 그렇지 않으면 "waiting" 선택.

### 2단계: 기술적 지표 확인
- **이동평균**: 현재가가 MA5, MA20, MA50 중 몇 개 위에 있는지 확인
  - 3개 모두 위: 강한 상승 추세
  - 2개 위: 약한 상승 추세
  - 1개 위: 중립
  - 모두 아래: 하락 추세
- **RSI**: {safe_format(rsi, '.1f')}
  - 70 이상: 과매수 (하락 가능성)
  - 30 이하: 과매도 (상승 가능성)
  - 30-70: 중립
- **크로스**: 골든크로스는 상승, 데드크로스는 하락 신호
- **판단**: 최소 2개 이상의 기술적 지표가 같은 방향을 가리켜야 함.

### 3단계: 추세선 및 지지/저항선 확인
- **상승 추세선**: {uptrend_text}
- **하락 추세선**: {downtrend_text}
- **지지선**: ${safe_format(support_level) if support_level else 'N/A'} {"(현재가와의 거리: " + f"{((current_price - support_level) / current_price * 100):.2f}%)" if support_level and current_price else ""}
- **저항선**: ${safe_format(resistance_level) if resistance_level else 'N/A'} {"(현재가와의 거리: " + f"{((resistance_level - current_price) / current_price * 100):.2f}%)" if resistance_level and current_price else ""}
- **판단**: 
  - 상승 추세선이 유효하고 현재가가 추세선 위에 있으면 상승 신호
  - 하락 추세선이 유효하고 현재가가 추세선 아래에 있으면 하락 신호
  - 지지선 근처면 상승 가능성, 저항선 근처면 하락 가능성
  - 추세선 데이터를 반드시 고려하여 분석하세요

### 4단계: 시장 지표 확인
{market_indicators_text}
- **판단**: 오더북 불균형, 청산 클러스터, CVD 추세가 같은 방향을 가리키는지 확인

### 5단계: 종합 판단 및 리스크 평가
- **신호 일치도 계산**: 위 4단계에서 같은 방향을 가리키는 신호가 몇 개인지 세세요
  - 4개 이상 일치: 강한 신호 (거래 추천 가능)
  - 3개 일치: 약한 신호 (신중하게 거래 추천)
  - 2개 이하: 불확실 (반드시 "waiting" 선택)
- **리스크 평가**:
  - 지지선/저항선과의 거리가 가까우면 돌파 가능성 높음
  - RSI가 극단값(70 이상 또는 30 이하)이면 반전 가능성 높음
  - 예측 변화율이 0.5% 미만이면 신호가 약함
  - 추세선이 없거나 유효하지 않으면 추세가 불명확함

## 🎯 최종 추천 기준

**"long" 추천 조건 (5개 중 3개 이상 만족 시 추천 가능)**:
1. 예측 모델이 상승 방향 (30분 또는 1시간 중 하나라도 +0.3% 이상)
2. 기술적 지표 2개 이상이 상승 신호 (MA 위, 골든크로스, RSI < 70 등)
3. 상승 추세선이 유효하고 현재가가 추세선 위 (또는 추세선 근처)
4. 시장 지표가 상승 방향 (또는 중립)
5. 저항선까지 여유가 있음 (최소 0.5% 이상) 또는 저항선이 없음

**"short" 추천 조건 (5개 중 3개 이상 만족 시 추천 가능)**:
1. 예측 모델이 하락 방향 (30분 또는 1시간 중 하나라도 -0.3% 이하)
2. 기술적 지표 2개 이상이 하락 신호 (MA 아래, 데드크로스, RSI > 30 등)
3. 하락 추세선이 유효하고 현재가가 추세선 아래 (또는 추세선 근처)
4. 시장 지표가 하락 방향 (또는 중립)
5. 지지선까지 여유가 있음 (최소 0.5% 이상) 또는 지지선이 없음

**"waiting" 선택 조건 (다음 중 하나라도 해당)**:
- 롱/숏 조건을 3개 이상 만족하지 않음
- 신호가 심하게 모순됨 (대부분의 지표가 서로 반대 방향)
- 데이터가 부족하거나 불확실함
- 리스크가 수익보다 현저히 큼
- 지지/저항선과 매우 가까워 불확실함 (0.2% 이내)

## 📝 응답 형식

다음 형식으로 JSON 응답을 제공하세요:

{{
  "waiting": ["관망 시 유의할 점 1", "관망 시 유의할 점 2", ...],
  "long": ["롱 포지션 시 유의할 점 1", "롱 포지션 시 유의할 점 2", ...],
  "short": ["숏 포지션 시 유의할 점 1", "숏 포지션 시 유의할 점 2", ...],
  "summary": "5단계 검증 절차를 거친 종합적인 시장 의견 (각 단계의 판단 결과 포함)",
  "recommendation": "waiting" 또는 "long" 또는 "short",
  "next_timing": "다음 매수/매도 타이밍 설명 (recommendation이 'waiting'일 때만 제공, 구체적인 조건 명시)",
  "target_price": 목표금액 숫자 (recommendation이 'long' 또는 'short'일 때만 제공, 현재가 대비 2-5% 수준),
  "stop_loss_price": 손절금액 숫자 (recommendation이 'long' 또는 'short'일 때만 제공, 현재가 대비 1-3% 수준)
}}

**중요 지침**:
1. **반드시 5단계 검증 절차를 순서대로 수행하고, 각 단계의 판단 결과를 summary에 포함하세요.**
2. **신호가 명확하지 않으면 "waiting"을 선택하되, 롱/숏 조건을 3개 이상 만족하면 해당 방향을 추천하세요.**
3. **"summary"에는 각 단계에서 확인한 내용과 최종 판단 근거를 상세히 작성하세요.**
4. **"recommendation"이 "waiting"인 경우: "next_timing"에 구체적인 조건을 명시하세요 (예: "지지선 $65,000 돌파 및 RSI 50 이상 회복 시", "저항선 $67,000 돌파 및 거래량 증가 시").**
5. **"recommendation"이 "long" 또는 "short"인 경우: "target_price"와 "stop_loss_price"를 현재 가격(${safe_format(current_price)})을 기준으로 구체적인 숫자로 제공하세요.**
6. **각 유의점은 위에서 제공한 데이터를 직접 인용하여 구체적으로 작성하세요.**
7. **한국어로 응답하세요.**
8. **⚠️ 매우 중요: recommendation 값과 관계없이 반드시 "waiting", "long", "short" 세 가지 모두에 대한 유의점을 제공하세요. 현재 추천이 "waiting"이어도 롱 포지션을 고려할 때의 유의점과 숏 포지션을 고려할 때의 유의점을 반드시 작성하세요. 각 유의점은 최소 3개 이상 제공하세요.**
9. **롱/숏 유의점 작성 시: 현재 시장 상황에서 해당 포지션을 진입한다면 어떤 리스크와 주의사항이 있는지, 어떤 조건이 충족되어야 하는지 구체적으로 설명하세요.**"""
    
    # 유사 패턴 지침 추가
    if similar_pattern:
        prompt += "\n8. **유사 패턴이 제공된 경우: 해당 패턴의 일반적인 특성을 참고하되, 다른 지표들과 충돌하면 패턴보다 다른 지표를 우선하세요.**"
    
    return prompt
    
    return prompt


def _parse_gemini_response(response_text: str) -> Dict:
    """Gemini API 응답 파싱"""
    json_text = response_text.strip()
    
    # 마크다운 코드 블록 제거
    if '```json' in json_text:
        json_text = json_text.split('```json')[1].split('```')[0].strip()
    elif '```' in json_text:
        json_text = json_text.split('```')[1].split('```')[0].strip()
    
    try:
        analysis_result = json.loads(json_text)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 정규식으로 추출
        import re
        json_match = re.search(r'\{[\s\S]*\}', json_text)
        if json_match:
            analysis_result = json.loads(json_match.group(0))
        else:
            # 완전히 실패한 경우 기본 구조 반환
            analysis_result = {
                'waiting': ['응답을 파싱할 수 없습니다.'],
                'long': ['원본 응답: ' + json_text[:200]],
                'short': ['응답 형식 오류'],
                'summary': json_text[:500]
            }
    
    # 필수 필드 검증
    if not analysis_result.get('waiting') and not analysis_result.get('long') and not analysis_result.get('short'):
        analysis_result = {
            'waiting': analysis_result.get('waiting', ['데이터 부족으로 분석 불가']),
            'long': analysis_result.get('long', ['데이터 부족으로 분석 불가']),
            'short': analysis_result.get('short', ['데이터 부족으로 분석 불가']),
            'summary': analysis_result.get('summary', '분석 결과를 구조화할 수 없습니다.')
        }
    
    # recommendation 필드 검증
    recommendation = str(analysis_result.get('recommendation', '')).lower()
    if recommendation not in ['waiting', 'long', 'short']:
        print(f"⚠️ 잘못된 recommendation 값: {recommendation}, 기본값 'waiting' 사용")
        # recommendation이 없거나 잘못된 경우, summary나 다른 데이터를 기반으로 추론 시도
        summary_text = str(analysis_result.get('summary', '')).lower()
        if '롱' in summary_text or '상승' in summary_text or '매수' in summary_text:
            recommendation = 'long'
        elif '숏' in summary_text or '하락' in summary_text or '매도' in summary_text:
            recommendation = 'short'
        else:
            recommendation = 'waiting'
    
    analysis_result['recommendation'] = recommendation
    
    # 추가 필드 검증 및 기본값 설정
    if recommendation == 'waiting':
        # 관망일 때는 next_timing이 있어야 함
        if not analysis_result.get('next_timing'):
            analysis_result['next_timing'] = '시장 상황을 지속적으로 모니터링하세요.'
    elif recommendation in ['long', 'short']:
        # 매수/매도 추천일 때는 목표가와 손절가가 있어야 함
        if not analysis_result.get('target_price'):
            analysis_result['target_price'] = None
        if not analysis_result.get('stop_loss_price'):
            analysis_result['stop_loss_price'] = None
    
    return analysis_result


@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    print('클라이언트 연결됨')
    emit('connected', {'message': '연결 성공'})


@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    print('클라이언트 연결 해제됨')


if __name__ == '__main__':
    print("=" * 60)
    print("백엔드 API 서버 시작")
    print("=" * 60)
    print("포트: 5333")
    print("WebSocket 지원: 활성화")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5333, debug=True)

"""
기술적 지표 및 파생변수 계산 모듈
RSI, CCI, 볼린저밴드 및 효과적인 파생변수 생성
"""
import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """기술적 지표 및 파생변수 계산 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index) 계산
        
        Args:
            prices: 가격 시리즈 (보통 close 가격)
            period: RSI 계산 기간 (기본값: 14)
        
        Returns:
            RSI 값 시리즈
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        gain = gain.fillna(0).replace([np.inf, -np.inf], 0)
        loss = loss.fillna(0).replace([np.inf, -np.inf], 0)
        
        # loss가 0에 가까운 경우를 안전하게 처리
        # RSI 공식: RSI = 100 - (100 / (1 + RS))
        # loss가 0이면 RS = inf가 되므로, loss가 매우 작을 때는 특별 처리
        loss_safe = loss.copy()
        loss_safe[loss_safe < 1e-6] = 1e-6  # loss가 너무 작으면 최소값으로
        
        rs = gain / loss_safe
        # RS가 너무 크면 제한 (RS > 100이면 RSI는 거의 100에 가까움)
        rs = np.clip(rs, 0, 1000)  # RS를 합리적 범위로 제한
        
        rsi = 100 - (100 / (1 + rs))
        # RSI는 0-100 범위여야 함
        rsi = np.clip(rsi, 0, 100)
        rsi = rsi.fillna(50)  # NaN이 있으면 중간값으로
        
        return rsi
    
    def calculate_cci(self, 
                     high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series, 
                     period: int = 20) -> pd.Series:
        """
        CCI (Commodity Channel Index) 계산
        
        Args:
            high: 고가 시리즈
            low: 저가 시리즈
            close: 종가 시리즈
            period: CCI 계산 기간 (기본값: 20)
        
        Returns:
            CCI 값 시리즈
        """
        tp = (high + low + close) / 3  # Typical Price
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean() if len(x) > 0 else 0
        )
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        sma = sma.fillna(0).replace([np.inf, -np.inf], 0)
        mad = mad.fillna(0).replace([np.inf, -np.inf], 0)
        
        # MAD가 0에 가까운 경우를 안전하게 처리
        mad_safe = mad.copy()
        mad_safe[mad_safe < 1e-6] = 1e-6  # MAD가 너무 작으면 최소값으로
        
        cci = (tp - sma) / (0.015 * mad_safe)
        # CCI는 보통 -200 ~ +200 범위이지만 더 넓게 제한
        cci = np.clip(cci, -1000, 1000)
        cci = cci.fillna(0)  # NaN이 있으면 0으로
        
        return cci
    
    def calculate_bollinger_bands(self, 
                                 prices: pd.Series, 
                                 period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """
        볼린저 밴드 계산
        
        Args:
            prices: 가격 시리즈 (보통 close 가격)
            period: 이동평균 기간 (기본값: 20)
            std_dev: 표준편차 배수 (기본값: 2.0)
        
        Returns:
            DataFrame with columns: 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position'
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # rolling 결과가 NaN이거나 무한대일 수 있으므로 안전하게 처리
        sma = sma.fillna(0).replace([np.inf, -np.inf], 0)
        std = std.fillna(0).replace([np.inf, -np.inf], 0)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # sma가 0에 가까운 경우 안전하게 처리
        sma_safe = sma.copy()
        sma_safe[sma_safe < 1e-6] = 1e-6
        
        width = (upper - lower) / sma_safe
        width = np.clip(width, 0, 10)  # 합리적 범위로 제한
        width = width.fillna(0).replace([np.inf, -np.inf], 0)
        
        band_width = upper - lower
        # band_width가 0에 가까운 경우 안전하게 처리
        band_width_safe = band_width.copy()
        band_width_safe[band_width_safe < 1e-6] = 1e-6
        
        position = (prices - lower) / band_width_safe
        position = np.clip(position, 0, 1)  # 0~1 범위로 제한
        position = position.fillna(0.5).replace([np.inf, -np.inf], 0.5)
        
        bb_df = pd.DataFrame({
            'bb_middle': sma,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_width': width,
            'bb_position': position
        })
        
        return bb_df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        효과적인 파생변수 계산
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            파생변수가 추가된 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # 가격 변화율 (안전하게 처리)
        # pct_change는 이전 값이 0이면 무한대 발생 가능
        close_prev = df['close'].shift(1)
        close_prev_safe = close_prev.copy()
        close_prev_safe[close_prev_safe < 1e-6] = 1e-6
        features['price_change'] = np.clip((df['close'] - close_prev) / close_prev_safe, -1, 1)
        
        close_prev5 = df['close'].shift(5)
        close_prev5_safe = close_prev5.copy()
        close_prev5_safe[close_prev5_safe < 1e-6] = 1e-6
        features['price_change_5'] = np.clip((df['close'] - close_prev5) / close_prev5_safe, -1, 1)
        
        close_prev10 = df['close'].shift(10)
        close_prev10_safe = close_prev10.copy()
        close_prev10_safe[close_prev10_safe < 1e-6] = 1e-6
        features['price_change_10'] = np.clip((df['close'] - close_prev10) / close_prev10_safe, -1, 1)
        
        # 안전한 나누기 함수 (무한대 방지)
        def safe_divide(numerator, denominator, default=0.0, min_denom=1e-6):
            """분모가 너무 작으면 기본값 반환, 그 외에는 나누기 수행 후 클리핑"""
            denominator_safe = denominator.copy()
            denominator_safe[denominator_safe < min_denom] = min_denom
            result = numerator / denominator_safe
            # 합리적 범위로 제한 (무한대 방지)
            result = np.clip(result, -1e6, 1e6)
            return result
        
        # 고가-저가 범위 (변동성)
        close_safe = df['close'].copy()
        close_safe[close_safe < 1e-6] = 1e-6
        features['high_low_ratio'] = safe_divide(df['high'] - df['low'], close_safe)
        features['high_low_range'] = df['high'] - df['low']
        
        # 종가 대비 위치
        hl_range = df['high'] - df['low']
        features['close_position'] = safe_divide(df['close'] - df['low'], hl_range, default=0.5)
        features['close_position'] = np.clip(features['close_position'], 0, 1)
        
        # 거래량 관련 (안전하게 처리)
        volume_prev = df['volume'].shift(1)
        volume_prev_safe = volume_prev.copy()
        volume_prev_safe[volume_prev_safe < 1e-8] = 1e-8
        features['volume_change'] = np.clip((df['volume'] - volume_prev) / volume_prev_safe, -1, 10)
        
        volume_ma = df['volume'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['volume_ma_ratio'] = safe_divide(df['volume'], volume_ma, default=1.0)
        
        # 이동평균 대비 가격 위치 (안전하게 처리)
        ma5 = df['close'].rolling(5).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma10 = df['close'].rolling(10).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma20 = df['close'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['price_ma5_ratio'] = safe_divide(df['close'], ma5, default=1.0)
        features['price_ma10_ratio'] = safe_divide(df['close'], ma10, default=1.0)
        features['price_ma20_ratio'] = safe_divide(df['close'], ma20, default=1.0)
        
        # 이동평균선 간격
        features['ma5_ma10_diff'] = safe_divide(ma5 - ma10, close_safe)
        features['ma10_ma20_diff'] = safe_divide(ma10 - ma20, close_safe)
        
        # 모멘텀 지표
        close_shift5 = df['close'].shift(5)
        close_shift10 = df['close'].shift(10)
        features['momentum_5'] = safe_divide(df['close'], close_shift5, default=1.0) - 1
        features['momentum_10'] = safe_divide(df['close'], close_shift10, default=1.0) - 1
        
        # 변동성 (ATR 스타일)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr_ma = tr.rolling(14).mean().fillna(0).replace([np.inf, -np.inf], 0)
        features['atr'] = safe_divide(tr_ma, close_safe)
        
        # 캔들 패턴 (간단한 형태)
        features['body_size'] = safe_divide(abs(df['close'] - df['open']), close_safe)
        features['upper_shadow'] = safe_divide(df['high'] - df[['open', 'close']].max(axis=1), close_safe)
        features['lower_shadow'] = safe_divide(df[['open', 'close']].min(axis=1) - df['low'], close_safe)
        
        # ===== 방향 정확도 개선을 위한 핵심 특징 추가 =====
        
        # 1. 가격 변화율의 가속도 (2차 미분) - 방향 전환 예측에 핵심
        price_change_prev = features['price_change'].shift(1)
        features['price_acceleration'] = features['price_change'] - price_change_prev
        features['price_acceleration'] = np.clip(features['price_acceleration'], -0.5, 0.5)
        
        # 2. 모멘텀의 변화율 - 모멘텀이 가속/감속하는지
        momentum_5_prev = features['momentum_5'].shift(1)
        features['momentum_change'] = features['momentum_5'] - momentum_5_prev
        features['momentum_change'] = np.clip(features['momentum_change'], -0.5, 0.5)
        
        # 3. 추세 강도의 변화 - 추세가 강해지거나 약해지는지
        trend_strength_20 = safe_divide(df['close'] - ma20, ma20)
        features['trend_strength_20_derived'] = trend_strength_20  # 임시로 저장 (calculate_trend_seasonal_volatility의 것과 구분)
        trend_strength_20_prev = trend_strength_20.shift(1)
        features['trend_strength_change'] = trend_strength_20 - trend_strength_20_prev
        features['trend_strength_change'] = np.clip(features['trend_strength_change'], -0.5, 0.5)
        
        # 4. 변동성의 변화 방향 - 변동성이 증가/감소하는지
        atr_prev = features['atr'].shift(1)
        atr_prev_safe = atr_prev.copy()
        atr_prev_safe[atr_prev_safe < 1e-8] = 1e-8
        features['volatility_change'] = safe_divide(features['atr'] - atr_prev, atr_prev_safe)
        features['volatility_change'] = np.clip(features['volatility_change'], -1, 1)
        
        # 5. 거래량과 가격 변화의 상관관계 - 거래량 증가 시 가격 방향
        volume_change = features['volume_change']
        price_change = features['price_change']
        # 거래량과 가격 변화의 곱 (같은 방향이면 양수, 반대면 음수)
        features['volume_price_correlation'] = volume_change * price_change
        features['volume_price_correlation'] = np.clip(features['volume_price_correlation'], -1, 1)
        
        # 6. 가격 위치 (최근 고점/저점 대비) - 지지/저항 레벨
        # 최근 20기간 고점/저점
        high_20 = df['high'].rolling(20).max().fillna(0)
        low_20 = df['low'].rolling(20).min().fillna(0)
        high_low_range_20 = high_20 - low_20
        high_low_range_20_safe = high_low_range_20.copy()
        high_low_range_20_safe[high_low_range_20_safe < 1e-6] = 1e-6
        features['price_position_20'] = safe_divide(df['close'] - low_20, high_low_range_20_safe)
        features['price_position_20'] = np.clip(features['price_position_20'], 0, 1)
        
        # ===== 지지/저항 레벨 특징 추가 (데이터 누수 방지) =====
        # 여러 기간의 지지/저항 레벨 계산 (과거 데이터만 사용)
        # 현재 시점을 제외한 이전 시점까지의 데이터만 사용하여 데이터 누수 방지
        support_resistance_periods = [20, 50, 100]
        
        for period in support_resistance_periods:
            # 이전 시점까지의 period 기간의 고점(저항)과 저점(지지) 계산
            # shift(1)을 사용하여 현재 시점을 제외하고 이전 시점까지의 데이터만 사용
            # rolling(period)는 현재 시점을 포함하므로, shift(1)로 한 시점 뒤로 밀어서
            # 현재 시점에서는 이전 시점까지의 레벨을 참조하도록 함
            resistance_level = df['high'].shift(1).rolling(period).max().fillna(df['close'].shift(1).fillna(df['close']))
            support_level = df['low'].shift(1).rolling(period).min().fillna(df['close'].shift(1).fillna(df['close']))
            
            # 현재 가격과 지지/저항 레벨 간의 거리 (정규화)
            # 현재 시점의 close를 사용하되, 레벨은 이전 시점까지의 데이터로 계산됨
            close_safe = df['close'].copy()
            close_safe[close_safe < 1e-6] = 1e-6
            
            # 저항 레벨까지의 거리 (정규화)
            resistance_distance = safe_divide(resistance_level - df['close'], close_safe)
            resistance_distance = np.clip(resistance_distance, 0, 1)  # 0~1 범위로 제한
            features[f'resistance_distance_{period}'] = resistance_distance
            
            # 지지 레벨까지의 거리 (정규화)
            support_distance = safe_divide(df['close'] - support_level, close_safe)
            support_distance = np.clip(support_distance, 0, 1)  # 0~1 범위로 제한
            features[f'support_distance_{period}'] = support_distance
            
            # 지지/저항 레벨 대비 현재 가격 위치 (0=지지, 1=저항)
            # 이전 시점까지의 레벨과 이전 시점의 close를 사용하여 계산
            close_prev = df['close'].shift(1).fillna(df['close'])
            sr_range = resistance_level - support_level
            sr_range_safe = sr_range.copy()
            sr_range_safe[sr_range_safe < 1e-6] = 1e-6
            features[f'sr_position_{period}'] = safe_divide(close_prev - support_level, sr_range_safe)
            features[f'sr_position_{period}'] = np.clip(features[f'sr_position_{period}'], 0, 1)
            
            # 지지/저항 레벨 근처에 있는지 여부 (임계값: 2%)
            # 이전 시점까지의 레벨과 이전 시점의 close를 사용하여 계산
            threshold = 0.02
            # 이전 시점의 close와 레벨 간의 거리 계산
            close_prev_safe = close_prev.copy()
            close_prev_safe[close_prev_safe < 1e-6] = 1e-6
            resistance_distance_prev = safe_divide(resistance_level - close_prev, close_prev_safe)
            resistance_distance_prev = np.clip(resistance_distance_prev, 0, 1)
            support_distance_prev = safe_divide(close_prev - support_level, close_prev_safe)
            support_distance_prev = np.clip(support_distance_prev, 0, 1)
            
            near_resistance = (resistance_distance_prev <= threshold).astype(float)
            near_support = (support_distance_prev <= threshold).astype(float)
            features[f'near_resistance_{period}'] = near_resistance
            features[f'near_support_{period}'] = near_support
            
            # 지지/저항 레벨 강도 (해당 레벨 근처에서 가격이 몇 번 터치했는지)
            # 각 시점에서 그 시점 이전의 period 기간 동안 저항/지지 레벨 근처에 가격이 있었던 횟수
            # 데이터 누수 방지를 위해 각 시점의 레벨은 해당 시점을 제외한 이전 데이터로만 계산
            
            # 각 시점에서 그 시점을 제외한 이전 period 기간의 레벨 계산
            # shift(1)로 현재 시점을 제외하고, rolling(period)로 이전 period 기간의 최대/최소값 계산
            resistance_level_for_strength = df['high'].shift(1).rolling(period).max().fillna(df['close'].shift(1).fillna(df['close']))
            support_level_for_strength = df['low'].shift(1).rolling(period).min().fillna(df['close'].shift(1).fillna(df['close']))
            
            # 각 시점에서 그 시점의 레벨을 기준으로 임계값 계산
            resistance_threshold_for_strength = resistance_level_for_strength * (1 - threshold)
            support_threshold_for_strength = support_level_for_strength * (1 + threshold)
            
            # 각 시점에서 그 시점의 high/low가 레벨 근처에 있었는지 확인
            # 각 시점 t에서: df['high'][t] >= resistance_level_for_strength[t] * (1 - threshold)
            # resistance_level_for_strength[t]는 t-1 시점까지의 레벨이므로, t 시점의 high와 비교하면
            # 현재 시점의 정보를 사용하게 되어 데이터 누수가 발생함
            # 따라서 각 시점 t에서 t-1 시점의 high/low와 t-1 시점까지의 레벨을 비교해야 함
            # 즉, shift(1)을 적용하여 이전 시점의 high/low와 비교
            near_resistance_mask = (df['high'].shift(1) >= resistance_threshold_for_strength).astype(float)
            near_support_mask = (df['low'].shift(1) <= support_threshold_for_strength).astype(float)
            
            # 과거 period 기간 동안의 터치 횟수 집계
            # 각 시점에서 이전 period 기간 동안의 터치 횟수를 계산
            # shift(1)이 적용되어 있으므로, rolling(period)는 현재 시점을 제외한 이전 period 기간을 고려
            resistance_touches = near_resistance_mask.rolling(period).sum().fillna(0)
            support_touches = near_support_mask.rolling(period).sum().fillna(0)
            
            # 정규화 (period로 나누어 0~1 범위로)
            period_safe = max(period, 1)
            features[f'resistance_strength_{period}'] = np.clip(
                resistance_touches / period_safe, 0, 1
            )
            features[f'support_strength_{period}'] = np.clip(
                support_touches / period_safe, 0, 1
            )
        
        # 지지/저항 레벨 간격 (변동성 대리 지표)
        # 짧은 기간과 긴 기간의 지지/저항 레벨 차이
        resistance_20 = df['high'].rolling(20).max().fillna(df['close'])
        support_20 = df['low'].rolling(20).min().fillna(df['close'])
        resistance_100 = df['high'].rolling(100).max().fillna(df['close'])
        support_100 = df['low'].rolling(100).min().fillna(df['close'])
        
        sr_range_20 = resistance_20 - support_20
        sr_range_100 = resistance_100 - support_100
        
        close_safe = df['close'].copy()
        close_safe[close_safe < 1e-6] = 1e-6
        
        # 정규화된 레벨 간격
        sr_range_20_norm = safe_divide(sr_range_20, close_safe)
        sr_range_100_norm = safe_divide(sr_range_100, close_safe)
        features['sr_range_20_norm'] = np.clip(sr_range_20_norm, 0, 1)
        features['sr_range_100_norm'] = np.clip(sr_range_100_norm, 0, 1)
        
        # 단기/장기 레벨 간격 비율
        sr_range_100_safe = sr_range_100_norm.copy()
        sr_range_100_safe[sr_range_100_safe < 1e-6] = 1e-6
        features['sr_range_ratio'] = np.clip(
            sr_range_20_norm / sr_range_100_safe, 0, 10
        )
        
        # 7. 가격 변화율의 지속성 - 같은 방향으로 계속 가는지
        price_change_sign = np.sign(features['price_change'])
        price_change_sign_prev = price_change_sign.shift(1)
        features['price_direction_persistence'] = (price_change_sign == price_change_sign_prev).astype(float)
        
        # 8. RSI와 가격 변화의 관계 - 과매수/과매도 구간에서의 방향 전환
        # (RSI는 나중에 추가되므로 여기서는 플레이스홀더)
        # RSI가 70 이상이면 과매수, 30 이하면 과매도
        
        # ===== 세부적인 변화 포착을 위한 추가 특징 =====
        
        # 9. 가격 미세 변화 (1기간, 2기간, 3기간) - 즉각적인 변화 포착
        features['price_change_1'] = features['price_change']  # 이미 있음
        features['price_change_2'] = features['price_change'].rolling(2).sum().fillna(0)  # 2기간 누적 변화
        features['price_change_3'] = features['price_change'].rolling(3).sum().fillna(0)  # 3기간 누적 변화
        
        # 10. 거래량 미세 변화 - 거래량의 즉각적인 변화
        volume_change_prev = features['volume_change'].shift(1)
        features['volume_change_acceleration'] = features['volume_change'] - volume_change_prev
        features['volume_change_acceleration'] = np.clip(features['volume_change_acceleration'], -1, 1)
        
        # 11. 가격-거래량 동시 변화 - 가격과 거래량이 동시에 변하는 정도
        price_volume_simultaneous = features['price_change'] * features['volume_change']
        features['price_volume_simultaneous'] = np.clip(price_volume_simultaneous, -1, 1)
        
        # 12. 가격 변화의 2차 미분 (가속도의 변화) - 세부적인 변화 포착
        price_acceleration_prev = features['price_acceleration'].shift(1)
        features['price_jerk'] = features['price_acceleration'] - price_acceleration_prev  # 가속도의 변화
        features['price_jerk'] = np.clip(features['price_jerk'], -0.5, 0.5)
        
        # 13. 변동성의 미세 변화 - 변동성의 즉각적인 변화
        # volatility_5가 없으면 직접 계산
        if 'volatility_5' not in features.columns:
            # 변동성 직접 계산
            close_prev = df['close'].shift(1)
            close_prev_safe = close_prev.copy()
            close_prev_safe[close_prev_safe < 1e-6] = 1e-6
            returns = np.clip((df['close'] - close_prev) / close_prev_safe, -1, 1)
            volatility_5 = returns.rolling(5).std().fillna(0).replace([np.inf, -np.inf], 0)
            volatility_5 = np.clip(volatility_5, 0, 1)
            features['volatility_5'] = volatility_5
        
        volatility_5_prev = features['volatility_5'].shift(1)
        features['volatility_5_change'] = features['volatility_5'] - volatility_5_prev
        features['volatility_5_change'] = np.clip(features['volatility_5_change'], -1, 1)
        
        # 14. 모멘텀의 미세 변화 - 모멘텀의 즉각적인 변화
        momentum_5_prev = features['momentum_5'].shift(1)
        features['momentum_5_change'] = features['momentum_5'] - momentum_5_prev
        features['momentum_5_change'] = np.clip(features['momentum_5_change'], -1, 1)
        
        # NaN 값만 처리 (무한대는 이미 safe_divide에서 방지됨)
        features = features.fillna(0)
        
        return features
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표와 파생변수를 추가
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            모든 특징이 추가된 DataFrame
        """
        # 원본 데이터 검증 및 무한대 제거
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        # 무한대가 있으면 앞/뒤 값으로 채우기
        for col in df_clean.columns:
            # NaN이 있는지 확인 (가장 확실한 방법)
            nan_values = pd.isna(df_clean[col])
            if nan_values.sum() > 0:  # sum()은 스칼라 정수를 반환
                df_clean[col] = df_clean[col].ffill().bfill()
                # 다시 NaN 확인
                nan_values_after = pd.isna(df_clean[col])
                nan_count_after = nan_values_after.sum()
                if nan_count_after > 0:
                    if nan_count_after < len(df_clean[col]):  # 일부만 NaN
                        median_val = df_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        df_clean[col] = df_clean[col].fillna(median_val)
                    else:  # 모두 NaN
                        df_clean[col] = df_clean[col].fillna(0)
        
        result_df = df_clean.copy()
        
        # RSI
        result_df['rsi'] = self.calculate_rsi(result_df['close'], period=14)
        result_df['rsi_7'] = self.calculate_rsi(result_df['close'], period=7)
        result_df['rsi_21'] = self.calculate_rsi(result_df['close'], period=21)
        
        # CCI
        result_df['cci'] = self.calculate_cci(
            result_df['high'], 
            result_df['low'], 
            result_df['close'], 
            period=20
        )
        result_df['cci_14'] = self.calculate_cci(
            result_df['high'], 
            result_df['low'], 
            result_df['close'], 
            period=14
        )
        
        # 볼린저 밴드
        bb_features = self.calculate_bollinger_bands(result_df['close'], period=20, std_dev=2.0)
        result_df = pd.concat([result_df, bb_features], axis=1)
        
        # 추가 볼린저 밴드 (다른 기간)
        bb_features_10 = self.calculate_bollinger_bands(result_df['close'], period=10, std_dev=2.0)
        bb_features_10.columns = [f'{col}_10' for col in bb_features_10.columns]
        result_df = pd.concat([result_df, bb_features_10], axis=1)
        
        # 파생변수
        derived_features = self.calculate_derived_features(df)
        result_df = pd.concat([result_df, derived_features], axis=1)
        
        # RSI 기반 방향성 특징 제거 (price_change 기반이므로 미래 정보 누수 가능)
        # rsi_overbought, rsi_oversold, rsi_price_divergence는 제거됨
        
        # 트렌드, 계절성, 변동성 특징 추가
        trend_seasonal_features = self.calculate_trend_seasonal_volatility(df)
        result_df = pd.concat([result_df, trend_seasonal_features], axis=1)
        
        # trend_strength_20가 생성된 후에 trend_strength_change_20 계산
        if 'trend_strength_20' in result_df.columns:
            result_df['trend_strength_change_20'] = result_df['trend_strength_20'].diff().fillna(0).replace([np.inf, -np.inf], 0)
        
        # 최종 검증: 무한대 값이 있는지 확인하고 제거
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # 각 컬럼별로 무한대를 중앙값으로 대체
        for col in result_df.columns:
            try:
                col_data = result_df[col]
                # Series인지 확인
                if not isinstance(col_data, pd.Series):
                    continue
                
                # NaN이 있는지 확인 (안전한 방법)
                nan_count = int(col_data.isna().sum())
                if nan_count > 0:
                    # 앞/뒤로 채우기
                    result_df[col] = col_data.ffill().bfill()
                    # 여전히 NaN이 있으면 중앙값으로
                    remaining_nan = int(result_df[col].isna().sum())
                    if remaining_nan > 0:
                        median_val = result_df[col].median()
                        # median_val이 스칼라인지 확인
                        if isinstance(median_val, pd.Series):
                            median_val = median_val.iloc[0] if len(median_val) > 0 else 0
                        if pd.isna(median_val) or (isinstance(median_val, (int, float)) and np.isinf(median_val)):
                            median_val = 0
                        result_df[col] = result_df[col].fillna(median_val)
            except Exception as e:
                # 에러 발생 시 해당 컬럼을 0으로 채우기
                print(f"Warning: Error processing column {col}: {e}")
                try:
                    result_df[col] = result_df[col].fillna(0)
                except:
                    pass
        
        # 최종적으로 NaN을 0으로
        result_df = result_df.fillna(0)
        
        # 최종 무한대 검증 (있으면 에러)
        if result_df.replace([np.inf, -np.inf], np.nan).isna().any().any():
            # 어떤 컬럼에 무한대가 있는지 찾기
            inf_cols = []
            for col in result_df.columns:
                if np.isinf(result_df[col]).any():
                    inf_cols.append(col)
            raise ValueError(f"무한대 값이 여전히 존재합니다: {inf_cols}")
        
        return result_df
    
    def calculate_trend_seasonal_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        트렌드, 계절성, 변동성 클러스터링 특징 계산
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            트렌드, 계절성, 변동성 특징 DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. 트렌드 특징
        # 선형 트렌드 (단순 이동평균 기울기)
        ma20 = df['close'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma50 = df['close'].rolling(50).mean().fillna(0).replace([np.inf, -np.inf], 0)
        ma100 = df['close'].rolling(100).mean().fillna(0).replace([np.inf, -np.inf], 0)
        
        # 안전한 나누기 함수
        def safe_divide(numerator, denominator, default=0.0, min_denom=1e-6):
            denominator_safe = denominator.copy()
            denominator_safe[denominator_safe < min_denom] = min_denom
            result = numerator / denominator_safe
            result = np.clip(result, -1e6, 1e6)
            return result
        
        features['trend_ma20_slope'] = safe_divide(ma20.diff(5), ma20.shift(5))
        features['trend_ma50_slope'] = safe_divide(ma50.diff(10), ma50.shift(10))
        features['trend_ma100_slope'] = safe_divide(ma100.diff(20), ma100.shift(20))
        
        # 트렌드 강도 (가격이 이동평균 위/아래에 있는지)
        features['trend_strength_20'] = safe_divide(df['close'] - ma20, ma20)
        features['trend_strength_50'] = safe_divide(df['close'] - ma50, ma50)
        features['trend_strength_100'] = safe_divide(df['close'] - ma100, ma100)
        
        # 이동평균 간 관계 (골든 크로스, 데드 크로스)
        features['ma_cross_20_50'] = safe_divide(ma20 - ma50, ma50)
        features['ma_cross_50_100'] = safe_divide(ma50 - ma100, ma100)
        
        # 2. 계절성 특징 (시간대별 패턴) - 제거됨 (데이터 누수 방지)
        # hour, day_of_week, day_of_month, hour_seasonality, dow_seasonality는 
        # 다음 시점과 높은 일치율을 보여 미래 정보 누수 가능성이 있음
        # 따라서 제외함
        
        # 3. 변동성 클러스터링 특징
        # GARCH 스타일 변동성 (과거 변동성이 미래 변동성에 영향)
        # pct_change 대신 안전한 방법 사용
        close_prev = df['close'].shift(1)
        close_prev_safe = close_prev.copy()
        close_prev_safe[close_prev_safe < 1e-6] = 1e-6
        returns = np.clip((df['close'] - close_prev) / close_prev_safe, -1, 1)
        
        volatility_5 = returns.rolling(5).std()
        volatility_20 = returns.rolling(20).std()
        volatility_60 = returns.rolling(60).std()
        
        # 변동성이 NaN이거나 무한대일 수 있으므로 안전하게 처리
        volatility_5 = volatility_5.fillna(0).replace([np.inf, -np.inf], 0)
        volatility_20 = volatility_20.fillna(0).replace([np.inf, -np.inf], 0)
        volatility_60 = volatility_60.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 변동성을 합리적 범위로 제한
        volatility_5 = np.clip(volatility_5, 0, 1)
        volatility_20 = np.clip(volatility_20, 0, 1)
        volatility_60 = np.clip(volatility_60, 0, 1)
        
        features['volatility_5'] = volatility_5
        features['volatility_20'] = volatility_20
        features['volatility_60'] = volatility_60
        
        # 변동성 비율 (단기/장기)
        volatility_20_safe = volatility_20.copy()
        volatility_20_safe[volatility_20_safe < 1e-8] = 1e-8
        volatility_60_safe = volatility_60.copy()
        volatility_60_safe[volatility_60_safe < 1e-8] = 1e-8
        features['volatility_ratio_5_20'] = np.clip(volatility_5 / volatility_20_safe, 0, 10)
        features['volatility_ratio_20_60'] = np.clip(volatility_20 / volatility_60_safe, 0, 10)
        
        # 변동성 추세 (변동성이 증가/감소하는지)
        vol5_shift = volatility_5.shift(5)
        vol5_shift_safe = vol5_shift.copy()
        vol5_shift_safe[vol5_shift_safe < 1e-8] = 1e-8
        vol20_shift = volatility_20.shift(10)
        vol20_shift_safe = vol20_shift.copy()
        vol20_shift_safe[vol20_shift_safe < 1e-8] = 1e-8
        features['volatility_trend_5'] = np.clip(volatility_5.diff(5) / vol5_shift_safe, -1, 1)
        features['volatility_trend_20'] = np.clip(volatility_20.diff(10) / vol20_shift_safe, -1, 1)
        
        # 변동성 클러스터링 (최근 변동성이 높으면 계속 높을 가능성)
        features['volatility_cluster'] = (volatility_5 > volatility_20).astype(float)
        # 변동성 레짐 제거 (volatility_regime은 다음 시점과 높은 일치율로 데이터 누수 가능)
        # volatility_regime은 제외됨
        
        # 4. 가격 모멘텀과 변동성의 관계
        close_shift5_safe = df['close'].shift(5).copy()
        close_shift5_safe[close_shift5_safe < 1e-6] = 1e-6
        momentum_5 = np.clip(df['close'] / close_shift5_safe - 1, -1, 1)
        features['momentum_volatility'] = np.clip(momentum_5 * volatility_5, -1, 1)
        
        # 5. 거래량과 변동성의 관계
        volume_ma = df['volume'].rolling(20).mean().fillna(0).replace([np.inf, -np.inf], 0)
        volume_ma_safe = volume_ma.copy()
        volume_ma_safe[volume_ma_safe < 1e-8] = 1e-8
        features['volume_volatility'] = np.clip((df['volume'] / volume_ma_safe) * volatility_20, 0, 10)
        
        # 6. 추가 유용한 특징 (Val Loss 수렴을 위한 특징)
        # 가격 추세 강도 (최근 가격 변화의 지속성)
        price_change_1 = df['close'].diff(1)
        price_change_5 = df['close'].diff(5)
        price_change_10 = df['close'].diff(10)
        close_safe = df['close'].copy()
        close_safe[close_safe < 1e-6] = 1e-6
        features['trend_persistence'] = np.clip(
            (price_change_1 * price_change_5) / (close_safe * close_safe), -1, 1
        )  # 같은 방향이면 양수, 반대면 음수
        
        # 가격 가속도 (변화율의 변화율)
        price_change_prev = price_change_1.shift(1)
        price_change_prev_safe = price_change_prev.copy()
        price_change_prev_safe[price_change_prev_safe.abs() < 1e-6] = np.sign(price_change_prev_safe) * 1e-6
        features['price_acceleration'] = np.clip(
            price_change_1 / price_change_prev_safe - 1, -10, 10
        )
        
        # 거래량 가격 상관관계 (거래량 증가 시 가격 변화)
        volume_change = df['volume'].diff(1)
        volume_change_safe = volume_change.copy()
        volume_change_safe[volume_change_safe.abs() < 1e-8] = np.sign(volume_change_safe) * 1e-8
        features['volume_price_correlation'] = np.clip(
            (price_change_1 / close_safe) / (volume_change_safe / (df['volume'] + 1e-8)), -10, 10
        )
        
        # 가격 범위 압축/확장 (최근 변동성 대비 현재 변동성)
        recent_vol = returns.rolling(5).std().fillna(0)
        recent_vol_safe = recent_vol.copy()
        recent_vol_safe[recent_vol_safe < 1e-8] = 1e-8
        features['volatility_compression'] = np.clip(volatility_20 / recent_vol_safe, 0, 10)
        
        # 가격 위치 (최근 고점/저점 대비 위치)
        high_20 = df['high'].rolling(20).max().fillna(df['high'])
        low_20 = df['low'].rolling(20).min().fillna(df['low'])
        range_20 = high_20 - low_20
        range_20_safe = range_20.copy()
        range_20_safe[range_20_safe < 1e-6] = 1e-6
        features['price_position_20'] = np.clip((df['close'] - low_20) / range_20_safe, 0, 1)
        
        # 거래량 가중 가격 (VWAP 스타일)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_20 = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        vwap_20 = vwap_20.fillna(df['close'])
        vwap_20_safe = vwap_20.copy()
        vwap_20_safe[vwap_20_safe < 1e-6] = 1e-6
        features['price_vwap_ratio'] = np.clip(df['close'] / vwap_20_safe - 1, -0.5, 0.5)
        
        # NaN 처리만 (무한대는 이미 방지됨)
        features = features.fillna(0)
        
        return features


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    import numpy as np
    
    # 샘플 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    engineer = FeatureEngineer()
    result = engineer.add_all_features(sample_data)
    print(f"원본 컬럼 수: {len(sample_data.columns)}")
    print(f"특징 추가 후 컬럼 수: {len(result.columns)}")
    print("\n컬럼 목록:")
    print(result.columns.tolist())
    print("\n샘플 데이터:")
    print(result.head(10))


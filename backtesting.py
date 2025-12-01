"""
백테스팅 모듈
과거 데이터를 사용하여 모델의 실제 거래 성능을 시뮬레이션
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from data_preprocessor import DataPreprocessor
from model import PatchCNNBiLSTM
from tensorflow import keras


class Backtester:
    """백테스팅 클래스"""
    
    def __init__(self,
                 model_path: str = 'models/best_model.h5',
                 window_size: int = 60,
                 min_change_30m: float = 0.004,  # 0.4%
                 min_change_1h: float = 0.002,  # 0.1%
                 strong_signal_threshold: float = 0.009,  # 0.9%
                 initial_capital: float = 10000.0,
                 leverage: int = 10,  # 레버리지 배수 (기본값: 10배)
                 commission: float = 0.0004,  # 0.04% 수수료 (선물 거래)
                 cooldown_minutes: int = 15,  # 손실 후 쿨다운 시간
                 use_market_indicators: bool = True):  # 시장 지표 사용 여부
        """
        Args:
            model_path: 학습된 모델 파일 경로
            window_size: 슬라이딩 윈도우 크기
            min_change_30m: 30분봉 최소 변화율 (기본값: 0.4%)
            min_change_1h: 1시간봉 최소 변화율 (기본값: 0.1%)
            strong_signal_threshold: 강한 신호 기준 (기본값: 0.9%)
            initial_capital: 초기 자본금
            leverage: 레버리지 배수 (기본값: 10배)
            commission: 거래 수수료 (기본값: 0.04%, 선물 거래)
            cooldown_minutes: 손실 후 쿨다운 시간 (분, 기본값: 15분)
            use_market_indicators: 시장 지표 사용 여부 (기본값: True)
        """
        self.model_path = model_path
        self.window_size = window_size
        self.min_change_30m = min_change_30m
        self.min_change_1h = min_change_1h
        self.strong_signal_threshold = strong_signal_threshold
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.cooldown_minutes = cooldown_minutes
        self.use_market_indicators = use_market_indicators
        
        # 컴포넌트 초기화
        self.fetcher = BinanceDataFetcher()
        self.engineer = FeatureEngineer()
        
        # 모델 및 전처리기
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        # 거래 기록
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        
        # 쿨다운 및 포지션 추적
        self.trade_cooldown_until = None
        self.last_position_info = None
        
        print("=" * 60)
        print("백테스팅 시스템 초기화 중...")
        print("=" * 60)
        self._load_model()
        print("초기화 완료!\n")
    
    def _load_model(self):
        """학습된 모델 및 전처리기 로드"""
        # 모델 파일 찾기
        if not os.path.exists(self.model_path):
            models_dir = 'models'
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if model_files:
                    model_files.sort(reverse=True)
                    self.model_path = os.path.join(models_dir, model_files[0])
                    print(f"모델 파일 자동 선택: {self.model_path}")
                else:
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            else:
                raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        
        # 전처리기 초기화
        self.preprocessor = DataPreprocessor(
            window_size=self.window_size,
            prediction_horizon=1,
            target_column='close',
            scaler_type='standard'
        )
        
        # 스케일러 로드
        scaler_path = 'models/scalers.pkl'
        if os.path.exists(scaler_path):
            print(f"스케일러 로드 중: {scaler_path}")
            self.preprocessor.load_scalers(scaler_path)
            self.feature_names = self.preprocessor.feature_columns
            if not self.feature_names:
                raise ValueError("스케일러에 feature_columns 정보가 없습니다.")
            
            scaler_expected_features = self.preprocessor.scaler.n_features_in_
            print(f"스케일러가 기대하는 feature 개수: {scaler_expected_features}개")
            self.num_features = scaler_expected_features
        else:
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        
        # 모델 로드
        print(f"모델 로드 중: {self.model_path}")
        
        # 저장된 모델의 실제 feature 개수 확인
        saved_num_features = None
        try:
            import h5py
            with h5py.File(self.model_path, 'r') as f:
                if 'model_weights' in f:
                    # 첫 번째 레이어의 가중치 shape에서 feature 개수 추출
                    layer_names = list(f['model_weights'].keys())
                    if layer_names:
                        first_layer = f['model_weights'][layer_names[0]]
                        if 'conv1d' in layer_names[0].lower() or 'input' in layer_names[0].lower():
                            # Conv1D 레이어의 경우: (kernel_size, n_features, n_filters)
                            weight_keys = list(first_layer.keys())
                            if 'kernel:0' in weight_keys:
                                weight_shape = first_layer['kernel:0'].shape
                                if len(weight_shape) >= 2:
                                    saved_num_features = int(weight_shape[1])
                                    print(f"모델 가중치에서 확인한 feature 개수: {saved_num_features}개")
        except Exception as e:
            print(f"⚠️ 경고: 가중치에서 feature 개수 추출 실패: {e}")
        
        # 모델 구조 재구성
        print("모델 구조 재구성 중...")
        if saved_num_features:
            num_features = saved_num_features
        else:
            num_features = self.num_features
        
        model_builder = PatchCNNBiLSTM(
            input_shape=(self.window_size, num_features),
            num_features=num_features,
            patch_size=5,
            cnn_filters=[],
            lstm_units=128,
            dropout_rate=0.2,
            learning_rate=0.0005
        )
        self.model = model_builder.build_model()
        
        # 가중치 로드
        print("가중치 로드 중...")
        self.model.load_weights(self.model_path)
        print("가중치 로드 완료!")
        print(f"✅ 최종 모델 feature 개수 확인: {num_features}개 (모델 입력 shape: {self.model.input_shape})")
    
    def _prepare_data_for_prediction(self, df_features: pd.DataFrame, start_idx: int) -> Tuple[np.ndarray, float]:
        """
        특정 시점에서 예측을 위한 데이터 준비 (realtime_trading.py와 동일한 로직)
        
        Args:
            df_features: 이미 피처 엔지니어링이 완료된 전체 데이터프레임
            start_idx: 시작 인덱스 (예측할 시점)
        
        Returns:
            X: 입력 시퀀스 (1, window_size, n_features)
            current_price: 현재 가격
        """
        # 필요한 데이터 범위 확인
        if start_idx < self.window_size:
            return None, None
        
        # window_size 개의 데이터 선택 (start_idx 이전의 window_size 개)
        # realtime_trading.py와 동일하게: 마지막 window_size 개를 사용
        start_data_idx = max(0, start_idx - self.window_size)
        end_data_idx = start_idx
        
        # 예측할 시점까지의 데이터만 사용 (미래 데이터 누수 방지)
        df_to_use = df_features.iloc[:end_data_idx].copy()
        
        # 결측치 처리
        df_to_use = df_to_use.ffill().bfill().fillna(0)
        
        # 미래 데이터 누수 방지 (realtime_trading.py와 동일)
        df_clean = self.preprocessor._remove_future_leakage(df_to_use)
        
        # 모델이 실제로 사용한 feature 목록 확인
        model_feature_names = getattr(self.preprocessor, 'model_feature_columns', None)
        if model_feature_names is None:
            model_feature_names = self.feature_names
        
        # 스케일러에 저장된 feature_columns 사용 (정확히 일치해야 함)
        if self.feature_names is None:
            raise ValueError("feature_names가 설정되지 않았습니다. 스케일러를 먼저 로드하세요.")
        
        # 모든 feature가 생성되었는지 확인 (스케일러 feature 기준)
        missing_features = [col for col in self.feature_names if col not in df_clean.columns]
        if missing_features:
            raise ValueError(f"Feature가 생성되지 않았습니다. 데이터가 충분하지 않을 수 있습니다. "
                           f"누락된 feature: {missing_features[:10]}... "
                           f"(총 {len(missing_features)}개).")
        
        # 마지막 window_size 개의 데이터로 시퀀스 생성
        if len(df_clean) < self.window_size:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.window_size}개 필요, 현재 {len(df_clean)}개")
        
        # 먼저 고유한 feature만 선택 (DataFrame에 추가) - realtime_trading.py와 동일
        unique_feature_df = pd.DataFrame(index=df_clean.index)
        missing_cols = []
        
        # 스케일러가 기대하는 feature 순서대로 선택
        for col in self.feature_names:
            if col not in unique_feature_df.columns and col in df_clean.columns:
                try:
                    col_data = df_clean.loc[:, col]
                    if isinstance(col_data, pd.DataFrame):
                        if col_data.shape[1] > 0:
                            col_data = col_data.iloc[:, 0]
                        else:
                            missing_cols.append(col)
                            continue
                    if len(col_data) > 0:
                        unique_feature_df[col] = col_data
                except (KeyError, IndexError, ValueError) as e:
                    missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Feature가 생성되지 않았습니다: {missing_cols[:10]}...")
        
        # 스케일러가 기대하는 feature 순서대로 값을 재배열 (중복 포함)
        # 스케일러는 전체 feature로 스케일링해야 함
        scaler_feature_values = []
        for col in self.feature_names:
            if col in unique_feature_df.columns:
                scaler_feature_values.append(unique_feature_df[col].values)
            else:
                # 누락된 feature는 0으로 채움
                scaler_feature_values.append(np.zeros(len(unique_feature_df)))
        
        # (n_samples, n_features) 형태로 변환 (스케일러용)
        recent_data_for_scaler = np.column_stack(scaler_feature_values)
        
        # 모델이 기대하는 feature만 선택 (순서대로)
        model_feature_values = []
        for col in model_feature_names:
            if col in unique_feature_df.columns:
                model_feature_values.append(unique_feature_df[col].values)
            else:
                # 누락된 feature는 0으로 채움
                model_feature_values.append(np.zeros(len(unique_feature_df)))
        
        # (n_samples, n_features) 형태로 변환 (모델용)
        recent_data_for_model = np.column_stack(model_feature_values)
        
        # 마지막 window_size 개 선택 (realtime_trading.py와 동일)
        recent_data_for_scaler = recent_data_for_scaler[-self.window_size:]
        recent_data_for_model = recent_data_for_model[-self.window_size:]
        
        # 모델이 기대하는 feature 개수 확인 (모델의 실제 입력 shape에서 직접 확인)
        if hasattr(self, 'model') and self.model is not None:
            try:
                model_input_shape = self.model.input_shape
                if model_input_shape and len(model_input_shape) >= 3:
                    model_expected_features = int(model_input_shape[2])  # (batch, timesteps, features)
                else:
                    model_expected_features = getattr(self, 'num_features', None)
                    if model_expected_features is None:
                        model_expected_features = self.preprocessor.scaler.n_features_in_
            except Exception as e:
                model_expected_features = getattr(self, 'num_features', None)
                if model_expected_features is None:
                    model_expected_features = self.preprocessor.scaler.n_features_in_
        else:
            model_expected_features = getattr(self, 'num_features', None)
            if model_expected_features is None:
                model_expected_features = self.preprocessor.scaler.n_features_in_
        
        # 스케일러가 기대하는 feature 개수 확인
        scaler_expected_features = self.preprocessor.scaler.n_features_in_
        model_expected_features = len(model_feature_names)
        
        # 스케일링 (스케일러는 이미 로드되어 있어야 함)
        if not self.preprocessor.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다. models/scalers.pkl 파일을 확인하세요.")
        
        # 스케일러는 전체 feature로 스케일링 (스케일러가 학습한 순서대로) - realtime_trading.py와 동일
        recent_flat_scaler = recent_data_for_scaler.reshape(-1, recent_data_for_scaler.shape[-1])
        recent_scaled_scaler = self.preprocessor.scaler.transform(recent_flat_scaler)
        recent_scaled_scaler = recent_scaled_scaler.reshape(1, self.window_size, -1)
        
        # 스케일링된 데이터에서 모델이 기대하는 feature만 선택
        # model_feature_names의 순서대로 스케일러 feature에서 선택
        if scaler_expected_features != model_expected_features:
            # 모델 feature의 인덱스를 스케일러 feature 목록에서 찾기
            model_feature_indices = []
            for col in model_feature_names:
                if col in self.feature_names:
                    # 스케일러 feature 목록에서 해당 feature의 첫 번째 인덱스 찾기
                    indices = [i for i, f in enumerate(self.feature_names) if f == col]
                    if indices:
                        model_feature_indices.append(indices[0])  # 첫 번째 인덱스만 사용
            
            if len(model_feature_indices) == model_expected_features:
                # 스케일링된 데이터에서 모델 feature만 선택
                recent_scaled = recent_scaled_scaler[:, :, model_feature_indices]
            else:
                # 인덱스 매칭 실패 시 앞부분만 사용
                recent_scaled = recent_scaled_scaler[:, :, :model_expected_features]
        else:
            # feature 개수가 같으면 그대로 사용
            recent_scaled = recent_scaled_scaler
        
        # 현재 가격 (마지막 시점의 close 가격) - realtime_trading.py와 동일
        current_price = float(df_clean['close'].iloc[-1])
        
        return recent_scaled, current_price
    
    def _check_position_close_result(self, previous_position: dict):
        """포지션이 닫혔을 때 손익 확인 및 쿨다운 설정
        
        Args:
            previous_position: 이전 포지션 정보 (side, entry_price, current_price 등)
        """
        try:
            entry_price = previous_position.get('entry_price', 0)
            current_price = previous_position.get('current_price', 0)
            side = previous_position.get('side', 'long')
            
            if entry_price > 0 and current_price > 0:
                # 롱/숏에 따라 ROI 계산
                if side == 'long':
                    final_roi = (current_price - entry_price) / entry_price
                else:  # short
                    final_roi = (entry_price - current_price) / entry_price
                
                # 손실이면 쿨다운 설정
                if final_roi < 0:
                    self.trade_cooldown_until = previous_position.get('timestamp', datetime.now())
                    if isinstance(self.trade_cooldown_until, pd.Timestamp):
                        self.trade_cooldown_until = self.trade_cooldown_until + pd.Timedelta(minutes=self.cooldown_minutes)
                    else:
                        self.trade_cooldown_until = self.trade_cooldown_until + timedelta(minutes=self.cooldown_minutes)
        except Exception as e:
            print(f"⚠️ 포지션 종료 결과 확인 실패: {e}")
    
    def check_trade_conditions(self, change_30m: float, change_1h: float) -> Optional[str]:
        """거래 조건 확인
        
        조건:
        - 롱 신호: 30분봉 >= 0.2%, 1시간봉 >= 0.05%, 둘 다 양수
        - 숏 신호: 30분봉 >= min_change_30m, 1시간봉 >= min_change_1h, 둘 다 음수
        
        Returns:
            'long': 롱 주문 조건 충족
            'short': 숏 주문 조건 충족
            None: 조건 미충족
        """
        epsilon = 1e-8
        
        # 롱 신호 체크 (별도 임계값 사용)
        both_positive = change_30m > 0 and change_1h > 0
        if both_positive:
            # 롱 신호: 30분봉 0.2%, 1시간봉 0.05%
            min_change_30m_long = 0.004  # 0.2%
            min_change_1h_long = 0.002  # 0.05%
            if abs(change_30m) >= min_change_30m_long - epsilon and abs(change_1h) >= min_change_1h_long - epsilon:
                return 'long'
        
        # 숏 신호 체크 (기존 임계값 사용)
        both_negative = change_30m < 0 and change_1h < 0
        if both_negative:
            if abs(change_30m) >= self.min_change_30m - epsilon and abs(change_1h) >= self.min_change_1h - epsilon:
                return 'short'
        
        return None
    
    def _get_market_signal_from_ohlcv(self, df_features: pd.DataFrame, current_idx: int) -> Dict:
        """OHLCV 데이터로부터 시장 지표 신호 생성 (백테스팅용)
        
        Args:
            df_features: 피처 엔지니어링된 데이터프레임
            current_idx: 현재 인덱스
            
        Returns:
            {
                'signal': str,  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
                'confidence': float,  # 신뢰도 (0 ~ 1)
                'reasons': list,  # 신호 근거 리스트
            }
        """
        try:
            # 최근 데이터 추출 (최소 20개 필요)
            lookback = min(20, current_idx)
            if lookback < 10:
                return {'signal': 'neutral', 'confidence': 0.0, 'reasons': []}
            
            recent_data = df_features.iloc[current_idx - lookback:current_idx + 1].copy()
            
            # 필수 컬럼 확인
            required_cols = ['high', 'low', 'close', 'volume']
            if not all(col in recent_data.columns for col in required_cols):
                return {'signal': 'neutral', 'confidence': 0.0, 'reasons': []}
            
            signals = []
            confidence_scores = []
            
            # 1. 변동성 압축 분석 (OHLCV 기반)
            if len(recent_data) >= 14:
                try:
                    high_low = recent_data['high'] - recent_data['low']
                    high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
                    low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
                    
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(window=14).mean()
                    
                    if len(atr) > 0:
                        atr_valid = atr.dropna()
                        if len(atr_valid) >= 2:
                            current_volatility = atr.iloc[-1]
                            avg_volatility = atr_valid.iloc[-min(lookback, len(atr_valid)):].mean()
                            
                            if not pd.isna(current_volatility) and not pd.isna(avg_volatility) and avg_volatility > 0:
                                volatility_ratio = current_volatility / avg_volatility
                                
                                if volatility_ratio < 0.8:  # 조건 완화 (0.7 -> 0.8)
                                    # 변동성 압축 - 폭발 가능성
                                    compression_level = 1.0 - volatility_ratio
                                    if compression_level > 0.2:  # 조건 완화 (0.4 -> 0.2)
                                        signals.append('변동성: 압축 후 폭발 가능성')
                                        confidence_scores.append(0.15)  # 가중치 조정
                except Exception:
                    pass  # 변동성 분석 실패 시 건너뛰기
            
            # 2. 가격 모멘텀 분석
            if len(recent_data) >= 5:
                try:
                    price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5]
                    
                    if len(recent_data) >= 10:
                        volume_trend = recent_data['volume'].iloc[-5:].mean() / recent_data['volume'].iloc[-10:-5].mean()
                    else:
                        volume_trend = 1.0
                    
                    # 조건 완화: 0.5% 이상 변화 + 거래량 1.1배 증가
                    if price_change > 0.005 and volume_trend > 1.1:
                        signals.append('모멘텀: 상승 추세 + 거래량 증가')
                        confidence_scores.append(0.2)
                    elif price_change < -0.005 and volume_trend > 1.1:
                        signals.append('모멘텀: 하락 추세 + 거래량 증가')
                        confidence_scores.append(-0.2)
                    elif price_change > 0.003:  # 추가: 작은 상승 모멘텀
                        signals.append('모멘텀: 상승 추세')
                        confidence_scores.append(0.1)
                    elif price_change < -0.003:  # 추가: 작은 하락 모멘텀
                        signals.append('모멘텀: 하락 추세')
                        confidence_scores.append(-0.1)
                except Exception:
                    pass  # 모멘텀 분석 실패 시 건너뛰기
            
            # 3. RSI 기반 과매수/과매도 분석
            if len(recent_data) >= 14:
                try:
                    delta = recent_data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    
                    gain_valid = gain.dropna()
                    loss_valid = loss.dropna()
                    
                    if len(gain_valid) > 0 and len(loss_valid) > 0:
                        last_gain = gain.iloc[-1]
                        last_loss = loss.iloc[-1]
                        
                        if not pd.isna(last_gain) and not pd.isna(last_loss) and last_loss > 0:
                            rs = last_gain / last_loss
                            rsi = 100 - (100 / (1 + rs))
                            
                            if not pd.isna(rsi):
                                if rsi < 35:  # 조건 완화 (30 -> 35)
                                    signals.append('RSI: 과매도 구간')
                                    confidence_scores.append(0.15)
                                elif rsi > 65:  # 조건 완화 (70 -> 65)
                                    signals.append('RSI: 과매수 구간')
                                    confidence_scores.append(-0.15)
                                elif rsi < 40:  # 추가: 약한 과매도
                                    signals.append('RSI: 약한 과매도')
                                    confidence_scores.append(0.08)
                                elif rsi > 60:  # 추가: 약한 과매수
                                    signals.append('RSI: 약한 과매수')
                                    confidence_scores.append(-0.08)
                except Exception:
                    pass  # RSI 분석 실패 시 건너뛰기
            
            # 4. 추가: 단순 가격 추세 분석
            if len(recent_data) >= 3:
                try:
                    short_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-3]) / recent_data['close'].iloc[-3]
                    if short_trend > 0.002:  # 0.2% 이상 상승
                        signals.append('단기 추세: 상승')
                        confidence_scores.append(0.05)
                    elif short_trend < -0.002:  # 0.2% 이상 하락
                        signals.append('단기 추세: 하락')
                        confidence_scores.append(-0.05)
                except Exception:
                    pass
            
            # 종합 신호 계산
            total_confidence = sum(confidence_scores)
            abs_confidence = abs(total_confidence)
            
            # 신호 강도 판단 (기준 완화)
            if total_confidence >= 0.25:  # 0.3 -> 0.25
                signal = 'strong_buy'
            elif total_confidence >= 0.1:  # 0.15 -> 0.1
                signal = 'buy'
            elif total_confidence <= -0.25:  # -0.3 -> -0.25
                signal = 'strong_sell'
            elif total_confidence <= -0.1:  # -0.15 -> -0.1
                signal = 'sell'
            else:
                signal = 'neutral'
            
            # 신뢰도 정규화 (0 ~ 1)
            confidence = min(abs_confidence, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasons': signals
            }
            
        except Exception as e:
            # 오류 발생 시 중립 신호 반환 (디버깅용으로 예외 메시지 포함)
            import traceback
            print(f"⚠️ 시장 지표 계산 오류 (인덱스 {current_idx}): {e}")
            return {'signal': 'neutral', 'confidence': 0.0, 'reasons': []}
    
    def run_backtest(self,
                     start_date: datetime,
                     end_date: datetime,
                     timeframe: str = '5m') -> Dict:
        """
        백테스팅 실행
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            timeframe: 시간 프레임
        
        Returns:
            백테스팅 결과 딕셔너리
        """
        print("=" * 60)
        print("백테스팅 시작")
        print("=" * 60)
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"초기 자본금: ${self.initial_capital:,.2f}")
        print(f"레버리지: {self.leverage}배")
        print(f"30분봉 최소 변화율: {self.min_change_30m*100:.2f}%")
        print(f"1시간봉 최소 변화율: {self.min_change_1h*100:.2f}%")
        print(f"강한 신호 기준: {self.strong_signal_threshold*100:.2f}%")
        print(f"손실 후 쿨다운: {self.cooldown_minutes}분")
        print()
        
        # 데이터 수집
        print("데이터 수집 중...")
        
        # 필요한 데이터 개수 계산 (기간에 맞게)
        # 5분봉 기준: 1일 = 288개, 여유분 포함하여 충분히 가져오기
        days_diff = (end_date - start_date).days + 1
        if timeframe == '5m':
            candles_per_day = 288  # 24시간 * 60분 / 5분
        elif timeframe == '1m':
            candles_per_day = 1440  # 24시간 * 60분
        elif timeframe == '15m':
            candles_per_day = 96  # 24시간 * 60분 / 15분
        elif timeframe == '1h':
            candles_per_day = 24  # 24시간
        else:
            candles_per_day = 288  # 기본값 (5분봉)
        
        # 필요한 개수 + 여유분 (window_size + warm-up 포함)
        required_limit = int(days_diff * candles_per_day) + self.window_size + 500
        # Binance API 제한: 최대 1000개씩만 가져올 수 있음
        # 따라서 여러 번 요청하는 방식으로 변경
        from datetime import timedelta
        import time
        
        all_data = []
        current_time = start_date
        end_time = end_date
        
        # 5분봉 기준으로 계산
        if timeframe == '5m':
            minutes_per_candle = 5
            max_candles_per_request = 1000
            max_hours_per_request = (max_candles_per_request * minutes_per_candle) / 60
        elif timeframe == '1m':
            minutes_per_candle = 1
            max_candles_per_request = 1000
            max_hours_per_request = (max_candles_per_request * minutes_per_candle) / 60
        elif timeframe == '15m':
            minutes_per_candle = 15
            max_candles_per_request = 1000
            max_hours_per_request = (max_candles_per_request * minutes_per_candle) / 60
        elif timeframe == '1h':
            minutes_per_candle = 60
            max_candles_per_request = 1000
            max_hours_per_request = (max_candles_per_request * minutes_per_candle) / 60
        else:
            minutes_per_candle = 5
            max_candles_per_request = 1000
            max_hours_per_request = 83
        
        print(f"데이터 수집 범위: {start_date} ~ {end_date} ({days_diff}일)")
        request_count = 0
        
        while current_time < end_time:
            request_count += 1
            next_time = min(
                current_time + timedelta(hours=max_hours_per_request),
                end_time
            )
            
            try:
                df_batch = self.fetcher.fetch_ohlcv(
                    since=current_time,
                    limit=max_candles_per_request,
                    timeframe=timeframe
                )
                
                if len(df_batch) == 0:
                    break
                
                # end_date를 넘지 않도록 필터링
                df_batch = df_batch[df_batch.index <= end_time]
                all_data.append(df_batch)
                
                # 마지막 타임스탬프의 다음 시간부터 시작
                last_timestamp = df_batch.index[-1]
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    current_time = last_timestamp + timedelta(minutes=minutes)
                else:
                    current_time = last_timestamp + timedelta(hours=1)
                
                print(f"  요청 {request_count}: {len(df_batch)}개 수집 (총 {sum(len(d) for d in all_data)}개)")
                
                # API rate limit 방지
                time.sleep(0.1)
                
                if current_time >= end_time:
                    break
                    
            except Exception as e:
                print(f"  요청 {request_count} 중 오류: {e}")
                time.sleep(1)
                current_time = next_time
        
        if not all_data:
            raise ValueError("수집된 데이터가 없습니다.")
        
        # 모든 데이터 합치기 및 중복 제거
        df_raw = pd.concat(all_data)
        df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
        df_raw = df_raw.sort_index()
        
        # 날짜 필터링 (인덱스가 datetime)
        df_raw = df_raw[(df_raw.index >= start_date) & (df_raw.index <= end_date)]
        
        if len(df_raw) < self.window_size + 200:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.window_size + 200}개 필요, 현재 {len(df_raw)}개")
        
        print(f"수집된 데이터: {len(df_raw)}개")
        print()
        
        # 피처 엔지니어링 (전체 데이터)
        print("피처 엔지니어링 중...")
        df_features = self.engineer.add_all_features(df_raw)
        print("피처 엔지니어링 완료")
        print()
        
        # 초기 상태 (선물 거래)
        cash = self.initial_capital  # 마진 잔액
        position = None  # 포지션: 'long', 'short', None
        position_size = 0.0  # 포지션 크기 (BTC 수량)
        portfolio_value = cash
        entry_price = 0.0
        entry_amount_usdt = 0.0
        tp_price = None  # Take Profit 가격
        sl_price = None  # Stop Loss 가격
        entry_signal_30m = None  # 포지션 진입 시 모델 신호 (30분봉)
        entry_signal_1h = None  # 포지션 진입 시 모델 신호 (1시간봉)
        
        # 거래 기록 초기화
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        
        # 쿨다운 및 포지션 추적 초기화
        self.trade_cooldown_until = None
        self.last_position_info = None
        
        # 백테스팅 루프
        print("백테스팅 실행 중...")
        min_warmup = 200  # 피처 계산을 위한 충분한 warm-up
        start_idx = self.window_size + min_warmup  # warm-up
        total_predictions = 0
        successful_predictions = 0
        
        for i in range(start_idx, len(df_features)):
            if i % 100 == 0:
                print(f"진행률: {i-start_idx}/{len(df_features)-start_idx} ({100*(i-start_idx)/(len(df_features)-start_idx):.1f}%)")
            
            # 데이터 준비
            X, current_price = self._prepare_data_for_prediction(df_features, i)
            
            if X is None or current_price is None:
                continue
            
            # 예측 수행
            try:
                # 포지션이 있는 경우 포지션 정보 업데이트
                if position is not None:
                    # 포지션 정보 저장 (다음 사이클에서 상태 변경 감지용)
                    if position == 'long':
                        position_roi = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    else:  # short
                        position_roi = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
                    
                    self.last_position_info = {
                        'side': position,
                        'entry_price': entry_price,
                        'entry_amount_usdt': entry_amount_usdt,
                        'current_price': current_price,
                        'roi': position_roi,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'timestamp': df_features.index[i]
                    }
                else:
                    # 포지션이 없는 경우 이전 포지션 정보 확인 (포지션이 닫혔는지 확인)
                    if self.last_position_info is not None:
                        # 포지션이 닫혔음 - 손익 확인 및 쿨다운 설정
                        self._check_position_close_result(self.last_position_info)
                        self.last_position_info = None
                
                # 쿨다운 체크
                current_timestamp = df_features.index[i]
                if self.trade_cooldown_until is not None:
                    if isinstance(current_timestamp, pd.Timestamp):
                        cooldown_timestamp = pd.Timestamp(self.trade_cooldown_until)
                        if current_timestamp < cooldown_timestamp:
                            # 쿨다운 중 - 거래 건너뛰기
                            continue
                        else:
                            # 쿨다운이 만료됨
                            self.trade_cooldown_until = None
                    else:
                        # datetime 객체인 경우
                        if current_timestamp < self.trade_cooldown_until:
                            continue
                        else:
                            # 쿨다운이 만료됨
                            self.trade_cooldown_until = None
                
                # 모델 예측 (멀티타겟: 30분, 1시간)
                y_pred_scaled = self.model.predict(X, verbose=0)  # (1, 2)
                
                # 역변환 (30분, 1시간)
                y_pred_changes = self.preprocessor.target_scaler.inverse_transform(y_pred_scaled)
                
                # 각 타겟의 변화율 추출
                change_30m = np.clip(y_pred_changes[0, 0], -0.5, 0.5)  # 30분봉
                change_1h = np.clip(y_pred_changes[0, 1], -0.5, 0.5)  # 1시간봉
                
                # 절대 가격으로 변환
                predicted_price_30m = current_price * (1 + change_30m)
                predicted_price_1h = current_price * (1 + change_1h)
                
                # 시그널 생성 (30분봉, 1시간봉 사용) - 'long', 'short', 'hold' 반환
                trade_signal = self.check_trade_conditions(change_30m, change_1h)
                
                # 시장 지표 분석 (옵션)
                if self.use_market_indicators:
                    market_signal = self._get_market_signal_from_ohlcv(df_features, i)
                    
                    # 시장 지표 방향 확인
                    market_signal_value = market_signal.get('signal', 'neutral')
                    market_confidence = market_signal.get('confidence', 0.0)
                    market_direction = None
                    if market_signal_value in ['strong_buy', 'buy']:
                        market_direction = 'long'
                    elif market_signal_value in ['strong_sell', 'sell']:
                        market_direction = 'short'
                    else:
                        market_direction = None  # neutral
                    
                    # 종합 조건 확인: 30분봉, 1시간봉, 시장 지표 모두 같은 방향이어야 함
                    final_trade_signal = None
                    
                    if trade_signal:
                        # 30분봉 방향 확인
                        direction_30m = 'long' if change_30m > 0 else ('short' if change_30m < 0 else None)
                        direction_1h = 'long' if change_1h > 0 else ('short' if change_1h < 0 else None)
                        
                        # 세 가지가 모두 같은 방향인지 확인
                        if trade_signal == 'long':
                            if direction_30m == 'long' and direction_1h == 'long' and market_direction == 'long':
                                final_trade_signal = 'long'
                        elif trade_signal == 'short':
                            if direction_30m == 'short' and direction_1h == 'short' and market_direction == 'short':
                                final_trade_signal = 'short'
                    
                    # 최종 거래 신호로 업데이트
                    trade_signal = final_trade_signal
                else:
                    # 시장 지표를 사용하지 않을 때는 중립 신호로 설정
                    market_signal = {'signal': 'neutral', 'confidence': 0.0, 'reasons': []}
                
                # 실제 가격 (30분 후, 1시간 후)
                actual_price_30m = current_price
                actual_price_1h = current_price
                actual_change_30m = 0.0
                actual_change_1h = 0.0
                
                # 30분 후 가격 (5분봉 기준 6개 = 30분)
                if i + 6 < len(df_features):
                    actual_price_30m = float(df_features['close'].iloc[i + 6])
                    actual_change_30m = (actual_price_30m - current_price) / current_price
                
                # 1시간 후 가격 (5분봉 기준 12개 = 1시간)
                if i + 12 < len(df_features):
                    actual_price_1h = float(df_features['close'].iloc[i + 12])
                    actual_change_1h = (actual_price_1h - current_price) / current_price
                    
                    # 예측 정확도 확인 (1시간봉 기준)
                    if (change_1h > 0 and actual_change_1h > 0) or (change_1h < 0 and actual_change_1h < 0):
                        successful_predictions += 1
                    total_predictions += 1
                
                # TP/SL 체크 및 포지션 자동 종료 (포지션이 있는 경우)
                # high/low를 확인하여 5분봉 사이에 TP/SL 도달했는지 정확히 체크
                if position is not None and entry_price > 0:
                    should_close_position = False
                    close_reason = None
                    close_price = current_price
                    
                    # 현재 봉의 high/low 가격 확인
                    current_high = float(df_features['high'].iloc[i]) if 'high' in df_features.columns else current_price
                    current_low = float(df_features['low'].iloc[i]) if 'low' in df_features.columns else current_price
                    
                    if position == 'long':
                        # 롱 포지션: TP/SL 체크
                        # TP: high가 tp_price 이상이면 도달 (가장 높은 가격으로 종료)
                        if tp_price is not None and current_high >= tp_price:
                            should_close_position = True
                            close_reason = 'TP'
                            close_price = max(tp_price, current_price)  # TP 가격 또는 현재 가격 중 높은 값
                        # SL: low가 sl_price 이하이면 도달 (가장 낮은 가격으로 종료)
                        elif sl_price is not None and current_low <= sl_price:
                            should_close_position = True
                            close_reason = 'SL'
                            close_price = min(sl_price, current_price)  # SL 가격 또는 현재 가격 중 낮은 값
                        # 모델 신호 10배 수익 체크 (TP/SL보다 낮은 우선순위)
                        elif entry_signal_30m is not None or entry_signal_1h is not None:
                            # 롱 포지션: 양수 신호의 10배 수익
                            target_roi_30m = abs(entry_signal_30m) * 10 if entry_signal_30m is not None else 0
                            target_roi_1h = abs(entry_signal_1h) * 10 if entry_signal_1h is not None else 0
                            # 더 큰 신호 기준 사용
                            target_roi = max(target_roi_30m, target_roi_1h)
                            
                            if target_roi > 0:
                                # 현재 수익률 계산
                                current_roi = (current_price - entry_price) / entry_price
                                # 레버리지 적용 전 실제 가격 변화율이 목표 수익률에 도달
                                if current_roi >= target_roi:
                                    should_close_position = True
                                    close_reason = 'Signal10x'
                                    close_price = current_price
                    elif position == 'short':
                        # 숏 포지션: TP/SL 체크
                        # TP: low가 tp_price 이하이면 도달 (가장 낮은 가격으로 종료)
                        if tp_price is not None and current_low <= tp_price:
                            should_close_position = True
                            close_reason = 'TP'
                            close_price = min(tp_price, current_price)  # TP 가격 또는 현재 가격 중 낮은 값
                        # SL: high가 sl_price 이상이면 도달 (가장 높은 가격으로 종료)
                        elif sl_price is not None and current_high >= sl_price:
                            should_close_position = True
                            close_reason = 'SL'
                            close_price = max(sl_price, current_price)  # SL 가격 또는 현재 가격 중 높은 값
                        # 모델 신호 10배 수익 체크 (TP/SL보다 낮은 우선순위)
                        elif entry_signal_30m is not None or entry_signal_1h is not None:
                            # 숏 포지션: 음수 신호의 절댓값 10배 수익
                            target_roi_30m = abs(entry_signal_30m) * 10 if entry_signal_30m is not None else 0
                            target_roi_1h = abs(entry_signal_1h) * 10 if entry_signal_1h is not None else 0
                            # 더 큰 신호 기준 사용
                            target_roi = max(target_roi_30m, target_roi_1h)
                            
                            if target_roi > 0:
                                # 현재 수익률 계산 (숏은 반대)
                                current_roi = (entry_price - current_price) / entry_price
                                # 레버리지 적용 전 실제 가격 변화율이 목표 수익률에 도달
                                if current_roi >= target_roi:
                                    should_close_position = True
                                    close_reason = 'Signal10x'
                                    close_price = current_price
                    
                    # TP/SL 도달 시 포지션 종료
                    if should_close_position:
                        if position == 'long':
                            # 롱 포지션 종료 (레버리지 적용)
                            long_roi = (close_price - entry_price) / entry_price
                            # 손익 = ROI * 레버리지 * 마진
                            close_profit_before_fee = entry_amount_usdt * long_roi * self.leverage
                            # 포지션 종료 수수료 (현재 가격 기준 포지션 가치)
                            position_value_at_close = entry_amount_usdt * self.leverage * (1 + long_roi)
                            close_commission = position_value_at_close * self.commission
                            # 최종 손익 (수수료 차감)
                            close_profit = close_profit_before_fee - close_commission
                            # 마진 반환 + 손익
                            cash += entry_amount_usdt + close_profit
                            
                            self.trades.append({
                                'timestamp': df_features.index[i],
                                'type': f'close_long_{close_reason}',
                                'price': close_price,
                                'entry_price': entry_price,
                                'profit': close_profit,
                                'roi': long_roi * 100,
                                'entry_amount': entry_amount_usdt,
                                'close_reason': close_reason
                            })
                            
                            # 손익 확인 및 쿨다운 설정 (실제 손익 기준)
                            if close_profit < 0:
                                current_timestamp = df_features.index[i]
                                if isinstance(current_timestamp, pd.Timestamp):
                                    self.trade_cooldown_until = current_timestamp + pd.Timedelta(minutes=self.cooldown_minutes)
                                else:
                                    self.trade_cooldown_until = current_timestamp + timedelta(minutes=self.cooldown_minutes)
                        
                        elif position == 'short':
                            # 숏 포지션 종료 (레버리지 적용)
                            short_roi = (entry_price - close_price) / entry_price
                            # 손익 = ROI * 레버리지 * 마진
                            close_profit_before_fee = entry_amount_usdt * short_roi * self.leverage
                            # 포지션 종료 수수료 (종료 시점 포지션 가치 = 현재 가격 기준)
                            # 포지션 가치 = 마진 * 레버리지 * (진입가격 / 종료가격)
                            position_value_at_close = entry_amount_usdt * self.leverage * (entry_price / close_price)
                            close_commission = position_value_at_close * self.commission
                            # 최종 손익 (수수료 차감)
                            close_profit = close_profit_before_fee - close_commission
                            # 마진 반환 + 손익
                            cash += entry_amount_usdt + close_profit
                            
                            self.trades.append({
                                'timestamp': df_features.index[i],
                                'type': f'close_short_{close_reason}',
                                'price': close_price,
                                'entry_price': entry_price,
                                'profit': close_profit,
                                'roi': short_roi * 100,
                                'entry_amount': entry_amount_usdt,
                                'close_reason': close_reason
                            })
                            
                            # 손익 확인 및 쿨다운 설정 (실제 손익 기준)
                            if close_profit < 0:
                                current_timestamp = df_features.index[i]
                                if isinstance(current_timestamp, pd.Timestamp):
                                    self.trade_cooldown_until = current_timestamp + pd.Timedelta(minutes=self.cooldown_minutes)
                                else:
                                    self.trade_cooldown_until = current_timestamp + timedelta(minutes=self.cooldown_minutes)
                        
                        # 포지션 초기화
                        position = None
                        entry_price = 0.0
                        entry_amount_usdt = 0.0
                        tp_price = None
                        sl_price = None
                        entry_signal_30m = None
                        entry_signal_1h = None
                        self.last_position_info = None
                        continue  # 포지션이 닫혔으므로 다음 루프로
                
                # 거래 실행 (롱/숏 선물 거래)
                # 포지션이 없을 때만 진입 (진입 중에는 다른 포지션 진입하지 않음)
                if trade_signal == 'long' and position is None:
                    # 롱 포지션 열기 (레버리지 적용)
                    # 선물 거래: 마진을 사용하여 레버리지 포지션 오픈
                    fee_rate = self.commission
                    # 마진 버퍼 (100% 사용)
                    margin_buffer = 1.0
                    
                    # 실제 사용 가능한 마진 계산 (수수료 제외 전)
                    usable_margin = cash * margin_buffer
                    
                    # 포지션 가치 = 마진 * 레버리지
                    position_value = usable_margin * self.leverage
                    
                    # BTC 수량 계산 (포지션 가치를 현재 가격으로 나눔)
                    position_size = position_value / current_price
                    
                    # Dynamic TP/SL 계산
                    # TP 8%, SL 2% (고정)
                    dynamic_roi = 0.4  # 8%
                    dynamic_sl = 0.05  # 2%
                    
                    # TP/SL 가격 계산 (레버리지 고려)
                    # 롱 포지션: TP = entry_price * (1 + dynamic_roi / leverage), SL = entry_price * (1 - dynamic_sl / leverage)
                    tp_price = current_price * (1 + dynamic_roi / self.leverage)
                    sl_price = current_price * (1 - dynamic_sl / self.leverage)
                    
                    # 포지션 오픈 수수료 (포지션 가치 기준)
                    commission_cost = position_value * fee_rate
                    
                    entry_amount_usdt = usable_margin  # 실제 사용한 마진
                    entry_price = current_price
                    position = 'long'
                    # 모델 신호 저장 (10배 수익 체크용)
                    entry_signal_30m = change_30m
                    entry_signal_1h = change_1h
                    # 선물 거래: 마진 잠금 + 오픈 수수료 차감
                    cash = cash - usable_margin - commission_cost
                    
                    self.trades.append({
                        'timestamp': df_features.index[i],
                        'type': 'open_long',
                        'price': current_price,
                        'amount': position_size,
                        'commission': commission_cost,
                        'change_30m': change_30m * 100,
                        'change_1h': change_1h * 100,
                        'strong_signal': (abs(change_30m) >= self.strong_signal_threshold and 
                                         abs(change_1h) >= self.strong_signal_threshold),
                        'market_signal': market_signal.get('signal', 'neutral'),
                        'market_confidence': market_signal.get('confidence', 0.0),
                        'market_reasons': ', '.join(market_signal.get('reasons', []))
                    })
                
                elif trade_signal == 'short' and position is None:
                    # 숏 포지션 열기 (레버리지 적용)
                    # 선물 거래: 마진을 사용하여 레버리지 포지션 오픈
                    fee_rate = self.commission
                    # 마진 버퍼 (100% 사용)
                    margin_buffer = 1.0
                    
                    # 실제 사용 가능한 마진 계산 (수수료 제외 전)
                    usable_margin = cash * margin_buffer
                    
                    # 포지션 가치 = 마진 * 레버리지
                    position_value = usable_margin * self.leverage
                    
                    # BTC 수량 계산 (포지션 가치를 현재 가격으로 나눔)
                    position_size = position_value / current_price
                    
                    # 포지션 오픈 수수료 (포지션 가치 기준)
                    commission_cost = position_value * fee_rate
                    
                    # Dynamic TP/SL 계산
                    # TP 8%, SL 2% (고정)
                    dynamic_roi = 0.4  # 8%
                    dynamic_sl = 0.05  # 2%
                    
                    # TP/SL 가격 계산 (레버리지 고려)
                    # 숏 포지션: TP = entry_price * (1 - dynamic_roi / leverage), SL = entry_price * (1 + dynamic_sl / leverage)
                    tp_price = current_price * (1 - dynamic_roi / self.leverage)
                    sl_price = current_price * (1 + dynamic_sl / self.leverage)
                    
                    entry_amount_usdt = usable_margin  # 실제 사용한 마진
                    entry_price = current_price
                    position = 'short'
                    # 모델 신호 저장 (10배 수익 체크용)
                    entry_signal_30m = change_30m
                    entry_signal_1h = change_1h
                    # 선물 거래: 마진 잠금 + 오픈 수수료 차감
                    cash = cash - usable_margin - commission_cost
                    
                    self.trades.append({
                        'timestamp': df_features.index[i],
                        'type': 'open_short',
                        'price': current_price,
                        'amount': position_size,
                        'commission': commission_cost,
                        'change_30m': change_30m * 100,
                        'change_1h': change_1h * 100,
                        'strong_signal': (abs(change_30m) >= self.strong_signal_threshold and 
                                         abs(change_1h) >= self.strong_signal_threshold),
                        'market_signal': market_signal.get('signal', 'neutral'),
                        'market_confidence': market_signal.get('confidence', 0.0),
                        'market_reasons': ', '.join(market_signal.get('reasons', []))
                    })
                
                # 포트폴리오 가치 계산 (롱/숏 선물 거래, 레버리지 적용)
                # 선물 거래: 포트폴리오 가치 = 현금 + 잠긴 마진 + 미실현 손익
                # (수수료는 포지션 종료 시에만 발생하므로 미실현 손익에는 포함하지 않음)
                if position == 'long':
                    # 롱 포지션: 미실현 손익 = (현재 가격 - 진입 가격) / 진입 가격 * 마진 * 레버리지
                    unrealized_roi = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    unrealized_pnl = entry_amount_usdt * unrealized_roi * self.leverage
                    portfolio_value = cash + entry_amount_usdt + unrealized_pnl
                elif position == 'short':
                    # 숏 포지션: 미실현 손익 = (진입 가격 - 현재 가격) / 진입 가격 * 마진 * 레버리지
                    unrealized_roi = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
                    unrealized_pnl = entry_amount_usdt * unrealized_roi * self.leverage
                    portfolio_value = cash + entry_amount_usdt + unrealized_pnl
                else:
                    portfolio_value = cash
                
                self.portfolio_values.append({
                    'timestamp': df_features.index[i],
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'position': position,
                    'price': current_price
                })
                
                self.signals.append({
                    'timestamp': df_features.index[i],
                    'signal': trade_signal if trade_signal else 'hold',
                    'position': position,
                    'current_price': current_price,
                    'predicted_price_30m': predicted_price_30m,
                    'predicted_price_1h': predicted_price_1h,
                    'actual_price_30m': actual_price_30m,
                    'actual_price_1h': actual_price_1h,
                    'predicted_change_30m': change_30m * 100,
                    'predicted_change_1h': change_1h * 100,
                    'actual_change_30m': actual_change_30m * 100,
                    'actual_change_1h': actual_change_1h * 100,
                    'strong_signal': (abs(change_30m) >= self.strong_signal_threshold and 
                                     abs(change_1h) >= self.strong_signal_threshold),
                    'market_signal': market_signal.get('signal', 'neutral'),
                    'market_confidence': market_signal.get('confidence', 0.0),
                    'market_reasons': ', '.join(market_signal.get('reasons', []))
                })
                
            except Exception as e:
                print(f"⚠️ 경고: 인덱스 {i}에서 예측 실패: {e}")
                continue
        
        # 최종 포트폴리오 가치 (롱/숏 선물 거래, 레버리지 적용)
        final_price = float(df_features['close'].iloc[-1])
        if position == 'long':
            # 롱 포지션 종료
            final_roi = (final_price - entry_price) / entry_price if entry_price > 0 else 0.0
            # 손익 = ROI * 레버리지 * 마진
            final_profit_before_fee = entry_amount_usdt * final_roi * self.leverage
            # 포지션 종료 수수료 (종료 시점 포지션 가치 = 현재 가격 기준)
            position_value_at_close = entry_amount_usdt * self.leverage * (final_price / entry_price)
            final_commission = position_value_at_close * self.commission
            # 최종 손익 (수수료 차감)
            final_profit = final_profit_before_fee - final_commission
            final_portfolio_value = cash + entry_amount_usdt + final_profit
        elif position == 'short':
            # 숏 포지션 종료
            final_roi = (entry_price - final_price) / entry_price if entry_price > 0 else 0.0
            # 손익 = ROI * 레버리지 * 마진
            final_profit_before_fee = entry_amount_usdt * final_roi * self.leverage
            # 포지션 종료 수수료 (종료 시점 포지션 가치 = 현재 가격 기준)
            position_value_at_close = entry_amount_usdt * self.leverage * (entry_price / final_price)
            final_commission = position_value_at_close * self.commission
            # 최종 손익 (수수료 차감)
            final_profit = final_profit_before_fee - final_commission
            final_portfolio_value = cash + entry_amount_usdt + final_profit
        else:
            final_portfolio_value = cash
        
        # 성능 지표 계산
        results = self._calculate_metrics(
            self.initial_capital,
            final_portfolio_value,
            self.portfolio_values,
            self.trades,
            successful_predictions,
            total_predictions
        )
        
        print("\n" + "=" * 60)
        print("백테스팅 완료")
        print("=" * 60)
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self,
                          initial_capital: float,
                          final_value: float,
                          portfolio_values: List[Dict],
                          trades: List[Dict],
                          successful_predictions: int,
                          total_predictions: int) -> Dict:
        """성능 지표 계산"""
        # 총 수익률
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # 연환산 수익률
        if len(portfolio_values) > 0:
            days = (portfolio_values[-1]['timestamp'] - portfolio_values[0]['timestamp']).days
            if days > 0:
                annual_return = ((final_value / initial_capital) ** (365 / days) - 1) * 100
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0
        
        # 최대 낙폭 (MDD)
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
        max_drawdown = portfolio_df['drawdown'].min()
        
        # 샤프 비율 (간단 버전)
        if len(portfolio_df) > 1:
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 288)  # 5분봉 기준 연환산
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 거래 통계 (롱/숏 선물 거래)
        open_long_trades = [t for t in trades if t['type'] == 'open_long']
        open_short_trades = [t for t in trades if t['type'] == 'open_short']
        close_long_trades = [t for t in trades if t['type'] == 'close_long']
        close_short_trades = [t for t in trades if t['type'] == 'close_short']
        
        # 종료된 거래 (수익/손실 계산 가능)
        closed_trades = close_long_trades + close_short_trades
        
        # 승률
        profitable_trades = [t for t in closed_trades if 'profit' in t and t['profit'] > 0]
        win_rate = len(profitable_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0.0
        
        # 평균 수익/손실
        if len(closed_trades) > 0:
            profits = [t.get('profit', 0) for t in closed_trades]
            avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0.0
            avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0.0
        else:
            avg_profit = 0.0
            avg_loss = 0.0
        
        # 방향 정확도
        direction_accuracy = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'open_long_trades': len(open_long_trades),
            'open_short_trades': len(open_short_trades),
            'close_long_trades': len(close_long_trades),
            'close_short_trades': len(close_short_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'direction_accuracy': direction_accuracy,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'signals': self.signals
        }
    
    def _print_results(self, results: Dict):
        """결과 출력"""
        print(f"\n초기 자본금: ${results['initial_capital']:,.2f}")
        print(f"최종 자산: ${results['final_value']:,.2f}")
        print(f"총 수익률: {results['total_return']:+.2f}%")
        print(f"연환산 수익률: {results['annual_return']:+.2f}%")
        print(f"최대 낙폭 (MDD): {results['max_drawdown']:.2f}%")
        print(f"샤프 비율: {results['sharpe_ratio']:.2f}")
        print(f"\n거래 통계:")
        print(f"  총 거래 횟수: {results['total_trades']}회")
        print(f"  롱 포지션 열기: {results['open_long_trades']}회")
        print(f"  숏 포지션 열기: {results['open_short_trades']}회")
        print(f"  롱 포지션 종료: {results['close_long_trades']}회")
        print(f"  숏 포지션 종료: {results['close_short_trades']}회")
        print(f"  승률: {results['win_rate']:.2f}%")
        print(f"  평균 수익: ${results['avg_profit']:,.2f}")
        print(f"  평균 손실: ${results['avg_loss']:,.2f}")
        print(f"\n예측 정확도:")
        print(f"  방향 정확도: {results['direction_accuracy']:.2f}%")
    
    def plot_results(self, results: Dict, save_path: str = 'results/backtest_results.png'):
        """결과 시각화"""
        portfolio_df = pd.DataFrame(results['portfolio_values'])
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df = portfolio_df.set_index('timestamp')
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 포트폴리오 가치
        axes[0].plot(portfolio_df.index, portfolio_df['portfolio_value'], label='Portfolio Value', linewidth=2)
        axes[0].axhline(y=results['initial_capital'], color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 가격 및 시그널
        axes[1].plot(portfolio_df.index, portfolio_df['price'], label='BTC Price', linewidth=1.5, alpha=0.7)
        
        # 롱/숏 포지션 표시
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            long_opens = trades_df[trades_df['type'] == 'open_long']
            short_opens = trades_df[trades_df['type'] == 'open_short']
            long_closes = trades_df[trades_df['type'] == 'close_long']
            short_closes = trades_df[trades_df['type'] == 'close_short']
            
            if len(long_opens) > 0:
                axes[1].scatter(long_opens['timestamp'], 
                              long_opens['price'].values,
                              color='green', marker='^', s=100, label='Long Open', zorder=5)
            if len(short_opens) > 0:
                axes[1].scatter(short_opens['timestamp'],
                              short_opens['price'].values,
                              color='red', marker='v', s=100, label='Short Open', zorder=5)
            if len(long_closes) > 0:
                axes[1].scatter(long_closes['timestamp'],
                              long_closes['price'].values,
                              color='blue', marker='x', s=50, label='Long Close', zorder=5)
            if len(short_closes) > 0:
                axes[1].scatter(short_closes['timestamp'],
                              short_closes['price'].values,
                              color='orange', marker='x', s=50, label='Short Close', zorder=5)
        
        axes[1].set_title('BTC Price and Trading Signals', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
        axes[2].fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, color='red', alpha=0.3)
        axes[2].plot(portfolio_df.index, portfolio_df['drawdown'], color='red', linewidth=1.5)
        axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Drawdown (%)', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n그래프 저장: {save_path}")
        plt.close()


def main():
    """백테스팅 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='백테스팅 실행')
    parser.add_argument('--model', type=str, default='models/best_model.h5', help='모델 파일 경로')
    parser.add_argument('--start', type=str, default=None, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30, help='백테스팅 기간 (일)')
    parser.add_argument('--capital', type=float, default=300, help='초기 자본금')
    parser.add_argument('--leverage', type=int, default=10, help='레버리지 배수 (기본값: 10배)')
    parser.add_argument('--min-change-30m', type=float, default=0.004, help='30분봉 최소 변화율 (기본값: 0.004 = 0.4%%)')
    parser.add_argument('--min-change-1h', type=float, default=0.003, help='1시간봉 최소 변화율 (기본값: 0.001 = 0.1%%)')
    parser.add_argument('--strong-signal', type=float, default=0.009, help='강한 신호 기준 (기본값: 0.009 = 0.9%%)')
    parser.add_argument('--cooldown', type=int, default=90, help='손실 후 쿨다운 시간 (분, 기본값: 15)')
    parser.add_argument('--use-market-indicators', dest='use_market_indicators', action='store_true', default=True, help='시장 지표 사용 (기본값: True)')
    parser.add_argument('--no-market-indicators', dest='use_market_indicators', action='store_false', help='시장 지표 사용 안 함')
    
    args = parser.parse_args()
    
    # 날짜 설정
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # 백테스팅 실행
    backtester = Backtester(
        model_path=args.model,
        min_change_30m=args.min_change_30m,
        min_change_1h=args.min_change_1h,
        strong_signal_threshold=args.strong_signal,
        initial_capital=args.capital,
        leverage=args.leverage,
        cooldown_minutes=args.cooldown,
        use_market_indicators=args.use_market_indicators
    )
    
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date
    )
    
    # 결과 시각화
    backtester.plot_results(results)
    
    # 결과 저장
    results_df = pd.DataFrame(results['portfolio_values'])
    results_df.to_csv('results/backtest_portfolio.csv', index=False)
    
    signals_df = pd.DataFrame(results['signals'])
    signals_df.to_csv('results/backtest_signals.csv', index=False)
    
    print("\n결과 저장 완료:")
    print("  - results/backtest_results.png")
    print("  - results/backtest_portfolio.csv")
    print("  - results/backtest_signals.csv")


if __name__ == '__main__':
    main()


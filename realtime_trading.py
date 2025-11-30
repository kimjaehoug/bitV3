"""
실시간 비트코인 가격 예측 및 매수/매도 시그널 생성 시스템
5분마다 자동으로 데이터를 수집하고 예측하여 시그널을 제공
"""
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

import ccxt
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from data_preprocessor import DataPreprocessor
from predictor import Predictor
from model import PatchCNNBiLSTM
from market_indicators import MarketIndicators


class RealtimeTradingSignal:
    """실시간 거래 시그널 생성 클래스"""
    
    def __init__(self, 
                 model_path: str = 'models/best_model.h5',
                 window_size: int = 60,
                 min_confidence: float = 0.02):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            window_size: 슬라이딩 윈도우 크기
            min_confidence: 최소 신뢰도 (가격 변화율, 2% = 0.02)
        """
        self.model_path = model_path
        self.window_size = window_size
        self.min_confidence = min_confidence
        
        # 컴포넌트 초기화
        self.fetcher = BinanceDataFetcher()
        self.engineer = FeatureEngineer()
        
        # 모델 및 전처리기 로드
        self.model = None
        self.preprocessor = None
        self.predictor = None
        self.feature_names = None
        
        # 이전 예측값 저장 (방향성 판단용)
        self.last_prediction = None
        self.last_price = None
        
        print("=" * 60)
        print("실시간 거래 시그널 시스템 초기화 중...")
        print("=" * 60)
        self._load_model()
        print("초기화 완료!\n")
    
    def _load_model(self):
        """학습된 모델 및 전처리기 로드"""
        # 모델 파일 찾기
        if not os.path.exists(self.model_path):
            # models 디렉토리에서 최신 모델 찾기
            models_dir = 'models'
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if model_files:
                    # 파일명에 날짜가 포함되어 있으면 최신 것 선택
                    model_files.sort(reverse=True)
                    self.model_path = os.path.join(models_dir, model_files[0])
                    print(f"모델 파일 자동 선택: {self.model_path}")
                else:
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            else:
                raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
        
        # 전처리기 초기화 (학습 시와 동일한 설정)
        self.preprocessor = DataPreprocessor(
            window_size=self.window_size,
            prediction_horizon=1,
            target_column='close',
            scaler_type='robust'
        )
        
        # 스케일러 먼저 로드 (feature 개수 확인용)
        scaler_path = 'models/scalers.pkl'
        if os.path.exists(scaler_path):
            print(f"스케일러 로드 중: {scaler_path}")
            self.preprocessor.load_scalers(scaler_path)
            self.feature_names = self.preprocessor.feature_columns
            if not self.feature_names:
                raise ValueError("스케일러에 feature_columns 정보가 없습니다. main.py를 다시 실행하여 스케일러를 저장하세요.")
            
            # 스케일러가 기대하는 feature 개수 확인
            scaler_expected_features = self.preprocessor.scaler.n_features_in_
            print(f"스케일러가 기대하는 feature 개수: {scaler_expected_features}개")
            print(f"스케일러에 저장된 feature_names 개수: {len(self.feature_names)}개")
            
            # 중복이 있어도 원본 그대로 사용 (스케일러는 64개로 학습되었으므로)
            # 중복 체크만 하고 제거하지 않음
            seen = set()
            unique_count = 0
            for col in self.feature_names:
                if col not in seen:
                    seen.add(col)
                    unique_count += 1
            
            if unique_count != len(self.feature_names):
                print(f"⚠️ 경고: feature_names에 중복이 있습니다. 고유 feature: {unique_count}개, 총 feature: {len(self.feature_names)}개")
                print(f"스케일러가 {scaler_expected_features}개를 기대하므로 원본 그대로 사용합니다.")
            
            # 원본 feature_names 그대로 사용 (중복 포함)
            num_features = scaler_expected_features
            print(f"사용할 feature 개수: {num_features}개 (원본 feature_names 그대로 사용)")
            # num_features를 인스턴스 변수로 저장 (나중에 사용)
            self.num_features = num_features
        else:
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}. main.py를 먼저 실행하여 모델과 스케일러를 학습하세요.")
        
        # 모델 로드 (Lambda 레이어 문제 해결을 위해 가중치만 로드)
        print(f"모델 로드 중: {self.model_path}")
        
        # 저장된 모델의 실제 feature 개수 확인 (가중치에서 추출)
        saved_num_features = None
        try:
            # 가중치 파일에서 첫 번째 레이어의 입력 feature 개수 확인
            import h5py
            with h5py.File(self.model_path, 'r') as f:
                # 모델 구조 탐색
                if 'model_weights' in f:
                    model_weights = f['model_weights']
                    # CNN 레이어 찾기
                    if 'cnn' in model_weights:
                        cnn_weights = model_weights['cnn']
                        if 'cnn' in cnn_weights and 'kernel:0' in cnn_weights['cnn']:
                            kernel_shape = cnn_weights['cnn']['kernel:0'].shape
                            # kernel shape: (kernel_size, input_features, output_features)
                            saved_num_features = kernel_shape[1]
                            print(f"저장된 모델의 feature 개수: {saved_num_features}개 (가중치에서 확인)")
        except Exception as e:
            print(f"⚠️ 경고: 가중치 파일에서 feature 개수를 확인할 수 없습니다: {e}")
        
        # 저장된 모델의 feature 개수가 확인되면 그것을 사용, 아니면 스케일러 기대값 사용
        if saved_num_features is not None and saved_num_features != num_features:
            print(f"⚠️ 경고: 스케일러는 {num_features}개를 기대하지만, 저장된 모델은 {saved_num_features}개로 학습되었습니다.")
            print(f"저장된 모델의 feature 개수({saved_num_features}개)에 맞춰 모델을 재구성합니다.")
            num_features = saved_num_features
            # num_features를 인스턴스 변수로 저장 (나중에 _prepare_realtime_data에서 사용)
            self.num_features = num_features
        
        # 모델 구조 재구성
        print("모델 구조 재구성 중...")
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
        try:
            print("가중치 로드 중...")
            self.model.load_weights(self.model_path)
            print("가중치 로드 완료!")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ 경고: 가중치 로드 실패: {error_msg}")
            
            # 오류 메시지에서 실제 feature 개수 추출 시도
            import re
            match = re.search(r'Received saved weight with shape \([^,]+,\s*(\d+),\s*\d+\)', error_msg)
            if match:
                saved_features = int(match.group(1))
                print(f"오류 메시지에서 추출한 저장된 feature 개수: {saved_features}개")
                if saved_features != num_features:
                    print(f"모델을 {saved_features}개 feature로 재구성합니다...")
                    num_features = saved_features
                    self.num_features = num_features  # 인스턴스 변수 업데이트
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
                    try:
                        self.model.load_weights(self.model_path)
                        print("가중치 로드 완료! (재구성 후)")
                        # 재구성 후 num_features 확실히 업데이트
                        self.num_features = saved_features
                        print(f"✅ num_features 업데이트: {self.num_features}개")
                    except Exception as e2:
                        print(f"⚠️ 경고: 재구성 후에도 가중치 로드 실패: {e2}")
                        print("새로운 모델로 시작합니다 (가중치 없음)")
            else:
                print("새로운 모델로 시작합니다 (가중치 없음)")
        
        # 최종 num_features 확인 (모델의 실제 입력 shape에서 직접 확인 - 가장 확실한 방법)
        if self.model is not None:
            try:
                model_input_shape = self.model.input_shape
                if model_input_shape and len(model_input_shape) >= 3:
                    actual_model_features = int(model_input_shape[2])
                    self.num_features = actual_model_features
                    print(f"✅ 최종 모델 feature 개수 확인: {self.num_features}개 (모델 입력 shape: {model_input_shape})")
            except Exception as e:
                print(f"⚠️ 경고: 모델 입력 shape 확인 실패: {e}")
        
        # 최종 num_features를 인스턴스 변수로 저장 (확실하게)
        self.num_features = num_features
        print(f"최종 모델 feature 개수: {self.num_features}개")
        
        # 모델 컴파일 (손실 함수 재설정)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        # 예측기 초기화
        self.predictor = Predictor(
            model=self.model,
            preprocessor=self.preprocessor,
            target_scaler=self.preprocessor.target_scaler
        )
        
        print("모델 로드 완료!")
    
    def _prepare_realtime_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        실시간 데이터를 모델 입력 형태로 변환
        
        Args:
            df: 최근 데이터 (최소 window_size + warmup 개 필요)
            
        Returns:
            모델 입력 시퀀스 (1, window_size, n_features), 현재 가격
        """
        # 특징 엔지니어링
        df_features = self.engineer.add_all_features(df)
        
        # Warm-up 제거 (main.py와 동일하게 100개)
        # 하지만 rolling window가 더 긴 feature들(예: ma100)을 고려하여 더 많은 warm-up 필요
        min_warmup = 150  # ma100을 고려하여 150개로 증가
        if len(df_features) <= min_warmup:
            raise ValueError(f"데이터가 부족합니다. 최소 {min_warmup + self.window_size}개 필요 (warm-up {min_warmup}개 + window_size {self.window_size}개), 현재 {len(df_features)}개")
        
        df_features = df_features.iloc[min_warmup:].copy()
        
        # 결측치 처리 (warm-up 제거 후)
        df_features = df_features.ffill().bfill().fillna(0)
        
        # 결측치 처리
        df_features = df_features.ffill().bfill().fillna(0)
        
        # 미래 데이터 누수 방지
        df_clean = self.preprocessor._remove_future_leakage(df_features)
        
        # 모델이 실제로 사용한 feature 목록 확인
        model_feature_names = getattr(self.preprocessor, 'model_feature_columns', None)
        if model_feature_names is None:
            # 모델 feature 목록이 없으면 스케일러 feature 목록 사용
            print("⚠️ 경고: 모델 feature 목록이 없습니다. 스케일러 feature 목록을 사용합니다.")
            model_feature_names = self.feature_names
        
        # 스케일러에 저장된 feature_columns 사용 (정확히 일치해야 함)
        if self.feature_names is None:
            raise ValueError("feature_names가 설정되지 않았습니다. 스케일러를 먼저 로드하세요.")
        
        print(f"디버깅: df_clean 컬럼 수: {len(df_clean.columns)}")
        print(f"디버깅: 스케일러 feature 수: {len(self.feature_names)}개 (중복 포함)")
        print(f"디버깅: 모델 feature 수: {len(model_feature_names)}개")
        
        # 모든 feature가 생성되었는지 확인 (스케일러 feature 기준)
        missing_features = [col for col in self.feature_names if col not in df_clean.columns]
        if missing_features:
            raise ValueError(f"Feature가 생성되지 않았습니다. 데이터가 충분하지 않을 수 있습니다. "
                           f"누락된 feature: {missing_features[:10]}... "
                           f"(총 {len(missing_features)}개). 더 많은 데이터를 수집하거나 warm-up 기간을 늘려주세요.")
        
        # 마지막 window_size 개의 데이터로 시퀀스 생성
        if len(df_clean) < self.window_size:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.window_size}개 필요, 현재 {len(df_clean)}개")
        
        # 먼저 고유한 feature만 선택 (DataFrame에 추가)
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
        
        # 마지막 window_size 개 선택
        recent_data_for_scaler = recent_data_for_scaler[-self.window_size:]
        recent_data_for_model = recent_data_for_model[-self.window_size:]
        
        # 모델이 기대하는 feature 개수 확인 (모델의 실제 입력 shape에서 직접 확인)
        if hasattr(self, 'model') and self.model is not None:
            # 모델의 입력 shape에서 feature 개수 확인 (가장 확실한 방법)
            try:
                model_input_shape = self.model.input_shape
                if model_input_shape and len(model_input_shape) >= 3:
                    model_expected_features = int(model_input_shape[2])  # (batch, timesteps, features)
                    print(f"디버깅: 모델 입력 shape에서 확인한 feature 개수: {model_expected_features}개")
                else:
                    # fallback: num_features 속성 사용
                    model_expected_features = getattr(self, 'num_features', None)
                    if model_expected_features is None:
                        model_expected_features = self.preprocessor.scaler.n_features_in_
                    print(f"디버깅: num_features 속성에서 확인한 feature 개수: {model_expected_features}개")
            except Exception as e:
                print(f"⚠️ 경고: 모델 입력 shape 확인 실패: {e}")
                model_expected_features = getattr(self, 'num_features', None)
                if model_expected_features is None:
                    model_expected_features = self.preprocessor.scaler.n_features_in_
        else:
            # 모델이 없으면 num_features 속성 사용
            model_expected_features = getattr(self, 'num_features', None)
            if model_expected_features is None:
                model_expected_features = self.preprocessor.scaler.n_features_in_
        
        # 스케일러가 기대하는 feature 개수 확인
        scaler_expected_features = self.preprocessor.scaler.n_features_in_
        model_expected_features = len(model_feature_names)
        
        print(f"디버깅: 스케일러 기대 feature: {scaler_expected_features}개, 모델 기대 feature: {model_expected_features}개")
        print(f"디버깅: 스케일러 데이터 shape: {recent_data_for_scaler.shape}, 모델 데이터 shape: {recent_data_for_model.shape}")
        
        # 스케일링 (스케일러는 이미 로드되어 있어야 함)
        if not self.preprocessor.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다. models/scalers.pkl 파일을 확인하세요.")
        
        # 스케일러는 전체 feature로 스케일링 (스케일러가 학습한 순서대로)
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
                    else:
                        print(f"⚠️ 경고: 모델 feature '{col}'가 스케일러 feature 목록에 없습니다.")
                else:
                    print(f"⚠️ 경고: 모델 feature '{col}'가 스케일러 feature 목록에 없습니다.")
            
            if len(model_feature_indices) == model_expected_features:
                # 스케일링된 데이터에서 모델 feature만 선택
                recent_scaled = recent_scaled_scaler[:, :, model_feature_indices]
                print(f"디버깅: 스케일링 후 모델 feature 선택 완료: {recent_scaled.shape}")
            else:
                print(f"⚠️ 경고: 모델 feature 인덱스 매칭 실패 ({len(model_feature_indices)}/{model_expected_features}). 앞 {model_expected_features}개만 사용합니다.")
                recent_scaled = recent_scaled_scaler[:, :, :model_expected_features]
        else:
            # feature 개수가 같으면 그대로 사용
            recent_scaled = recent_scaled_scaler
        
        print(f"디버깅: 최종 입력 shape: {recent_scaled.shape} (모델 기대: (1, {self.window_size}, {model_expected_features}))")
        
        # 현재 가격 (마지막 시점의 close 가격)
        current_price = float(df_clean['close'].iloc[-1])
        
        return recent_scaled, current_price
    
    def _generate_signal(self, 
                        current_price: float, 
                        predicted_price: float,
                        confidence: float) -> dict:
        """
        매수/매도 시그널 생성
        
        Args:
            current_price: 현재 가격
            predicted_price: 예측 가격 (5분 후)
            confidence: 신뢰도 (가격 변화율)
            
        Returns:
            시그널 딕셔너리
        """
        # 가격 변화율 계산
        price_change_pct = (predicted_price - current_price) / current_price
        
        # 방향성 판단
        if price_change_pct > self.min_confidence:
            signal = "매수"
            strength = min(abs(price_change_pct) / self.min_confidence, 3.0)  # 최대 3배
        elif price_change_pct < -self.min_confidence:
            signal = "매도"
            strength = min(abs(price_change_pct) / self.min_confidence, 3.0)
        else:
            signal = "보유"
            strength = 0.0
        
        # 이전 예측과 비교하여 방향성 일관성 확인
        direction_consistency = "일관"
        if self.last_prediction is not None:
            last_change = (self.last_prediction - self.last_price) / self.last_price if self.last_price > 0 else 0
            current_change = price_change_pct
            
            if (last_change > 0 and current_change < 0) or (last_change < 0 and current_change > 0):
                direction_consistency = "변화"
        
        return {
            'signal': signal,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pct': price_change_pct * 100,  # 퍼센트로 변환
            'strength': strength,
            'confidence': abs(price_change_pct) * 100,
            'direction_consistency': direction_consistency,
            'timestamp': datetime.now()
        }
    
    def predict_and_signal(self) -> dict:
        """
        실시간 예측 및 시그널 생성
        
        Returns:
            예측 결과 및 시그널 딕셔너리
        """
        try:
            # 최근 데이터 수집 (window_size + warmup + 여유분)
            # warm-up 150개 + window_size 60개 + 여유분 = 최소 250개 이상 필요
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터 수집 중...")
            min_required = 250  # warm-up 150 + window_size 60 + 여유분
            hours_back = max(24, (min_required * 5) / 60)  # 충분한 데이터 확보 (5분봉 기준)
            df_raw = self.fetcher.fetch_recent_data(hours=int(hours_back), timeframe='5m')
            
            if len(df_raw) < min_required:
                raise ValueError(f"데이터가 부족합니다. 최소 {min_required}개 필요, 현재 {len(df_raw)}개")
            
            # 데이터 준비
            X, current_price = self._prepare_realtime_data(df_raw)
            
            # 예측 수행
            print("예측 수행 중...")
            
            # 입력 데이터 확인
            X_last_timestep = X[0, -1, :]  # 마지막 시점의 feature 값들
            X_first_timestep = X[0, 0, :]  # 첫 번째 시점의 feature 값들
            
            print(f"디버깅: 입력 데이터 마지막 시점 통계 - min: {X_last_timestep.min():.4f}, max: {X_last_timestep.max():.4f}, mean: {X_last_timestep.mean():.4f}")
            print(f"디버깅: 입력 데이터 첫 시점 통계 - min: {X_first_timestep.min():.4f}, max: {X_first_timestep.max():.4f}, mean: {X_first_timestep.mean():.4f}")
            print(f"디버깅: 입력 데이터 전체 범위 - min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}")
            
            # 입력 데이터의 첫 번째와 마지막 시점 비교 (시퀀스 내 변화 확인)
            input_diff = np.abs(X_first_timestep - X_last_timestep).mean()
            print(f"디버깅: 입력 데이터 변화 (첫 시점 vs 마지막 시점 평균 차이): {input_diff:.6f}")
            
            # 이전 입력과 비교 (실제 시간 경과에 따른 변화)
            X_diff = 0.0  # 초기화
            if hasattr(self, 'last_X_input'):
                X_diff = np.abs(X[0] - self.last_X_input).mean()
                X_diff_last = np.abs(X[0, -1, :] - self.last_X_input[-1, :]).mean()
                print(f"디버깅: 이전 입력과의 차이 (전체 시퀀스 평균): {X_diff:.6f}")
                print(f"디버깅: 이전 입력과의 차이 (마지막 시점만): {X_diff_last:.6f}")
                if X_diff < 0.001:
                    print("⚠️ 경고: 입력 데이터가 거의 변하지 않습니다!")
                elif X_diff > 0.1:
                    print(f"⚠️ 경고: 입력 데이터가 크게 변했습니다 ({X_diff:.6f})")
            else:
                print("디버깅: 첫 실행이므로 이전 입력과 비교할 수 없습니다.")
            self.last_X_input = X[0].copy()
            
            # 모델 예측 (멀티타겟: 3분, 5분, 15분)
            y_pred_scaled = self.model.predict(X, verbose=0)  # (1, 3)
            
            # 모델 원시 출력 확인 (스케일링 전)
            print(f"디버깅: 모델 원시 출력 (스케일링 전) - 3분: {y_pred_scaled[0, 0]:.6f}, 5분: {y_pred_scaled[0, 1]:.6f}, 15분: {y_pred_scaled[0, 2]:.6f}")
            
            # 이전 예측과 비교
            if hasattr(self, 'last_model_output_scaled'):
                output_diff = np.abs(y_pred_scaled[0] - self.last_model_output_scaled).mean()
                print(f"디버깅: 이전 예측과의 차이 (스케일링 전): {output_diff:.6f}")
                if output_diff < 0.0001:
                    print("⚠️ 경고: 모델 출력이 거의 변하지 않습니다! 모델이 입력에 반응하지 않습니다.")
                
                # 입력 변화와 출력 변화 비교
                if hasattr(self, '_last_X_diff') and self._last_X_diff > 0.01 and output_diff < 0.0001:
                    print(f"⚠️ 심각: 입력이 크게 변했지만 ({self._last_X_diff:.6f}), 모델 출력은 거의 변하지 않습니다 ({output_diff:.6f})!")
                    print("   → 모델이 입력 변화에 반응하지 않습니다. 모델 재학습이 필요할 수 있습니다.")
            else:
                print("디버깅: 첫 실행이므로 이전 예측과 비교할 수 없습니다.")
            self.last_model_output_scaled = y_pred_scaled[0].copy()
            self._last_X_diff = X_diff  # 다음 실행을 위해 저장
            
            y_pred_changes = self.preprocessor.target_scaler.inverse_transform(y_pred_scaled)  # (1, 3)
            
            # 각 타겟의 변화율 추출
            change_3m = y_pred_changes[0, 0]
            change_5m = y_pred_changes[0, 1]
            change_15m = y_pred_changes[0, 2]
            
            print(f"디버깅: 멀티타겟 예측 변화율 - 3분: {change_3m:.6f}, 5분: {change_5m:.6f}, 15분: {change_15m:.6f}")
            
            # 변화율 클리핑 (극단값 방지)
            change_3m = np.clip(change_3m, -0.5, 0.5)
            change_5m = np.clip(change_5m, -0.5, 0.5)
            change_15m = np.clip(change_15m, -0.5, 0.5)
            
            # 5분 타겟을 메인으로 사용 (기존 호환성 유지)
            y_pred_change = change_5m
            
            # 절대 가격으로 변환 (3분, 5분, 15분)
            predicted_price_3m = current_price * (1 + change_3m)
            predicted_price_5m = current_price * (1 + change_5m)
            predicted_price_15m = current_price * (1 + change_15m)
            
            # 중기 추세 분석 (15분 변화율이 5분과 같은 방향이면 추세 일관성 높음)
            trend_consistency = "일관" if (change_5m * change_15m > 0) else "불일치"
            
            # 시그널 생성 (5분 타겟 기준, 하지만 중기 추세도 고려)
            # 15분 추세가 강하면 신뢰도 증가
            confidence = abs(change_5m)
            if abs(change_15m) > 0.001:  # 15분 추세가 있으면
                confidence = (abs(change_5m) + abs(change_15m) * 0.5) / 1.5  # 중기 추세 반영
            
            signal_info = self._generate_signal(current_price, predicted_price_5m, confidence)
            
            # 멀티타겟 정보 추가
            signal_info['change_3m'] = change_3m
            signal_info['change_5m'] = change_5m
            signal_info['change_15m'] = change_15m
            signal_info['predicted_price_3m'] = predicted_price_3m
            signal_info['predicted_price_15m'] = predicted_price_15m
            signal_info['trend_consistency'] = trend_consistency
            
            # 이전 예측값 저장 (5분 타겟 사용)
            self.last_prediction = predicted_price_5m
            self.last_price = current_price
            
            return {
                'success': True,
                **signal_info
            }
            
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def run_continuous(self, interval_minutes: int = 5):
        """
        연속적으로 예측 및 시그널 생성 (5분마다)
        
        Args:
            interval_minutes: 실행 간격 (분)
        """
        print("=" * 60)
        print("실시간 거래 시그널 시스템 시작")
        print(f"실행 간격: {interval_minutes}분")
        print("종료하려면 Ctrl+C를 누르세요")
        print("=" * 60)
        
        try:
            while True:
                result = self.predict_and_signal()
                
                if result['success']:
                    # 결과 출력
                    print("\n" + "=" * 60)
                    print("예측 결과 (멀티타겟)")
                    print("=" * 60)
                    print(f"현재 가격: ${result['current_price']:,.2f}")
                    print(f"\n예측 가격:")
                    print(f"  3분 후: ${result.get('predicted_price_3m', result['predicted_price']):,.2f} ({result.get('change_3m', 0)*100:+.2f}%)")
                    print(f"  5분 후: ${result['predicted_price']:,.2f} ({result['price_change_pct']:+.2f}%)")
                    print(f"  15분 후: ${result.get('predicted_price_15m', result['predicted_price']):,.2f} ({result.get('change_15m', 0)*100:+.2f}%)")
                    print(f"\n시그널: {result['signal']}")
                    print(f"강도: {result['strength']:.2f}x")
                    print(f"신뢰도: {result['confidence']:.2f}%")
                    print(f"방향 일관성: {result['direction_consistency']}")
                    print(f"중기 추세 일관성: {result.get('trend_consistency', 'N/A')}")
                    print("=" * 60)
                    
                    # 시그널이 강할 때 강조
                    if result['signal'] != '보유' and result['strength'] > 1.5:
                        print(f"\n⚠️  강한 {result['signal']} 시그널! (강도: {result['strength']:.2f}x)")
                else:
                    print(f"\n❌ 예측 실패: {result.get('error', 'Unknown error')}")
                
                # 다음 실행까지 대기
                print(f"\n다음 예측까지 {interval_minutes}분 대기 중...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n시스템 종료 중...")
            print("감사합니다!")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 비트코인 거래 시그널 생성')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='모델 파일 경로')
    parser.add_argument('--interval', type=int, default=1,
                       help='실행 간격 (분, 기본값: 1분)')
    parser.add_argument('--min-confidence', type=float, default=0.02,
                       help='최소 신뢰도 (기본값: 0.02 = 2%%)')
    parser.add_argument('--once', action='store_true',
                       help='한 번만 실행 (연속 실행 안 함)')
    
    args = parser.parse_args()
    
    # 시그널 생성기 초기화
    signal_generator = RealtimeTradingSignal(
        model_path=args.model,
        min_confidence=args.min_confidence
    )
    
    if args.once:
        # 한 번만 실행
        result = signal_generator.predict_and_signal()
        if result['success']:
            print("\n" + "=" * 60)
            print("예측 결과 (멀티타겟)")
            print("=" * 60)
            print(f"현재 가격: ${result['current_price']:,.2f}")
            print(f"\n예측 가격:")
            print(f"  3분 후: ${result.get('predicted_price_3m', result['predicted_price']):,.2f} ({result.get('change_3m', 0)*100:+.2f}%)")
            print(f"  5분 후: ${result['predicted_price']:,.2f} ({result['price_change_pct']:+.2f}%)")
            print(f"  15분 후: ${result.get('predicted_price_15m', result['predicted_price']):,.2f} ({result.get('change_15m', 0)*100:+.2f}%)")
            print(f"\n시그널: {result['signal']}")
            print(f"강도: {result['strength']:.2f}x")
            print(f"신뢰도: {result['confidence']:.2f}%")
            print(f"중기 추세 일관성: {result.get('trend_consistency', 'N/A')}")
            print("=" * 60)
        else:
            print(f"예측 실패: {result.get('error', 'Unknown error')}")
    else:
        # 연속 실행
        signal_generator.run_continuous(interval_minutes=args.interval)


class RealtimeTrader:
    """실시간 자동 거래 클래스"""
    
    def __init__(self,
                 model_path: str = 'models/best_model.h5',
                 window_size: int = 60,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 leverage: int = 30,
                 take_profit_roi: float = 0.05):  # 5% ROI
        """
        Args:
            model_path: 학습된 모델 파일 경로
            window_size: 슬라이딩 윈도우 크기
            api_key: 바이낸스 API 키 (환경변수 BINANCE_API_KEY에서도 읽을 수 있음)
            api_secret: 바이낸스 API 시크릿 (환경변수 BINANCE_API_SECRET에서도 읽을 수 있음)
            leverage: 레버리지 배수 (기본값: 30)
            take_profit_roi: Take Profit ROI (기본값: 0.05 = 5%)
        """
        self.model_path = model_path
        self.window_size = window_size
        self.leverage = leverage
        self.take_profit_roi = take_profit_roi
        
        # API 키 설정 (.env 파일 또는 환경변수에서 읽기)
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "바이낸스 API 키와 시크릿이 필요합니다.\n"
                "다음 중 하나의 방법으로 설정하세요:\n"
                "1. .env 파일에 BINANCE_API_KEY와 BINANCE_API_SECRET 추가\n"
                "2. 환경변수 BINANCE_API_KEY, BINANCE_API_SECRET 설정\n"
                "3. 생성자에 api_key, api_secret 직접 전달"
            )
        
        # 바이낸스 선물 거래소 초기화
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'options': {
                'defaultType': 'future',  # 선물 거래
            },
            'enableRateLimit': True,
            'sandbox': False,  # 실제 거래 (테스트는 True로 설정)
        })
        self.symbol = 'BTC/USDT'
        
        # 시그널 생성기 초기화
        self.signal_generator = RealtimeTradingSignal(
            model_path=model_path,
            window_size=window_size
        )
        
        # 시장 지표 분석기 초기화
        self.market_indicators = MarketIndicators(exchange=self.exchange)
        
        # 현재 포지션 정보
        self.current_position = None  # {'side': 'long', 'entry_price': float, 'size': float, 'entry_time': datetime}
        
        # 거래 조건 (3분봉 제외)
        self.min_change_5m = 0.0012  # 0.12%
        self.min_change_15m = 0.0020  # 0.20%
        
        print("=" * 60)
        print("실시간 자동 거래 시스템 초기화 완료")
        print(f"레버리지: {leverage}배")
        print(f"Take Profit ROI: {take_profit_roi*100:.1f}%")
        print("=" * 60)
    
    def get_account_balance(self) -> Dict:
        """계좌 잔액 조회"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {})
            free = usdt_balance.get('free', 0.0)
            total = usdt_balance.get('total', 0.0)
            
            return {
                'free': free,
                'total': total,
                'available': free  # 거래 가능 금액
            }
        except Exception as e:
            print(f"계좌 잔액 조회 실패: {e}")
            return {'free': 0.0, 'total': 0.0, 'available': 0.0}
    
    def get_current_position(self) -> Optional[Dict]:
        """현재 포지션 조회"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol and pos['contracts'] > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': abs(pos['contracts']),
                        'entry_price': pos['entryPrice'],
                        'mark_price': pos['markPrice'],
                        'unrealized_pnl': pos['unrealizedPnl'],
                        'percentage': pos['percentage']
                    }
            return None
        except Exception as e:
            print(f"포지션 조회 실패: {e}")
            return None
    
    def set_leverage(self, leverage: int):
        """레버리지 설정"""
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            print(f"레버리지 {leverage}배로 설정 완료")
        except Exception as e:
            print(f"레버리지 설정 실패: {e}")
    
    def open_short_position(self, amount_usdt: float, roi: Optional[float] = None) -> bool:
        """숏 포지션 열기 (시장가, 30배 레버리지, TP 자동 설정)
        
        Args:
            amount_usdt: 사용할 USDT 금액
            roi: Take Profit ROI (None이면 기본값 self.take_profit_roi 사용)
        """
        try:
            # ROI 설정 (파라미터가 없으면 기본값 사용)
            if roi is None:
                roi = self.take_profit_roi
            
            # 레버리지 설정
            self.set_leverage(self.leverage)
            
            # 현재 가격 조회
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # 수수료 고려 (바이낸스 선물 거래 수수료 약 0.04%)
            fee_rate = 0.0004
            # 마진 버퍼 (100% 사용하지 않고 95%만 사용하여 안전 마진 확보)
            margin_buffer = 0.95
            
            # 실제 사용 가능한 마진 계산
            usable_margin = amount_usdt * margin_buffer * (1 - fee_rate)
            
            # 레버리지 30배를 사용하므로, 포지션 가치 = 마진 * 레버리지
            position_value = usable_margin * self.leverage
            
            # BTC 수량 계산 (포지션 가치를 현재 가격으로 나눔)
            btc_quantity = position_value / current_price
            
            # 바이낸스 최소 주문 수량 확인 (BTC/USDT 선물: 0.001 BTC)
            min_quantity = 0.001
            if btc_quantity < min_quantity:
                print(f"⚠️ 주문 수량이 최소값보다 작습니다: {btc_quantity:.6f} BTC < {min_quantity} BTC")
                print(f"   필요한 최소 마진: ${(min_quantity * current_price / self.leverage / margin_buffer / (1 - fee_rate)):,.2f} USDT")
                return False
            
            # 수량을 바이낸스 규격에 맞게 반올림 (소수점 3자리)
            btc_quantity = round(btc_quantity, 3)
            
            # 시장가 매도 주문 (숏 포지션)
            # One-way Mode에서는 positionSide 파라미터를 사용하지 않음
            order = self.exchange.create_market_sell_order(
                self.symbol,
                btc_quantity,
                params={
                    'leverage': self.leverage
                }
            )
            
            print(f"✅ 숏 포지션 열기 성공")
            print(f"   주문 ID: {order.get('id', 'N/A')}")
            print(f"   주문 수량: {btc_quantity:.3f} BTC")
            print(f"   포지션 가치: ${position_value:,.2f} USDT")
            print(f"   사용 마진: ${usable_margin:,.2f} USDT")
            print(f"   가격: ${current_price:,.2f}")
            print(f"   레버리지: {self.leverage}배")
            
            # Take Profit 가격 계산 (레버리지 고려)
            # 레버리지 30배일 때, 실제 자본 대비 ROI 수익 = 가격 변동 ROI/30
            # 숏이므로 가격 하락 시 수익
            take_profit_price = current_price * (1 - roi / self.leverage)
            
            # Take Profit 주문 생성 (바이낸스에서 자동으로 포지션 닫기)
            try:
                try:
                    # TAKE_PROFIT_MARKET 주문 (closePosition: True 사용 시 수량 불필요)
                    tp_order = self.exchange.create_order(
                        self.symbol,
                        'TAKE_PROFIT_MARKET',
                        'buy',  # 숏 포지션을 닫기 위해 매수
                        None,  # closePosition: True일 때는 수량 불필요
                        None,
                        params={
                            'stopPrice': take_profit_price,
                            'closePosition': True
                        }
                    )
                except Exception as e1:
                    try:
                        # TAKE_PROFIT 주문 (triggerPrice 필요, closePosition 사용)
                        tp_order = self.exchange.create_order(
                            self.symbol,
                            'TAKE_PROFIT',
                            'buy',
                            None,  # closePosition 사용 시 수량 불필요
                            None,
                            params={
                                'triggerPrice': take_profit_price,
                                'closePosition': True,
                                'timeInForce': 'GTC'
                            }
                        )
                    except Exception as e2:
                        raise Exception(f"TP 주문 생성 실패 (방법 1: {e1}, 방법 2: {e2})")
                
                print(f"✅ Take Profit 주문 생성 성공")
                print(f"   TP 주문 ID: {tp_order.get('id', 'N/A')}")
                print(f"   TP 가격: ${take_profit_price:,.2f} (ROI {self.take_profit_roi*100:.1f}%)")
                print(f"   → 가격이 ${take_profit_price:,.2f}에 도달하면 바이낸스에서 자동으로 포지션이 닫힙니다")
            except Exception as tp_error:
                print(f"⚠️ Take Profit 주문 생성 실패: {tp_error}")
                print(f"   바이낸스 웹사이트에서 수동으로 TP를 설정하거나 포지션을 모니터링해야 합니다")
                print(f"   권장 TP 가격: ${take_profit_price:,.2f} (ROI {self.take_profit_roi*100:.1f}%)")
            
            return True
        except Exception as e:
            print(f"❌ 숏 포지션 열기 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def open_long_position(self, amount_usdt: float, roi: Optional[float] = None) -> bool:
        """롱 포지션 열기 (시장가, 30배 레버리지, TP 자동 설정)
        
        Args:
            amount_usdt: 사용할 USDT 금액
            roi: Take Profit ROI (None이면 기본값 self.take_profit_roi 사용)
        """
        try:
            # ROI 설정 (파라미터가 없으면 기본값 사용)
            if roi is None:
                roi = self.take_profit_roi
            
            # 레버리지 설정
            self.set_leverage(self.leverage)
            
            # 현재 가격 조회
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # 수수료 고려 (바이낸스 선물 거래 수수료 약 0.04%)
            fee_rate = 0.0004
            # 마진 버퍼 (100% 사용하지 않고 95%만 사용하여 안전 마진 확보)
            margin_buffer = 0.95
            
            # 실제 사용 가능한 마진 계산
            usable_margin = amount_usdt * margin_buffer * (1 - fee_rate)
            
            # 레버리지 30배를 사용하므로, 포지션 가치 = 마진 * 레버리지
            position_value = usable_margin * self.leverage
            
            # BTC 수량 계산 (포지션 가치를 현재 가격으로 나눔)
            btc_quantity = position_value / current_price
            
            # 바이낸스 최소 주문 수량 확인 (BTC/USDT 선물: 0.001 BTC)
            min_quantity = 0.001
            if btc_quantity < min_quantity:
                print(f"⚠️ 주문 수량이 최소값보다 작습니다: {btc_quantity:.6f} BTC < {min_quantity} BTC")
                print(f"   필요한 최소 마진: ${(min_quantity * current_price / self.leverage / margin_buffer / (1 - fee_rate)):,.2f} USDT")
                return False
            
            # 수량을 바이낸스 규격에 맞게 반올림 (소수점 3자리)
            btc_quantity = round(btc_quantity, 3)
            
            # 시장가 매수 주문 (롱 포지션)
            # One-way Mode에서는 positionSide 파라미터를 사용하지 않음
            order = self.exchange.create_market_buy_order(
                self.symbol,
                btc_quantity,
                params={
                    'leverage': self.leverage
                }
            )
            
            print(f"✅ 롱 포지션 열기 성공")
            print(f"   주문 ID: {order.get('id', 'N/A')}")
            print(f"   주문 수량: {btc_quantity:.3f} BTC")
            print(f"   포지션 가치: ${position_value:,.2f} USDT")
            print(f"   사용 마진: ${usable_margin:,.2f} USDT")
            print(f"   가격: ${current_price:,.2f}")
            print(f"   레버리지: {self.leverage}배")
            
            # Take Profit 가격 계산 (레버리지 고려)
            # 레버리지 30배일 때, 실제 자본 대비 ROI 수익 = 가격 변동 ROI/30
            take_profit_price = current_price * (1 + roi / self.leverage)
            
            # Take Profit 주문 생성 (바이낸스에서 자동으로 포지션 닫기)
            try:
                # 바이낸스 선물 거래소의 TP 주문 생성
                # 방법 1: TAKE_PROFIT_MARKET 주문 타입 사용
                try:
                    tp_order = self.exchange.create_order(
                        self.symbol,
                        'TAKE_PROFIT_MARKET',  # 주문 타입
                        'sell',  # 롱 포지션을 닫기 위해 매도
                        None,  # closePosition: True일 때는 수량 불필요
                        None,  # 가격은 stopPrice 사용
                        params={
                            'stopPrice': take_profit_price,  # TP 트리거 가격
                            'closePosition': True  # 포지션 전체 닫기
                        }
                    )
                except Exception as e1:
                    # 방법 2: TAKE_PROFIT 주문 (triggerPrice 필요)
                    try:
                        tp_order = self.exchange.create_order(
                            self.symbol,
                            'TAKE_PROFIT',
                            'sell',
                            None,  # closePosition 사용 시 수량 불필요
                            None,
                            params={
                                'triggerPrice': take_profit_price,
                                'closePosition': True,
                                'timeInForce': 'GTC'
                            }
                        )
                    except Exception as e2:
                        raise Exception(f"TP 주문 생성 실패 (방법 1: {e1}, 방법 2: {e2})")
                
                print(f"✅ Take Profit 주문 생성 성공")
                print(f"   TP 주문 ID: {tp_order.get('id', 'N/A')}")
                print(f"   TP 가격: ${take_profit_price:,.2f} (ROI {roi*100:.1f}%)")
                print(f"   → 가격이 ${take_profit_price:,.2f}에 도달하면 바이낸스에서 자동으로 포지션이 닫힙니다")
            except Exception as tp_error:
                print(f"⚠️ Take Profit 주문 생성 실패: {tp_error}")
                print(f"   바이낸스 웹사이트에서 수동으로 TP를 설정하거나 포지션을 모니터링해야 합니다")
                print(f"   권장 TP 가격: ${take_profit_price:,.2f} (ROI {roi*100:.1f}%)")
                # TP 주문 실패해도 포지션은 열렸으므로 계속 진행
            
            return True
        except Exception as e:
            print(f"❌ 롱 포지션 열기 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close_position(self) -> bool:
        """현재 포지션 닫기 (시장가)"""
        try:
            position = self.get_current_position()
            if not position:
                print("닫을 포지션이 없습니다.")
                return False
            
            # 반대 주문으로 포지션 닫기
            # One-way Mode에서는 positionSide 파라미터를 사용하지 않음
            if position['side'] == 'long':
                # 롱 포지션은 매도로 닫기
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    position['size']
                )
            else:
                # 숏 포지션은 매수로 닫기
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    position['size']
                )
            
            print(f"✅ 포지션 닫기 성공")
            print(f"   주문 ID: {order.get('id', 'N/A')}")
            print(f"   포지션: {position['side'].upper()}")
            print(f"   수량: {position['size']:.6f} BTC")
            print(f"   진입 가격: ${position['entry_price']:,.2f}")
            print(f"   종료 가격: ${position['mark_price']:,.2f}")
            print(f"   수익: ${position['unrealized_pnl']:,.2f} ({position['percentage']:.2f}%)")
            self.current_position = None
            return True
        except Exception as e:
            print(f"❌ 포지션 닫기 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_existing_tp_orders(self) -> bool:
        """기존 TP 주문 확인 (바이낸스에서 자동으로 처리하므로 확인만)"""
        try:
            # 열린 주문 조회
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            tp_orders = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET' or 'TAKE_PROFIT' in o.get('type', '')]
            
            if tp_orders:
                print(f"📋 활성 TP 주문: {len(tp_orders)}개")
                for tp in tp_orders:
                    print(f"   TP 주문 ID: {tp.get('id')}, 가격: ${tp.get('stopPrice', tp.get('price', 0)):,.2f}")
                return True
            else:
                return False
        except Exception as e:
            print(f"⚠️ TP 주문 확인 실패: {e}")
            return False
    
    def check_trade_conditions(self, change_3m: float, change_5m: float, change_15m: float) -> Optional[str]:
        """거래 조건 확인
        
        조건:
        - 5분봉 변화율 >= 0.12%
        - 15분봉 변화율 >= 0.20%
        - 5분봉과 15분봉이 같은 부호
        (3분봉은 조건에서 제외)
        
        Returns:
            'long': 롱 주문 조건 충족
            'short': 숏 주문 조건 충족
            None: 조건 미충족
        """
        # 최소 변화율 조건 확인 (5분봉, 15분봉만)
        # 부동소수점 오차를 고려하여 >= 비교 사용
        epsilon = 1e-8  # 매우 작은 epsilon (부동소수점 오차 보정용)
        if abs(change_5m) < self.min_change_5m - epsilon:
            return None
        if abs(change_15m) < self.min_change_15m - epsilon:
            return None
        
        # 같은 부호 확인
        both_positive = change_5m > 0 and change_15m > 0
        both_negative = change_5m < 0 and change_15m < 0
        
        if both_positive:
            return 'long'  # 롱 주문
        elif both_negative:
            return 'short'  # 숏 주문
        else:
            return None  # 조건 미충족
    
    def execute_trading_cycle(self):
        """한 번의 거래 사이클 실행 (1분마다)"""
        try:
            # 1. 예측 수행
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 예측 및 거래 조건 확인 중...")
            result = self.signal_generator.predict_and_signal()
            
            if not result['success']:
                print(f"❌ 예측 실패: {result.get('error', 'Unknown error')}")
                return
            
            # 예측 결과 출력
            change_3m = result.get('change_3m', 0)
            change_5m = result.get('change_5m', 0)
            change_15m = result.get('change_15m', 0)
            current_price = result.get('current_price', 0)
            
            print("\n" + "=" * 60)
            print("예측 결과")
            print("=" * 60)
            print(f"현재 가격: ${current_price:,.2f}")
            print(f"3분봉 변화율: {change_3m*100:+.2f}%")
            print(f"5분봉 변화율: {change_5m*100:+.2f}%")
            print(f"15분봉 변화율: {change_15m*100:+.2f}%")
            
            # 1.5. 시장 지표 분석
            print("\n" + "-" * 60)
            print("시장 지표 분석")
            print("-" * 60)
            market_signal = {'signal': 'neutral', 'confidence': 0.0}  # 기본값 설정
            try:
                market_signal = self.market_indicators.get_trading_signal_from_indicators()
                indicators = market_signal.get('indicators', {})
                
                # 오더북 불균형
                ob = indicators.get('orderbook_imbalance', {})
                print(f"📊 오더북 불균형: {ob.get('imbalance_strength', 'neutral')} (비율: {ob.get('imbalance_ratio', 0)*100:+.2f}%)")
                
                # 청산 클러스터
                lc = indicators.get('liquidation_clusters', {})
                print(f"💥 청산 클러스터: {lc.get('liquidation_strength', 'neutral')} (비율: {lc.get('liquidation_ratio', 0)*100:+.2f}%)")
                
                # 변동성 압축
                vs = indicators.get('volatility_squeeze', {})
                print(f"📉 변동성: {vs.get('squeeze_status', 'normal')} (폭발 가능성: {vs.get('expansion_potential', 'low')})")
                
                # OI 급증
                oi = indicators.get('oi_surge', {})
                print(f"💰 OI: {oi.get('oi_surge_status', 'normal')} (방향: {oi.get('oi_direction', 'balanced')}, 펀딩: {oi.get('funding_rate_pct', 0):+.4f}%)")
                
                # CVD 전환
                cvd = indicators.get('cvd_turnover', {})
                print(f"🔄 CVD: {cvd.get('cvd_trend', 'neutral')} (전환: {'예' if cvd.get('cvd_turnover', False) else '아니오'})")
                
                # 종합 신호
                print(f"\n🎯 시장 지표 종합 신호: {market_signal.get('signal', 'neutral')} (신뢰도: {market_signal.get('confidence', 0)*100:.1f}%)")
                if market_signal.get('reasons'):
                    print("   근거:")
                    for reason in market_signal['reasons']:
                        print(f"     - {reason}")
                
            except Exception as e:
                print(f"⚠️ 시장 지표 분석 실패: {e}")
                market_signal = {'signal': 'neutral', 'confidence': 0.0}
            
            # 2. 계좌 정보 조회 및 표시
            balance = self.get_account_balance()
            print(f"\n계좌 정보")
            print(f"  총 자산: ${balance['total']:,.2f} USDT")
            print(f"  거래 가능: ${balance['available']:,.2f} USDT")
            print(f"  사용 중: ${balance['total'] - balance['available']:,.2f} USDT")
            
            # 3. 현재 포지션 확인
            position = self.get_current_position()
            
            if position:
                print(f"\n현재 포지션: {position['side'].upper()}")
                print(f"진입 가격: ${position['entry_price']:,.2f}")
                print(f"현재 가격: ${position['mark_price']:,.2f}")
                print(f"미실현 손익: ${position['unrealized_pnl']:,.2f} ({position['percentage']:.2f}%)")
                
                # 4. TP 주문 확인 (바이낸스에서 자동으로 처리하므로 확인만)
                self.check_existing_tp_orders()
                
                # ROI 계산 및 표시
                if position['side'] == 'long':
                    roi = (position['mark_price'] - position['entry_price']) / position['entry_price']
                else:
                    roi = (position['entry_price'] - position['mark_price']) / position['entry_price']
                
                target_roi = self.take_profit_roi
                print(f"현재 ROI: {roi*100:.2f}% (목표: {target_roi*100:.1f}%)")
                
                if roi >= target_roi:
                    print(f"🎯 Take Profit 목표 달성! (바이낸스에서 자동으로 포지션이 닫힐 예정)")
                else:
                    remaining = ((target_roi - roi) / target_roi) * 100
                    print(f"목표까지 남은 수익률: {remaining:.1f}%")
            else:
                print("\n현재 포지션 없음")
                
                # 4. 거래 조건 확인 (롱/숏)
                trade_signal = self.check_trade_conditions(change_3m, change_5m, change_15m)
                
                # 4.5. 시장 지표 방향 확인
                market_signal_value = market_signal.get('signal', 'neutral')
                market_confidence = market_signal.get('confidence', 0.0)
                
                # 시장 지표 방향 판단
                market_direction = None
                if market_signal_value in ['strong_buy', 'buy']:
                    market_direction = 'long'
                elif market_signal_value in ['strong_sell', 'sell']:
                    market_direction = 'short'
                else:
                    market_direction = None  # neutral
                
                # 4.6. 종합 조건 확인: 5분봉, 15분봉, 시장 지표 모두 같은 방향이어야 함
                final_trade_signal = None
                
                if trade_signal:
                    # 5분봉 방향 확인
                    direction_5m = 'long' if change_5m > 0 else ('short' if change_5m < 0 else None)
                    # 15분봉 방향 확인
                    direction_15m = 'long' if change_15m > 0 else ('short' if change_15m < 0 else None)
                    
                    print(f"\n📊 방향성 분석:")
                    print(f"   5분봉 예측: {direction_5m} ({change_5m*100:+.2f}%)")
                    print(f"   15분봉 예측: {direction_15m} ({change_15m*100:+.2f}%)")
                    print(f"   시장 지표: {market_direction} ({market_signal_value}, 신뢰도: {market_confidence*100:.1f}%)")
                    
                    # 세 가지가 모두 같은 방향인지 확인
                    if trade_signal == 'long':
                        if direction_5m == 'long' and direction_15m == 'long' and market_direction == 'long':
                            final_trade_signal = 'long'
                            print(f"\n✅ 롱 주문 조건 충족! (5분봉, 15분봉, 시장지표 모두 상승 방향)")
                        else:
                            print(f"\n❌ 롱 주문 조건 미충족:")
                            if direction_5m != 'long':
                                print(f"   - 5분봉 방향 불일치: {direction_5m}")
                            if direction_15m != 'long':
                                print(f"   - 15분봉 방향 불일치: {direction_15m}")
                            if market_direction != 'long':
                                print(f"   - 시장 지표 방향 불일치: {market_direction} ({market_signal_value})")
                    
                    elif trade_signal == 'short':
                        if direction_5m == 'short' and direction_15m == 'short' and market_direction == 'short':
                            final_trade_signal = 'short'
                            print(f"\n✅ 숏 주문 조건 충족! (5분봉, 15분봉, 시장지표 모두 하락 방향)")
                        else:
                            print(f"\n❌ 숏 주문 조건 미충족:")
                            if direction_5m != 'short':
                                print(f"   - 5분봉 방향 불일치: {direction_5m}")
                            if direction_15m != 'short':
                                print(f"   - 15분봉 방향 불일치: {direction_15m}")
                            if market_direction != 'short':
                                print(f"   - 시장 지표 방향 불일치: {market_direction} ({market_signal_value})")
                
                # 최종 거래 신호로 업데이트
                trade_signal = final_trade_signal
                
                if trade_signal:
                    
                    # 5. 거래 가능 금액 확인 (이미 조회한 balance 사용)
                    available = balance['available']
                    
                    # 최소 거래 금액 계산 (수수료 및 마진 버퍼 고려)
                    # 최소 주문 수량 0.001 BTC를 위한 최소 마진 계산
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                    fee_rate = 0.0004
                    margin_buffer = 0.95
                    min_margin_required = (0.001 * current_price / self.leverage / margin_buffer / (1 - fee_rate))
                    min_trade_amount = max(10.0, min_margin_required * 1.1)  # 10% 여유
                    
                    if available >= min_trade_amount:
                        # 6. ROI 계산 (5분봉 변화율에 따라 동적 조정)
                        # 5분봉 변화율이 0.30% (0.003) 이상이면 ROI 10%, 아니면 기본값 5%
                        if abs(change_5m) >= 0.003:
                            dynamic_roi = 0.10  # 10%
                            print(f"\n📊 5분봉 변화율 {abs(change_5m)*100:.2f}% >= 0.30% → ROI 10%로 설정")
                        else:
                            dynamic_roi = self.take_profit_roi  # 기본값 5%
                            print(f"\n📊 5분봉 변화율 {abs(change_5m)*100:.2f}% < 0.30% → ROI {dynamic_roi*100:.1f}% 사용")
                        
                        # 7. 포지션 열기 (95% 자금 사용, 30배 레버리지, 수수료 고려)
                        print(f"\n💰 포지션 열기 시도: ${available:,.2f} USDT (95% 사용, {self.leverage}배 레버리지, ROI {dynamic_roi*100:.1f}%)")
                        
                        if trade_signal == 'long':
                            success = self.open_long_position(available, roi=dynamic_roi)
                        else:  # short
                            success = self.open_short_position(available, roi=dynamic_roi)
                        
                        if success:
                            print(f"✅ {trade_signal.upper()} 포지션 열기 완료!")
                        else:
                            print(f"❌ {trade_signal.upper()} 포지션 열기 실패")
                    else:
                        print(f"⚠️ 거래 가능 금액이 부족합니다: ${available:,.2f} USDT")
                        print(f"   최소 필요 금액: ${min_trade_amount:,.2f} USDT (최소 주문 수량 0.001 BTC 기준)")
                else:
                    print("\n❌ 거래 조건 미충족")
                    print(f"  - 3분봉: {change_3m*100:+.2f}% (조건에서 제외)")
                    
                    # 출력용 체크 (실제 조건 체크와 동일한 로직 사용)
                    epsilon = 1e-8
                    check_5m = abs(change_5m) >= (self.min_change_5m - epsilon)
                    check_15m = abs(change_15m) >= (self.min_change_15m - epsilon)
                    
                    # 실제 값과 비교값을 더 정확하게 표시
                    print(f"  - 5분봉: {change_5m*100:+.2f}% (절댓값: {abs(change_5m)*100:.4f}%) {'✓' if check_5m else '✗'} (최소 {self.min_change_5m*100:.2f}% = {self.min_change_5m:.6f})")
                    print(f"  - 15분봉: {change_15m*100:+.2f}% (절댓값: {abs(change_15m)*100:.4f}%) {'✓' if check_15m else '✗'} (최소 {self.min_change_15m*100:.2f}% = {self.min_change_15m:.6f})")
                    
                    # 시장 지표 방향 확인
                    market_signal_value = market_signal.get('signal', 'neutral')
                    market_confidence = market_signal.get('confidence', 0.0)
                    market_direction = None
                    if market_signal_value in ['strong_buy', 'buy']:
                        market_direction = 'long'
                    elif market_signal_value in ['strong_sell', 'sell']:
                        market_direction = 'short'
                    else:
                        market_direction = 'neutral'
                    
                    # 5분봉, 15분봉 방향 확인
                    direction_5m = 'long' if change_5m > 0 else ('short' if change_5m < 0 else 'neutral')
                    direction_15m = 'long' if change_15m > 0 else ('short' if change_15m < 0 else 'neutral')
                    
                    # 부호 일치 확인 (5분봉과 15분봉)
                    both_positive = change_5m > 0 and change_15m > 0
                    both_negative = change_5m < 0 and change_15m < 0
                    same_sign = both_positive or both_negative
                    
                    print(f"  - 부호 일치: {'✓' if same_sign else '✗'} (5분봉과 15분봉 같은 부호)")
                    
                    # 방향성 종합 분석
                    print(f"\n📊 방향성 분석:")
                    print(f"   5분봉 예측: {direction_5m} ({change_5m*100:+.2f}%)")
                    print(f"   15분봉 예측: {direction_15m} ({change_15m*100:+.2f}%)")
                    print(f"   시장 지표: {market_direction} ({market_signal_value}, 신뢰도: {market_confidence*100:.1f}%)")
                    
                    # 세 가지 방향 일치 여부 확인
                    all_long = direction_5m == 'long' and direction_15m == 'long' and market_direction == 'long'
                    all_short = direction_5m == 'short' and direction_15m == 'short' and market_direction == 'short'
                    
                    if all_long:
                        print(f"  - 방향 일치: ✓ (모두 상승 → 롱 주문 가능)")
                    elif all_short:
                        print(f"  - 방향 일치: ✓ (모두 하락 → 숏 주문 가능)")
                    else:
                        print(f"  - 방향 일치: ✗ (5분봉, 15분봉, 시장지표 방향 불일치)")
                        if direction_5m != direction_15m:
                            print(f"     → 5분봉({direction_5m})과 15분봉({direction_15m}) 불일치")
                        if direction_5m != market_direction and market_direction != 'neutral':
                            print(f"     → 5분봉({direction_5m})과 시장지표({market_direction}) 불일치")
                        if direction_15m != market_direction and market_direction != 'neutral':
                            print(f"     → 15분봉({direction_15m})과 시장지표({market_direction}) 불일치")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 거래 사이클 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def run_continuous(self, interval_minutes: int = 1):
        """연속적으로 거래 실행 (1분마다)
        
        Args:
            interval_minutes: 실행 간격 (기본값: 1분)
        """
        print("=" * 60)
        print("실시간 자동 거래 시스템 시작")
        print(f"실행 간격: {interval_minutes}분 (예측 + 시장 지표)")
        print("종료하려면 Ctrl+C를 누르세요")
        print("=" * 60)
        
        # 초기 레버리지 설정
        try:
            self.set_leverage(self.leverage)
        except Exception as e:
            print(f"⚠️ 레버리지 설정 경고: {e}")
        
        try:
            while True:
                self.execute_trading_cycle()
                
                # 다음 실행까지 대기
                print(f"\n다음 실행까지 {interval_minutes}분 대기 중...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n시스템 종료 중...")
            
            # 종료 시 포지션이 있으면 확인
            position = self.get_current_position()
            if position:
                print(f"\n⚠️ 현재 포지션이 있습니다: {position['side'].upper()}")
                response = input("포지션을 닫으시겠습니까? (y/n): ")
                if response.lower() == 'y':
                    self.close_position()
            
            print("감사합니다!")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 비트코인 거래 시그널 생성')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                       help='모델 파일 경로')
    parser.add_argument('--interval', type=int, default=1,
                       help='실행 간격 (분, 기본값: 1분)')
    parser.add_argument('--min-confidence', type=float, default=0.02,
                       help='최소 신뢰도 (기본값: 0.02 = 2%%)')
    parser.add_argument('--once', action='store_true',
                       help='한 번만 실행 (연속 실행 안 함)')
    parser.add_argument('--trade', action='store_true',
                       help='실거래 모드 활성화 (바이낸스 API 필요)')
    parser.add_argument('--leverage', type=int, default=30,
                       help='레버리지 배수 (기본값: 30)')
    parser.add_argument('--take-profit', type=float, default=0.05,
                       help='Take Profit ROI (기본값: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    if args.trade:
        # 실거래 모드
        trader = RealtimeTrader(
            model_path=args.model,
            leverage=args.leverage,
            take_profit_roi=args.take_profit
        )
        trader.run_continuous(interval_minutes=args.interval)
    else:
        # 시그널만 생성 모드
        signal_generator = RealtimeTradingSignal(
            model_path=args.model,
            min_confidence=args.min_confidence
        )
        
        if args.once:
            # 한 번만 실행
            result = signal_generator.predict_and_signal()
            if result['success']:
                print("\n" + "=" * 60)
                print("예측 결과 (멀티타겟)")
                print("=" * 60)
                print(f"현재 가격: ${result['current_price']:,.2f}")
                print(f"\n예측 가격:")
                print(f"  3분 후: ${result.get('predicted_price_3m', result['predicted_price']):,.2f} ({result.get('change_3m', 0)*100:+.2f}%)")
                print(f"  5분 후: ${result['predicted_price']:,.2f} ({result['price_change_pct']:+.2f}%)")
                print(f"  15분 후: ${result.get('predicted_price_15m', result['predicted_price']):,.2f} ({result.get('change_15m', 0)*100:+.2f}%)")
                print(f"\n시그널: {result['signal']}")
                print(f"강도: {result['strength']:.2f}x")
                print(f"신뢰도: {result['confidence']:.2f}%")
                print(f"중기 추세 일관성: {result.get('trend_consistency', 'N/A')}")
                print("=" * 60)
            else:
                print(f"예측 실패: {result.get('error', 'Unknown error')}")
        else:
            # 연속 실행
            signal_generator.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    main()


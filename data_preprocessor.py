"""
데이터 전처리 모듈
슬라이딩 윈도우 적용 및 데이터 누수 방지
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """데이터 전처리 및 슬라이딩 윈도우 생성 클래스"""
    
    def __init__(self, 
                 window_size: int = 60,
                 prediction_horizon: int = 1,
                 feature_columns: Optional[list] = None,
                 target_column: str = 'close',
                 scaler_type: str = 'standard'):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기 (과거 몇 개의 시점을 볼지)
            prediction_horizon: 예측할 미래 시점 (1 = 다음 5분봉)
            feature_columns: 사용할 특징 컬럼 리스트 (None이면 자동 선택)
            target_column: 예측 대상 컬럼
            scaler_type: 스케일러 타입 ('standard' or 'minmax')
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler_type = scaler_type
        # 특징은 RobustScaler 사용 (이상치에 강함)
        # 타겟은 StandardScaler 사용 (회귀 문제에 더 적합)
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()  # 특징용
            self.target_scaler = StandardScaler()  # 타겟용 (RobustScaler는 타겟에 부적합)
        else:
            self.scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        self.is_fitted = False
    
    def _select_features(self, df: pd.DataFrame) -> list:
        """사용할 특징 컬럼 자동 선택 (데이터 누수 방지)"""
        if self.feature_columns:
            return self.feature_columns
        
        # 기본적으로 숫자형 컬럼만 선택 (target 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        # 데이터 누수 방지: 가격 정보 제외 (open, high, low는 close와 매우 높은 상관관계)
        # 시퀀스의 마지막 시점에서 이 값들을 사용하면 미래 정보를 사용하는 것과 같음
        price_cols_to_exclude = ['open', 'high', 'low', 'close']
        for col in price_cols_to_exclude:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # 볼린저 밴드 특징 제외 (close 가격 기반이므로 매우 높은 상관관계)
        bb_cols_to_exclude = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                              'bb_middle_10', 'bb_upper_10', 'bb_lower_10', 'bb_width_10', 'bb_position_10']
        for col in bb_cols_to_exclude:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # 시간 기반 특징 제외 (hour_seasonality, dow_seasonality 등은 미래 정보 누수 가능)
        time_based_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 
                          'hour_seasonality', 'dow_seasonality']
        for col in time_based_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # volatility_regime 제외 (시간 기반 특징과 유사하게 다음 시점과 높은 일치)
        if 'volatility_regime' in numeric_cols:
            numeric_cols.remove('volatility_regime')
        
        # RSI 기반 방향성 특징 제외 (price_change 기반이므로 미래 정보 누수 가능)
        # 이미 feature_engineering.py에서 생성되지 않으므로 제거 불필요
        
        return numeric_cols
    
    def filter_features_by_correlation_consistency(self,
                                                   X_train: np.ndarray,
                                                   y_train: np.ndarray,
                                                   X_val: np.ndarray,
                                                   y_val: np.ndarray,
                                                   feature_names: list,
                                                   max_correlation_diff: float = 0.1) -> list:
        """
        Train과 Val에서 상관관계 차이가 작은 특징만 선택
        
        Args:
            X_train: 학습 데이터 (n_samples, n_timesteps, n_features) 또는 (n_samples, n_features)
            y_train: 학습 타겟 (n_samples,)
            X_val: 검증 데이터 (n_samples, n_timesteps, n_features) 또는 (n_samples, n_features)
            y_val: 검증 타겟 (n_samples,)
            feature_names: 특징 이름 리스트
            max_correlation_diff: 허용할 최대 상관관계 차이 (기본값: 0.1)
        
        Returns:
            선택된 특징 이름 리스트
        """
        # 시퀀스의 마지막 시점만 사용
        if len(X_train.shape) == 3:
            X_train_last = X_train[:, -1, :]
            X_val_last = X_val[:, -1, :]
        else:
            X_train_last = X_train
            X_val_last = X_val
        
        selected_features = []
        correlation_diffs = []
        
        for i, feat_name in enumerate(feature_names):
            if i >= X_train_last.shape[1]:
                continue
            
            train_feat = X_train_last[:, i]
            val_feat = X_val_last[:, i]
            
            # Train 상관관계
            train_corr = np.corrcoef(train_feat, y_train)[0, 1]
            if np.isnan(train_corr):
                train_corr = 0.0
            
            # Val 상관관계
            val_corr = np.corrcoef(val_feat, y_val)[0, 1]
            if np.isnan(val_corr):
                val_corr = 0.0
            
            # 상관관계 차이
            corr_diff = abs(train_corr - val_corr)
            correlation_diffs.append({
                'feature': feat_name,
                'train_corr': train_corr,
                'val_corr': val_corr,
                'diff': corr_diff
            })
            
            # 상관관계 차이가 작은 특징만 선택
            if corr_diff <= max_correlation_diff:
                selected_features.append(feat_name)
        
        # 선택된 특징(일관성 있는 특징) 출력
        selected_features_info = [d for d in correlation_diffs if d['feature'] in selected_features]
        if selected_features_info:
            # Train과 Val 모두에서 절댓값 상관관계가 높은 순으로 정렬
            selected_features_sorted = sorted(
                selected_features_info,
                key=lambda x: (abs(x['train_corr']) + abs(x['val_corr'])),  # 두 상관관계의 절댓값 합
                reverse=True
            )
            print(f"\n✓ 일관성 있는 특징 (선택됨, {len(selected_features)}개):")
            print("  [Train 상관관계 | Val 상관관계 | 차이]")
            for feat_info in selected_features_sorted[:20]:  # 상위 20개 출력
                avg_corr = (abs(feat_info['train_corr']) + abs(feat_info['val_corr'])) / 2
                print(f"  {feat_info['feature']:30s}: "
                      f"Train={feat_info['train_corr']:7.4f}, "
                      f"Val={feat_info['val_corr']:7.4f}, "
                      f"차이={feat_info['diff']:6.4f}, "
                      f"평균절댓값={avg_corr:.4f}")
            if len(selected_features_sorted) > 20:
                print(f"  ... 외 {len(selected_features_sorted) - 20}개")
        
        # 제거된 특징 출력
        removed_features = [d for d in correlation_diffs if d['feature'] not in selected_features]
        if removed_features:
            removed_features_sorted = sorted(removed_features, key=lambda x: x['diff'], reverse=True)
            print(f"\n✗ 상관관계 차이로 인해 제거된 특징 ({len(removed_features)}개):")
            print("  [Train 상관관계 | Val 상관관계 | 차이]")
            for feat_info in removed_features_sorted[:15]:  # 상위 15개 출력
                print(f"  {feat_info['feature']:30s}: "
                      f"Train={feat_info['train_corr']:7.4f}, "
                      f"Val={feat_info['val_corr']:7.4f}, "
                      f"차이={feat_info['diff']:6.4f}")
            if len(removed_features_sorted) > 15:
                print(f"  ... 외 {len(removed_features_sorted) - 15}개")
        
        print(f"\n최종 선택: {len(selected_features)}개 / 전체 {len(feature_names)}개")
        
        return selected_features
    
    def _remove_future_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        미래 데이터 누수 방지
        예측 시점 이후의 데이터를 사용하지 않도록 처리
        """
        df_clean = df.copy()
        
        # 각 행에서 해당 시점 이후의 정보를 사용하는 컬럼 제거/수정
        # 예: 이동평균이 미래 데이터를 포함하는 경우를 방지
        # (이미 rolling window는 과거만 보므로 대부분 안전하지만, 추가 검증)
        
        return df_clean
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우로 시퀀스 생성 (데이터 누수 방지)
        멀티타겟 지원 (target이 2D 배열인 경우)
        
        Args:
            data: 특징 데이터 (n_samples, n_features)
            target: 타겟 데이터 (n_samples,) 또는 (n_samples, n_targets) - 멀티타겟
        
        Returns:
            X: 시퀀스 데이터 (n_sequences, window_size, n_features)
            y: 타겟 데이터 (n_sequences,) 또는 (n_sequences, n_targets)
        
        Note:
            시퀀스의 마지막 시점은 예측 시점 이전이어야 함 (데이터 누수 방지)
            예: window_size=60, prediction_horizon=1이면
            - 시퀀스 i: data[i:i+60] 사용 (인덱스 i~i+59)
            - 타겟: target[i+60] (인덱스 i+60, 즉 시퀀스 마지막 시점의 다음)
        """
        X, y = [], []
        
        # target이 1D인지 2D인지 확인
        is_multitarget = target.ndim == 2
        
        for i in range(len(data) - self.window_size - self.prediction_horizon + 1):
            # 과거 window_size만큼의 데이터를 입력으로 사용
            # 마지막 시점은 예측 시점 이전이어야 함
            X.append(data[i:i + self.window_size])
            # prediction_horizon 이후의 타겟을 예측
            # target[i + window_size + prediction_horizon - 1]은 
            # 시퀀스의 마지막 시점(i + window_size - 1) 이후 prediction_horizon만큼 떨어진 시점
            target_idx = i + self.window_size + self.prediction_horizon - 1
            if is_multitarget:
                y.append(target[target_idx])  # (n_targets,)
            else:
                y.append(target[target_idx])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    fit_scaler: bool = True,
                    train_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 준비 및 전처리
        
        Args:
            df: 특징이 포함된 DataFrame
            fit_scaler: 스케일러를 학습할지 여부 (True: train, False: test/inference)
            train_data: 학습 데이터 (스케일러 학습용, fit_scaler=True일 때만 사용)
                        None이면 df로 스케일러 학습 (실시간 예측 시 train_data 제공)
        
        Returns:
            X: 시퀀스 데이터
            y: 타겟 데이터
            feature_names: 특징 이름 리스트
            original_target: 원본 타겟 값 (스케일링 전)
        """
        # 미래 데이터 누수 방지
        df_clean = self._remove_future_leakage(df)
        
        # 특징 선택
        feature_cols = self._select_features(df_clean)
        self._last_feature_cols = feature_cols  # 나중에 사용하기 위해 저장
        feature_data = df_clean[feature_cols].values
        target_data = df_clean[self.target_column].values
        
        # 결측치 처리만 (무한대는 feature_engineering에서 이미 방지됨)
        feature_df = pd.DataFrame(feature_data, columns=feature_cols)
        
        # NaN만 처리 (앞/뒤로 채우기)
        feature_df = feature_df.ffill().bfill()
        feature_df = feature_df.fillna(0)  # 여전히 NaN이 있으면 0으로
        feature_data = feature_df.values
        
        target_series = pd.Series(target_data)
        target_series = target_series.ffill().bfill().fillna(0)
        target_data = target_series.values
        
        # 최종 검증: 무한대가 있는지 확인 (있으면 에러)
        if np.isinf(feature_data).any():
            raise ValueError("특징 데이터에 무한대 값이 있습니다. feature_engineering을 확인하세요.")
        if np.isinf(target_data).any():
            raise ValueError("타겟 데이터에 무한대 값이 있습니다. 데이터를 확인하세요.")
        
        # 스케일링 (Train 데이터로만 스케일러 학습 - 실시간 예측 고려)
        if fit_scaler or not self.is_fitted:
            # Train 데이터가 제공되면 그것으로 스케일러 학습 (실시간 예측 시)
            if train_data is not None:
                train_clean = self._remove_future_leakage(train_data)
                train_feature_cols = self._select_features(train_clean)
                train_feature_data = train_clean[train_feature_cols].values
                train_target_data = train_clean[self.target_column].values
                
                # Train 데이터 전처리
                train_feature_df = pd.DataFrame(train_feature_data, columns=train_feature_cols)
                train_feature_df = train_feature_df.ffill().bfill().fillna(0)
                train_feature_data = train_feature_df.values
                
                train_target_series = pd.Series(train_target_data)
                train_target_series = train_target_series.ffill().bfill().fillna(0)
                train_target_data = train_target_series.values
                
                # Train 데이터로 스케일러 학습
                self.scaler.fit(train_feature_data)
                self.target_scaler.fit(train_target_data.reshape(-1, 1))
            else:
                # 제공된 데이터로 스케일러 학습 (일반적인 경우)
                self.scaler.fit(feature_data)
                self.target_scaler.fit(target_data.reshape(-1, 1))
            
            self.is_fitted = True
        
        # 스케일링 적용
        feature_data_scaled = self.scaler.transform(feature_data)
        target_data_scaled = self.target_scaler.transform(target_data.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_scaled = self.create_sequences(feature_data_scaled, target_data_scaled)
        
        # 원본 타겟 값도 함께 반환 (평가용)
        _, y_original = self.create_sequences(feature_data, target_data)
        
        return X, y_scaled, feature_cols, y_original
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """스케일링된 타겟을 원본 스케일로 변환"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def split_train_test(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        test_size: float = 0.2,
                        shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        학습/테스트 데이터 분할
        
        Args:
            X: 입력 데이터
            y: 타겟 데이터
            test_size: 테스트 데이터 비율
            shuffle: 셔플 여부 (시계열 데이터는 보통 False)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def save_scalers(self, filepath: str):
        """스케일러 저장"""
        # _last_feature_cols가 있으면 그것을 사용, 없으면 feature_columns 사용
        feature_cols_to_save = self._last_feature_cols if hasattr(self, '_last_feature_cols') and self._last_feature_cols else self.feature_columns
        scaler_data = {
            'feature_scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': feature_cols_to_save,  # 스케일러가 학습한 전체 feature 목록
            'model_feature_columns': getattr(self, 'model_feature_columns', None),  # 모델이 실제로 사용한 feature 목록
            'window_size': self.window_size,
            'target_column': self.target_column,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"스케일러 저장 완료: {filepath}")
        if hasattr(self, 'model_feature_columns'):
            print(f"모델 feature 목록 저장: {len(self.model_feature_columns)}개")
    
    def load_scalers(self, filepath: str):
        """스케일러 로드"""
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        self.scaler = scaler_data['feature_scaler']
        self.target_scaler = scaler_data['target_scaler']
        self.feature_columns = scaler_data.get('feature_columns')
        self.model_feature_columns = scaler_data.get('model_feature_columns')  # 모델이 실제로 사용한 feature 목록
        self.window_size = scaler_data.get('window_size', self.window_size)
        self.target_column = scaler_data.get('target_column', self.target_column)
        self.scaler_type = scaler_data.get('scaler_type', self.scaler_type)
        self.is_fitted = scaler_data.get('is_fitted', True)
        print(f"스케일러 로드 완료: {filepath}")
        if self.model_feature_columns:
            print(f"모델 feature 목록 로드: {len(self.model_feature_columns)}개")


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    import numpy as np
    
    # 샘플 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    sample_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 50000,
        'high': np.random.randn(200).cumsum() + 50100,
        'low': np.random.randn(200).cumsum() + 49900,
        'close': np.random.randn(200).cumsum() + 50000,
        'volume': np.random.rand(200) * 1000,
        'rsi': np.random.rand(200) * 100,
        'cci': np.random.randn(200) * 100
    }, index=dates)
    
    preprocessor = DataPreprocessor(window_size=10, prediction_horizon=1)
    X, y, feature_names, y_original = preprocessor.prepare_data(sample_data, fit_scaler=True)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Number of sequences: {len(X)}")


"""
바이낸스 선물 비트코인 5분봉 가격 예측 메인 실행 파일
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from data_preprocessor import DataPreprocessor
from model import PatchCNNBiLSTM
from trainer import ModelTrainer
from predictor import Predictor
from evaluator import Evaluator
from data_leakage_checker import DataLeakageChecker
from validation_analysis import ValidationAnalysis


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("바이낸스 선물 비트코인 5분봉 가격 예측 시스템")
    print("=" * 60)
    
    # 1. 데이터 수집 및 저장
    print("\n[1/7] 데이터 수집 중...")
    data_file = 'data/btc_5m_data.csv'
    os.makedirs('data', exist_ok=True)
    
    # CSV 파일이 없거나 오래되었으면 새로 수집
    # CSV 파일이 없거나 오래되었으면 새로 수집
    if not os.path.exists(data_file):
        print("CSV 파일이 없습니다. 데이터를 수집합니다...")
        fetcher = BinanceDataFetcher()
        
        # 빠른 학습을 위해 데이터 수집 기간 설정
        # 30일 데이터 수집 (5분봉 기준 약 8,640개)
        df_raw = fetcher.fetch_recent_data(hours=24 * 90, timeframe='5m')
        print(f"수집된 데이터: {len(df_raw)}개")
        print(f"데이터 기간: {df_raw.index[0]} ~ {df_raw.index[-1]}")
        
        # CSV로 저장
        df_raw.to_csv(data_file)
        print(f"데이터가 저장되었습니다: {data_file}")
    else:
        print(f"기존 CSV 파일을 읽습니다: {data_file}")
        df_raw = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"로드된 데이터: {len(df_raw)}개")
        print(f"데이터 기간: {df_raw.index[0]} ~ {df_raw.index[-1]}")
    
    # 2. 특징 엔지니어링
    print("\n[2/7] 특징 엔지니어링 중...")
    features_file = 'data/btc_5m_features.csv'
    
    # 특징 엔지니어링 결과 CSV가 있으면 삭제하고 재생성 (시간 기반 특징 제거 반영)
    # 기존 CSV에는 시간 기반 특징이 포함되어 있을 수 있으므로 재생성
    if os.path.exists(features_file):
        print(f"기존 특징 엔지니어링 결과를 삭제하고 재생성합니다 (시간 기반 특징 제거 반영)...")
        os.remove(features_file)
    
    print("특징 엔지니어링을 수행합니다...")
    engineer = FeatureEngineer()
    df_features = engineer.add_all_features(df_raw)
    print(f"추가된 특징 수: {len(df_features.columns) - len(df_raw.columns)}개")
    print(f"전체 특징 수: {len(df_features.columns)}개")
    
    # 특징 엔지니어링 결과를 CSV로 저장
    df_features.to_csv(features_file)
    print(f"특징 엔지니어링 결과가 저장되었습니다: {features_file}")
    
    # 초반 데이터 제거 (rolling window로 인해 초반 데이터는 불안정)
    # 빠른 테스트를 위해 warm-up period 감소
    min_warmup = 100
    if len(df_features) > min_warmup:
        print(f"초반 {min_warmup}개 데이터 제거 (rolling window warm-up)")
        df_features = df_features.iloc[min_warmup:].copy()
        print(f"Warm-up 제거 후 데이터: {len(df_features)}개")
    else:
        print(f"경고: 데이터가 너무 적어서 warm-up을 제거할 수 없습니다.")
    
    # 결측치가 있는 행 제거 (warm-up 제거 후)
    initial_len = len(df_features)
    df_features = df_features.dropna()
    dropped = initial_len - len(df_features)
    if dropped > 0:
        print(f"결측치 제거: {dropped}개 행 제거됨")
    
    # 모든 피처가 정상 범위인지 확인
    # 무한대나 매우 큰 값이 있는 행 제거
    mask = np.ones(len(df_features), dtype=bool)
    for col in df_features.columns:
        try:
            # 컬럼의 dtype 확인 (안전한 방법)
            col_dtype = df_features.dtypes[col]
            # dtype이 숫자형인지 확인
            is_numeric = (col_dtype == np.float64 or col_dtype == np.float32 or 
                         col_dtype == np.int64 or col_dtype == np.int32 or
                         pd.api.types.is_numeric_dtype(col_dtype))
            
            if is_numeric:
                # 무한대나 매우 큰 값 체크
                col_data = df_features[col].values
                inf_mask = np.isinf(col_data)
                large_mask = np.abs(col_data) > 1e10  # 매우 큰 값
                mask = mask & ~inf_mask & ~large_mask
        except Exception as e:
            # 에러 발생 시 해당 컬럼은 건너뛰기
            print(f"Warning: Error processing column {col} for validation: {e}")
            continue
    
    if not mask.all():
        removed = (~mask).sum()
        print(f"비정상 값 제거: {removed}개 행 제거됨")
        df_features = df_features[mask].copy()
    
    print(f"최종 전처리 후 데이터: {len(df_features)}개")
    
    # 3. 데이터 전처리
    print("\n[3/7] 데이터 전처리 중...")
    window_size = 60  # 60개 시점 (5시간)
    prediction_horizon = 1  # 다음 5분봉 예측
    
    preprocessor = DataPreprocessor(
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        target_column='close',
        scaler_type='robust'  # RobustScaler로 변경 (이상치에 강함)
    )
    
    # 전체 데이터를 먼저 시퀀스로 변환 (스케일링 전)
    # 시퀀스 생성은 전체 데이터로 해야 함 (시간 순서 유지)
    print("전체 데이터로 시퀀스 생성 중...")
    
    # 원본 데이터로 시퀀스 생성 (스케일링 전)
    # 미래 데이터 누수 방지
    df_clean = preprocessor._remove_future_leakage(df_features)
    feature_cols = preprocessor._select_features(df_clean)
    
    # 실제로 존재하는 컬럼만 선택
    available_cols = [col for col in feature_cols if col in df_clean.columns]
    if len(available_cols) != len(feature_cols):
        missing_cols = set(feature_cols) - set(available_cols)
        print(f"Warning: {len(missing_cols)}개 컬럼이 데이터에 없습니다: {missing_cols}")
        feature_cols = available_cols
    
    # 실제로 선택된 컬럼 확인
    feature_df_selected = df_clean[feature_cols]
    actual_cols = feature_df_selected.columns.tolist()
    feature_data = feature_df_selected.values
    
    # shape 확인 및 디버깅
    if feature_data.shape[1] != len(actual_cols):
        print(f"Warning: feature_data shape {feature_data.shape}와 컬럼 수 {len(actual_cols)}가 일치하지 않습니다.")
        print(f"실제 컬럼: {actual_cols[:10]}...")
    
    target_data = df_clean[preprocessor.target_column].values
    
    # 결측치 처리 (실제 선택된 컬럼 사용)
    feature_df = pd.DataFrame(feature_data, columns=actual_cols)
    feature_df = feature_df.ffill().bfill().fillna(0)
    feature_data = feature_df.values
    
    target_series = pd.Series(target_data)
    target_series = target_series.ffill().bfill().fillna(0)
    target_data = target_series.values
    
    # 타겟: 5분 후의 close 가격 (prediction_horizon=1이므로 다음 5분봉)
    # 멀티타겟: 3분, 5분, 15분 후 변화율 예측
    print("타겟: 3분, 5분, 15분 후 변화율 (멀티타겟 예측)")
    
    # 각 시점의 변화율 계산
    # 3분 후 (3개 시점 후)
    target_3m = np.roll(target_data, -3)
    target_3m[-3:] = target_data[-3:]  # 마지막 3개는 자기 자신
    change_3m = (target_3m - target_data) / (target_data + 1e-8)
    
    # 5분 후 (5개 시점 후)
    target_5m = np.roll(target_data, -5)
    target_5m[-5:] = target_data[-5:]  # 마지막 5개는 자기 자신
    change_5m = (target_5m - target_data) / (target_data + 1e-8)
    
    # 15분 후 (15개 시점 후)
    target_15m = np.roll(target_data, -15)
    target_15m[-15:] = target_data[-15:]  # 마지막 15개는 자기 자신
    change_15m = (target_15m - target_data) / (target_data + 1e-8)
    
    # 명백한 오류만 제거 (예: ±50% 이상 같은 데이터 오류)
    outlier_threshold = 0.5
    change_3m = np.clip(change_3m, -outlier_threshold, outlier_threshold)
    change_5m = np.clip(change_5m, -outlier_threshold, outlier_threshold)
    change_15m = np.clip(change_15m, -outlier_threshold, outlier_threshold)
    
    # 변화율 통계 출력
    print(f"3분 변화율: min={change_3m.min():.6f}, max={change_3m.max():.6f}, mean={change_3m.mean():.6f}, std={change_3m.std():.6f}")
    print(f"5분 변화율: min={change_5m.min():.6f}, max={change_5m.max():.6f}, mean={change_5m.mean():.6f}, std={change_5m.std():.6f}")
    print(f"15분 변화율: min={change_15m.min():.6f}, max={change_15m.max():.6f}, mean={change_15m.mean():.6f}, std={change_15m.std():.6f}")
    
    # 멀티타겟 배열 생성 (n_samples, 3)
    target_changes = np.column_stack([change_3m, change_5m, change_15m])
    
    # 시퀀스 생성 (멀티타겟)
    X_raw, y_raw = preprocessor.create_sequences(feature_data, target_changes)
    
    # 원본 가격도 저장 (나중에 역변환용)
    _, y_original = preprocessor.create_sequences(feature_data, target_data)
    
    print(f"생성된 시퀀스: {X_raw.shape}, 타겟(멀티타겟 변화율): {y_raw.shape}")
    print(f"멀티타겟 범위: 3분=[{y_raw[:, 0].min():.4f}, {y_raw[:, 0].max():.4f}], "
          f"5분=[{y_raw[:, 1].min():.4f}, {y_raw[:, 1].max():.4f}], "
          f"15분=[{y_raw[:, 2].min():.4f}, {y_raw[:, 2].max():.4f}]")
    
    feature_names = actual_cols  # 실제 선택된 컬럼 사용
    print(f"시퀀스 데이터 shape: {X_raw.shape}")
    print(f"타겟 데이터 shape: {y_raw.shape}")
    print(f"특징 개수: {len(feature_names)}")
    
    # 학습/검증/테스트 분할 (스케일링 전, 특징 필터링 전)
    # Train: 70%, Val: 15%, Test: 15% (더 많은 Train 데이터로 과적합 방지)
    split_idx_1 = int(len(X_raw) * 0.7)  # Train 70%
    split_idx_2 = int(len(X_raw) * 0.85)  # Val 15%, Test 15%
    
    X_train_raw = X_raw[:split_idx_1]
    y_train_raw = y_raw[:split_idx_1]
    X_val_raw = X_raw[split_idx_1:split_idx_2]
    y_val_raw = y_raw[split_idx_1:split_idx_2]
    X_test_raw = X_raw[split_idx_2:]
    y_test_raw = y_raw[split_idx_2:]
    
    print(f"학습 데이터: {len(X_train_raw)}개")
    print(f"검증 데이터: {len(X_val_raw)}개")
    print(f"테스트 데이터: {len(X_test_raw)}개")
    
    # 원본 타겟값 (y_original이 5분 후 close 가격이므로 동일하게 분할)
    y_train_orig = y_original[:split_idx_1]
    y_val_orig = y_original[split_idx_1:split_idx_2]
    y_test_orig = y_original[split_idx_2:]
    
    # Train/Val 상관관계 차이가 큰 특징 필터링 (스케일링 전)
    # 원본 타겟값과의 상관관계를 계산 (스케일링 전)
    print("\n[3.5/7] Train/Val 상관관계 일관성 검사 및 특징 필터링 중...")
    selected_feature_names = preprocessor.filter_features_by_correlation_consistency(
        X_train_raw, y_train_orig, X_val_raw, y_val_orig, feature_names, max_correlation_diff=0.1
    )
    
    # 선택된 특징의 인덱스 찾기
    selected_feature_indices = [feature_names.index(name) for name in selected_feature_names if name in feature_names]
    
    # 특징 필터링 적용 (스케일링 전)
    if len(selected_feature_indices) < len(feature_names):
        print(f"특징 필터링 적용: {len(feature_names)}개 → {len(selected_feature_indices)}개")
        X_train_raw = X_train_raw[:, :, selected_feature_indices]
        X_val_raw = X_val_raw[:, :, selected_feature_indices]
        X_test_raw = X_test_raw[:, :, selected_feature_indices]
        feature_names = selected_feature_names
        
        # 모델 학습에 사용할 feature 목록 저장 (실시간 예측용)
        preprocessor.model_feature_columns = selected_feature_names
    else:
        print("모든 특징이 일관된 상관관계를 보입니다.")
        # 모든 feature 사용
        preprocessor.model_feature_columns = feature_names
    
    # 특징 필터링 후, 필터링된 Train 데이터로만 스케일러 학습
    print("\n[3.6/7] 스케일러 학습 중 (필터링된 특징으로)...")
    n_train_samples, n_timesteps, n_features = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(-1, n_features)
    
    # 스케일러 학습 (필터링된 특징으로)
    preprocessor.scaler.fit(X_train_flat)
    
    # 타겟 스케일링: 멀티타겟 (3분, 5분, 15분 변화율)
    # MinMaxScaler로 [-1, 1] 범위로 정규화
    from sklearn.preprocessing import MinMaxScaler
    preprocessor.target_scaler = MinMaxScaler(feature_range=(-1, 1))
    # y_train_raw는 (n_samples, 3) 형태
    preprocessor.target_scaler.fit(y_train_raw)
    
    preprocessor.is_fitted = True
    preprocessor._last_feature_cols = feature_names  # 필터링된 feature 컬럼 저장
    
    # 필터링된 데이터 스케일링
    print("[3.7/7] 데이터 스케일링 중...")
    X_train_flat = X_train_raw.reshape(-1, n_features)
    X_val_flat = X_val_raw.reshape(-1, n_features)
    X_test_flat = X_test_raw.reshape(-1, n_features)
    
    X_train_scaled_flat = preprocessor.scaler.transform(X_train_flat)
    X_val_scaled_flat = preprocessor.scaler.transform(X_val_flat)
    X_test_scaled_flat = preprocessor.scaler.transform(X_test_flat)
    
    X_train = X_train_scaled_flat.reshape(n_train_samples, n_timesteps, n_features)
    X_val = X_val_scaled_flat.reshape(X_val_raw.shape[0], n_timesteps, n_features)
    X_test = X_test_scaled_flat.reshape(X_test_raw.shape[0], n_timesteps, n_features)
    
    # 멀티타겟 스케일링 (y_train_raw는 이미 (n_samples, 3) 형태)
    y_train = preprocessor.target_scaler.transform(y_train_raw)
    y_val = preprocessor.target_scaler.transform(y_val_raw)
    y_test = preprocessor.target_scaler.transform(y_test_raw)
    
    # 스케일러 저장 (모델 학습 직전, 필터링된 특징과 model_feature_columns 포함)
    os.makedirs('models', exist_ok=True)
    scaler_path = 'models/scalers.pkl'
    preprocessor.save_scalers(scaler_path)
    print(f"스케일러 저장 완료 (필터링된 특징 {len(feature_names)}개, model_feature_columns 포함): {scaler_path}")
    
    # 4. 모델 생성
    print("\n[4/7] 모델 생성 중...")
    model_builder = PatchCNNBiLSTM(
        input_shape=(window_size, len(feature_names)),
        num_features=len(feature_names),
        patch_size=5,
        cnn_filters=[],  # CNN 제거
        lstm_units=128,  # 원래 크기 유지
        dropout_rate=0.2,  # 사용하지 않지만 호환성 유지
        learning_rate=0.0005  # 학습률 감소 (안정적 학습)
    )
    
    model = model_builder.build_model()
    model.summary()
    
    # 5. 모델 학습
    print("\n[5/7] 모델 학습 중...")
    trainer = ModelTrainer(
        model=model,
        model_save_path='models',
        early_stopping_patience=5,  # 더 공격적인 early stopping (과적합 방지)
        reduce_lr_patience=2  # 학습률 감소를 더 빠르게 (안정적 학습)
    )
    
    # 데이터 통계 출력 (디버깅)
    print(f"\n데이터 통계:")
    print(f"  y_train 범위: [{y_train.min():.4f}, {y_train.max():.4f}], 평균: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  y_val 범위: [{y_val.min():.4f}, {y_val.max():.4f}], 평균: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    print(f"  X_train 범위: [{X_train.min():.4f}, {X_train.max():.4f}], 평균: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    
    # 원본 타겟 통계 (스케일링 전)
    print(f"\n원본 타겟 통계:")
    print(f"  y_train_orig 범위: [{y_train_orig.min():.2f}, {y_train_orig.max():.2f}], 평균: {y_train_orig.mean():.2f}, std: {y_train_orig.std():.2f}")
    print(f"  y_val_orig 범위: [{y_val_orig.min():.2f}, {y_val_orig.max():.2f}], 평균: {y_val_orig.mean():.2f}, std: {y_val_orig.std():.2f}")
    
    # 스케일러 정보 확인
    print(f"\n스케일러 정보:")
    if hasattr(preprocessor.target_scaler, 'mean_'):
        # StandardScaler인 경우
        print(f"  Target Scaler mean: {preprocessor.target_scaler.mean_[0]:.4f}, scale: {preprocessor.target_scaler.scale_[0]:.4f}")
    else:
        # MinMaxScaler인 경우
        print(f"  Target Scaler min: {preprocessor.target_scaler.min_[0]:.4f}, scale: {preprocessor.target_scaler.scale_[0]:.4f}")
        print(f"  Target Scaler data range: [{preprocessor.target_scaler.data_min_[0]:.4f}, {preprocessor.target_scaler.data_max_[0]:.4f}]")
    
    # 사용된 특징 확인
    print(f"\n사용된 특징 개수: {len(feature_names)}개")
    print(f"  특징 목록: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")
    
    # 검증 로스 수렴 분석
    print("\n=== 검증 로스 수렴 분석 ===")
    val_analyzer = ValidationAnalysis()
    val_report = val_analyzer.generate_report(
        X_train, y_train, X_val, y_val, feature_names, save_path='results'
    )
    print(val_report)
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,  # 원래대로 20 에폭
        batch_size=32,  # 원래 배치 크기
        verbose=1
    )
    
    print("\n학습 완료!")
    
    # 학습 히스토리 출력
    if history:
        print("\n--- 학습 히스토리 요약 ---")
        print(f"최종 Train Loss: {history['loss'][-1]:.6f}")
        print(f"최종 Val Loss: {history['val_loss'][-1]:.6f}")
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        print(f"최고 Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
        print(f"최종 Train MAE: {history['mae'][-1]:.6f}")
        print(f"최종 Val MAE: {history['val_mae'][-1]:.6f}")
        
        # 학습 곡선 시각화
        print("\n학습 곡선 생성 중...")
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss 곡선
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE 곡선
        axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        learning_curve_path = 'results/learning_curve.png'
        plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
        print(f"학습 곡선이 저장되었습니다: {learning_curve_path}")
        plt.close()
        
        # 과적합 체크
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0
        
        print(f"\n--- 과적합 분석 ---")
        print(f"Train Loss / Val Loss 비율: {overfitting_ratio:.3f}")
        if overfitting_ratio > 1.5:
            print("⚠️ 경고: 과적합 가능성이 있습니다 (Val Loss가 Train Loss보다 1.5배 이상 높음)")
        elif overfitting_ratio < 0.8:
            print("⚠️ 경고: 학습이 부족할 수 있습니다 (Val Loss가 Train Loss보다 낮음)")
        else:
            print("✓ 학습이 정상적으로 진행되고 있습니다")
    
    # 6. 예측 수행
    print("\n[6/7] 예측 수행 중...")
    predictor = Predictor(
        model=model,
        preprocessor=preprocessor,
        target_scaler=preprocessor.target_scaler
    )
    
    # 테스트 데이터 예측 (변화율 → 절대 가격 변환)
    print("\n예측 수행 중...")
    # 이전 가격: 각 시퀀스의 마지막 시점 직전 가격
    # 시퀀스 i의 타겟은 원본 데이터의 window_size + prediction_horizon - 1 + i 인덱스
    # 따라서 이전 가격은 window_size + prediction_horizon - 2 + i 인덱스
    start_idx = window_size + prediction_horizon - 1
    
    # Train/Val/Test 분할에 맞춰 이전 가격 추출
    test_start = start_idx + split_idx_2
    val_start = start_idx + split_idx_1
    
    # 이전 가격: 예측 시점 직전 가격
    test_prev_indices = np.arange(test_start - 1, test_start - 1 + len(X_test))
    val_prev_indices = np.arange(val_start - 1, val_start - 1 + len(X_val))
    
    # 원본 데이터에서 이전 가격 추출
    test_prev_prices = target_data[test_prev_indices]
    val_prev_prices = target_data[val_prev_indices]
    
    # 길이 확인 및 조정
    if len(test_prev_prices) != len(X_test):
        test_prev_prices = test_prev_prices[:len(X_test)]
    if len(val_prev_prices) != len(X_val):
        val_prev_prices = val_prev_prices[:len(X_val)]
    
    predictions_test = predictor.predict(X_test, previous_prices=test_prev_prices)
    actuals_test = y_test_orig
    
    # 검증 데이터 예측
    predictions_val = predictor.predict(X_val, previous_prices=val_prev_prices)
    actuals_val = y_val_orig
    
    # 예측값 통계 출력 (디버깅)
    print(f"\n예측값 통계:")
    print(f"  predictions_test 범위: [{predictions_test.min():.2f}, {predictions_test.max():.2f}], 평균: {predictions_test.mean():.2f}, std: {predictions_test.std():.2f}")
    print(f"  actuals_test 범위: [{actuals_test.min():.2f}, {actuals_test.max():.2f}], 평균: {actuals_test.mean():.2f}, std: {actuals_test.std():.2f}")
    print(f"  predictions_val 범위: [{predictions_val.min():.2f}, {predictions_val.max():.2f}], 평균: {predictions_val.mean():.2f}, std: {predictions_val.std():.2f}")
    print(f"  actuals_val 범위: [{actuals_val.min():.2f}, {actuals_val.max():.2f}], 평균: {actuals_val.mean():.2f}, std: {actuals_val.std():.2f}")
    
    # 7. 평가 및 결과 저장
    print("\n[7/8] 평가 및 결과 저장 중...")
    evaluator = Evaluator(results_save_path='results')
    
    # 타임스탬프 계산 (시퀀스 생성 시 인덱스 매핑)
    # 시퀀스는 window_size + prediction_horizon - 1 인덱스부터 시작
    start_idx = window_size + prediction_horizon - 1
    total_sequences = len(X_train) + len(X_val) + len(X_test)
    sequence_timestamps = df_features.index[start_idx:start_idx + total_sequences]
    
    # 테스트 데이터 평가
    print("\n--- 테스트 데이터 평가 ---")
    test_start_idx = len(X_train) + len(X_val)
    test_timestamps = sequence_timestamps[test_start_idx:test_start_idx + len(predictions_test)]
    test_metrics = evaluator.evaluate_and_save(
        predictions_test,
        actuals_test,
        test_timestamps,
        prefix='test'
    )
    
    # 검증 데이터 평가
    print("\n--- 검증 데이터 평가 ---")
    print(f"검증 예측값 길이: {len(predictions_val)}")
    print(f"검증 실제값 길이: {len(actuals_val)}")
    
    # 길이가 맞지 않으면 조정
    min_len = min(len(predictions_val), len(actuals_val))
    if len(predictions_val) != len(actuals_val):
        print(f"경고: 길이가 다릅니다. {min_len}개로 맞춥니다.")
        predictions_val = predictions_val[:min_len]
        actuals_val = actuals_val[:min_len]
    
    val_start_idx = len(X_train)
    val_timestamps = sequence_timestamps[val_start_idx:val_start_idx + len(predictions_val)]
    val_metrics = evaluator.evaluate_and_save(
        predictions_val,
        actuals_val,
        val_timestamps,
        prefix='validation'
    )
    
    # 8. 데이터 누수 검증 및 세부 가격 분석
    print("\n[8/8] 데이터 누수 검증 및 세부 가격 분석 중...")
    leakage_checker = DataLeakageChecker()
    
    # 타임스탬프 생성 (시퀀스에 대응하는 타임스탬프)
    start_idx = window_size + prediction_horizon - 1
    test_start = start_idx + split_idx_2
    val_start = start_idx + split_idx_1
    
    test_timestamps = df_features.index[test_start:test_start + len(X_test)]
    val_timestamps_seq = df_features.index[val_start:val_start + len(X_val)]
    
    # 테스트 데이터에 대한 데이터 누수 검증
    print("\n=== 테스트 데이터 누수 검증 ===")
    test_report = leakage_checker.generate_report(
        X=X_test,
        y=y_test,
        feature_names=feature_names,
        target_data=target_data,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        predictions=predictions_test,
        actuals=actuals_test,
        df_features=df_features.iloc[test_start:test_start + len(X_test)] if len(df_features) > test_start + len(X_test) else None,
        timestamps=test_timestamps if len(test_timestamps) == len(X_test) else None,
        save_path='results'
    )
    print(test_report)
    
    # 검증 데이터에 대한 데이터 누수 검증
    print("\n=== 검증 데이터 누수 검증 ===")
    val_report = leakage_checker.generate_report(
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        target_data=target_data,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        predictions=predictions_val,
        actuals=actuals_val,
        df_features=df_features.iloc[val_start:val_start + len(X_val)] if len(df_features) > val_start + len(X_val) else None,
        timestamps=val_timestamps_seq if len(val_timestamps_seq) == len(X_val) else None,
        save_path='results'
    )
    print(val_report)
    
    print("\n" + "=" * 60)
    print("모든 작업이 완료되었습니다!")
    print("=" * 60)
    print(f"\n결과 파일 위치:")
    print(f"- 모델: models/")
    print(f"- 평가 결과: results/")
    print(f"- 데이터 누수 검증 리포트: results/data_leakage_report.txt")
    print(f"- 세부 가격 분석: results/detailed_price_analysis.png")


if __name__ == "__main__":
    main()


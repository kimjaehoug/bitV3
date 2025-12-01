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
    # 멀티타겟: 30분, 1시간 후 변화율 예측
    print("타겟: 30분, 1시간 후 변화율 (멀티타겟 예측)")
    
    # 각 시점의 변화율 계산
    # 30분 후 (6개 시점 후, 30분 / 5분 = 6)
    # 직접 인덱싱 사용 (np.roll 대신)
    change_30m = np.zeros_like(target_data)
    for i in range(len(target_data) - 6):
        change_30m[i] = (target_data[i + 6] - target_data[i]) / (target_data[i] + 1e-8)
    # 마지막 6개는 0으로 설정 (예측 불가)
    change_30m[-6:] = 0
    
    # 1시간 후 (12개 시점 후, 60분 / 5분 = 12)
    change_1h = np.zeros_like(target_data)
    for i in range(len(target_data) - 12):
        change_1h[i] = (target_data[i + 12] - target_data[i]) / (target_data[i] + 1e-8)
    # 마지막 12개는 0으로 설정 (예측 불가)
    change_1h[-12:] = 0
    
    # 명백한 오류만 제거 (예: ±50% 이상 같은 데이터 오류)
    outlier_threshold = 0.5
    change_30m = np.clip(change_30m, -outlier_threshold, outlier_threshold)
    change_1h = np.clip(change_1h, -outlier_threshold, outlier_threshold)
    
    # 변화율 통계 출력 (스케일링 전)
    print(f"\n=== 변화율 통계 (스케일링 전) ===")
    print(f"30분 변화율: min={change_30m.min():.6f}, max={change_30m.max():.6f}, mean={change_30m.mean():.6f}, std={change_30m.std():.6f}")
    print(f"1시간 변화율: min={change_1h.min():.6f}, max={change_1h.max():.6f}, mean={change_1h.mean():.6f}, std={change_1h.std():.6f}")
    print(f"30분 변화율 절댓값 평균: {np.abs(change_30m).mean():.6f}")
    print(f"1시간 변화율 절댓값 평균: {np.abs(change_1h).mean():.6f}")
    
    # 멀티타겟 배열 생성 (n_samples, 2)
    target_changes = np.column_stack([change_30m, change_1h])
    
    # 시퀀스 생성 (멀티타겟)
    X_raw, y_raw = preprocessor.create_sequences(feature_data, target_changes)
    
    # 원본 가격도 저장 (나중에 역변환용)
    _, y_original = preprocessor.create_sequences(feature_data, target_data)
    
    print(f"생성된 시퀀스: {X_raw.shape}, 타겟(멀티타겟 변화율): {y_raw.shape}")
    print(f"멀티타겟 범위: 30분=[{y_raw[:, 0].min():.4f}, {y_raw[:, 0].max():.4f}], "
          f"1시간=[{y_raw[:, 1].min():.4f}, {y_raw[:, 1].max():.4f}]")
    
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
    
    # 타겟 스케일링: 멀티타겟 (30분, 1시간 변화율)
    # MinMaxScaler로 [-1, 1] 범위로 정규화
    from sklearn.preprocessing import MinMaxScaler
    preprocessor.target_scaler = MinMaxScaler(feature_range=(-1, 1))
    # y_train_raw는 (n_samples, 2) 형태
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
    
    # 멀티타겟 스케일링 (y_train_raw는 이미 (n_samples, 2) 형태)
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
    print(f"\n=== 타겟 스케일러 정보 (30분/1시간 변화율용) ===")
    if hasattr(preprocessor.target_scaler, 'mean_'):
        # StandardScaler인 경우
        print(f"  30분 타겟 - mean: {preprocessor.target_scaler.mean_[0]:.4f}, scale: {preprocessor.target_scaler.scale_[0]:.4f}")
        print(f"  1시간 타겟 - mean: {preprocessor.target_scaler.mean_[1]:.4f}, scale: {preprocessor.target_scaler.scale_[1]:.4f}")
    else:
        # MinMaxScaler인 경우 (feature_range=(-1, 1))
        print(f"  30분 타겟 - 원본 데이터 범위: [{preprocessor.target_scaler.data_min_[0]:.6f}, {preprocessor.target_scaler.data_max_[0]:.6f}]")
        print(f"  30분 타겟 - 스케일링 후 범위: [-1.0, 1.0]")
        print(f"  1시간 타겟 - 원본 데이터 범위: [{preprocessor.target_scaler.data_min_[1]:.6f}, {preprocessor.target_scaler.data_max_[1]:.6f}]")
        print(f"  1시간 타겟 - 스케일링 후 범위: [-1.0, 1.0]")
    
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
    
    # 멀티타겟 예측 (30분, 1시간 모두)
    y_pred_changes_test = predictor.predict_multi(X_test)
    y_pred_changes_val = predictor.predict_multi(X_val)
    
    # 실제 타겟값도 멀티타겟 형태로 변환 (변화율 → 절대 가격)
    # y_test_orig, y_val_orig는 다음 5분봉 가격이므로, 30분/1시간 후 가격을 계산해야 함
    # 실제 30분 후 가격 = target_data[test_start + 6], 1시간 후 = target_data[test_start + 12]
    test_30m_indices = np.arange(test_start + 5, test_start + 5 + len(X_test))  # 6개 시점 후
    test_1h_indices = np.arange(test_start + 11, test_start + 11 + len(X_test))  # 12개 시점 후
    val_30m_indices = np.arange(val_start + 5, val_start + 5 + len(X_val))
    val_1h_indices = np.arange(val_start + 11, val_start + 11 + len(X_val))
    
    # 인덱스 범위 확인
    test_30m_indices = test_30m_indices[test_30m_indices < len(target_data)]
    test_1h_indices = test_1h_indices[test_1h_indices < len(target_data)]
    val_30m_indices = val_30m_indices[val_30m_indices < len(target_data)]
    val_1h_indices = val_1h_indices[val_1h_indices < len(target_data)]
    
    # 실제 30분/1시간 후 가격
    actuals_test_30m = target_data[test_30m_indices[:len(X_test)]]
    actuals_test_1h = target_data[test_1h_indices[:len(X_test)]]
    actuals_val_30m = target_data[val_30m_indices[:len(X_val)]]
    actuals_val_1h = target_data[val_1h_indices[:len(X_val)]]
    
    # 예측 가격 계산 (변화율 → 절대 가격)
    predictions_test_30m = test_prev_prices[:len(y_pred_changes_test)] * (1 + y_pred_changes_test[:, 0])
    predictions_test_1h = test_prev_prices[:len(y_pred_changes_test)] * (1 + y_pred_changes_test[:, 1])
    predictions_val_30m = val_prev_prices[:len(y_pred_changes_val)] * (1 + y_pred_changes_val[:, 0])
    predictions_val_1h = val_prev_prices[:len(y_pred_changes_val)] * (1 + y_pred_changes_val[:, 1])
    
    # 길이 맞추기
    min_len_test_30m = min(len(predictions_test_30m), len(actuals_test_30m))
    min_len_test_1h = min(len(predictions_test_1h), len(actuals_test_1h))
    min_len_val_30m = min(len(predictions_val_30m), len(actuals_val_30m))
    min_len_val_1h = min(len(predictions_val_1h), len(actuals_val_1h))
    
    predictions_test_30m = predictions_test_30m[:min_len_test_30m]
    actuals_test_30m = actuals_test_30m[:min_len_test_30m]
    predictions_test_1h = predictions_test_1h[:min_len_test_1h]
    actuals_test_1h = actuals_test_1h[:min_len_test_1h]
    predictions_val_30m = predictions_val_30m[:min_len_val_30m]
    actuals_val_30m = actuals_val_30m[:min_len_val_30m]
    predictions_val_1h = predictions_val_1h[:min_len_val_1h]
    actuals_val_1h = actuals_val_1h[:min_len_val_1h]
    
    # 예측값 통계 출력 (디버깅)
    print(f"\n=== 30분 예측 통계 ===")
    print(f"  Test - 예측 범위: [{predictions_test_30m.min():.2f}, {predictions_test_30m.max():.2f}], 평균: {predictions_test_30m.mean():.2f}")
    print(f"  Test - 실제 범위: [{actuals_test_30m.min():.2f}, {actuals_test_30m.max():.2f}], 평균: {actuals_test_30m.mean():.2f}")
    print(f"  Val - 예측 범위: [{predictions_val_30m.min():.2f}, {predictions_val_30m.max():.2f}], 평균: {predictions_val_30m.mean():.2f}")
    print(f"  Val - 실제 범위: [{actuals_val_30m.min():.2f}, {actuals_val_30m.max():.2f}], 평균: {actuals_val_30m.mean():.2f}")
    
    print(f"\n=== 1시간 예측 통계 ===")
    print(f"  Test - 예측 범위: [{predictions_test_1h.min():.2f}, {predictions_test_1h.max():.2f}], 평균: {predictions_test_1h.mean():.2f}")
    print(f"  Test - 실제 범위: [{actuals_test_1h.min():.2f}, {actuals_test_1h.max():.2f}], 평균: {actuals_test_1h.mean():.2f}")
    print(f"  Val - 예측 범위: [{predictions_val_1h.min():.2f}, {predictions_val_1h.max():.2f}], 평균: {predictions_val_1h.mean():.2f}")
    print(f"  Val - 실제 범위: [{actuals_val_1h.min():.2f}, {actuals_val_1h.max():.2f}], 평균: {actuals_val_1h.mean():.2f}")
    
    # 7. 평가 및 결과 저장
    print("\n[7/8] 평가 및 결과 저장 중...")
    evaluator = Evaluator(results_save_path='results')
    
    # 타임스탬프 계산 (시퀀스 생성 시 인덱스 매핑)
    # 시퀀스는 window_size + prediction_horizon - 1 인덱스부터 시작
    start_idx = window_size + prediction_horizon - 1
    total_sequences = len(X_train) + len(X_val) + len(X_test)
    sequence_timestamps = df_features.index[start_idx:start_idx + total_sequences]
    
    # 30분 예측 평가
    print("\n" + "="*60)
    print("30분 예측 평가")
    print("="*60)
    
    # 테스트 데이터 평가 (30분)
    print("\n--- 테스트 데이터 평가 (30분) ---")
    test_start_idx = len(X_train) + len(X_val)
    test_timestamps_30m = sequence_timestamps[test_start_idx:test_start_idx + len(predictions_test_30m)]
    test_metrics_30m = evaluator.evaluate_and_save(
        predictions_test_30m,
        actuals_test_30m,
        test_timestamps_30m,
        prefix='test_30m'
    )
    
    # 검증 데이터 평가 (30분)
    print("\n--- 검증 데이터 평가 (30분) ---")
    val_start_idx = len(X_train)
    val_timestamps_30m = sequence_timestamps[val_start_idx:val_start_idx + len(predictions_val_30m)]
    val_metrics_30m = evaluator.evaluate_and_save(
        predictions_val_30m,
        actuals_val_30m,
        val_timestamps_30m,
        prefix='validation_30m'
    )
    
    # 1시간 예측 평가
    print("\n" + "="*60)
    print("1시간 예측 평가")
    print("="*60)
    
    # 테스트 데이터 평가 (1시간)
    print("\n--- 테스트 데이터 평가 (1시간) ---")
    test_timestamps_1h = sequence_timestamps[test_start_idx:test_start_idx + len(predictions_test_1h)]
    test_metrics_1h = evaluator.evaluate_and_save(
        predictions_test_1h,
        actuals_test_1h,
        test_timestamps_1h,
        prefix='test_1h'
    )
    
    # 검증 데이터 평가 (1시간)
    print("\n--- 검증 데이터 평가 (1시간) ---")
    val_timestamps_1h = sequence_timestamps[val_start_idx:val_start_idx + len(predictions_val_1h)]
    val_metrics_1h = evaluator.evaluate_and_save(
        predictions_val_1h,
        actuals_val_1h,
        val_timestamps_1h,
        prefix='validation_1h'
    )
    
    # 8. 데이터 누수 검증 및 세부 가격 분석
    print("\n[8/8] 데이터 누수 검증 및 세부 가격 분석 중...")
    leakage_checker = DataLeakageChecker()
    
    # 타임스탬프 생성 (시퀀스에 대응하는 타임스탬프)
    start_idx = window_size + prediction_horizon - 1
    test_start = start_idx + split_idx_2
    val_start = start_idx + split_idx_1
    
    # 30분 예측에 대한 데이터 누수 검증
    print("\n=== 30분 예측 데이터 누수 검증 ===")
    test_timestamps_30m = df_features.index[test_start:test_start + len(predictions_test_30m)]
    val_timestamps_30m = df_features.index[val_start:val_start + len(predictions_val_30m)]
    
    # 30분 예측 타겟 (y_test의 첫 번째 컬럼)
    y_test_30m = y_test[:, 0] if y_test.ndim == 2 else y_test
    y_val_30m = y_val[:, 0] if y_val.ndim == 2 else y_val
    
    test_report_30m = leakage_checker.generate_report(
        X=X_test[:len(predictions_test_30m)],
        y=y_test_30m[:len(predictions_test_30m)],
        feature_names=feature_names,
        target_data=target_data,
        window_size=window_size,
        prediction_horizon=6,  # 30분 = 6개 시점
        predictions=predictions_test_30m,
        actuals=actuals_test_30m,
        df_features=df_features.iloc[test_start:test_start + len(predictions_test_30m)] if len(df_features) > test_start + len(predictions_test_30m) else None,
        timestamps=test_timestamps_30m if len(test_timestamps_30m) == len(predictions_test_30m) else None,
        save_path='results'
    )
    print(test_report_30m)
    
    # 1시간 예측에 대한 데이터 누수 검증
    print("\n=== 1시간 예측 데이터 누수 검증 ===")
    test_timestamps_1h = df_features.index[test_start:test_start + len(predictions_test_1h)]
    val_timestamps_1h = df_features.index[val_start:val_start + len(predictions_val_1h)]
    
    # 1시간 예측 타겟 (y_test의 두 번째 컬럼)
    y_test_1h = y_test[:, 1] if y_test.ndim == 2 else y_test
    y_val_1h = y_val[:, 1] if y_val.ndim == 2 else y_val
    
    test_report_1h = leakage_checker.generate_report(
        X=X_test[:len(predictions_test_1h)],
        y=y_test_1h[:len(predictions_test_1h)],
        feature_names=feature_names,
        target_data=target_data,
        window_size=window_size,
        prediction_horizon=12,  # 1시간 = 12개 시점
        predictions=predictions_test_1h,
        actuals=actuals_test_1h,
        df_features=df_features.iloc[test_start:test_start + len(predictions_test_1h)] if len(df_features) > test_start + len(predictions_test_1h) else None,
        timestamps=test_timestamps_1h if len(test_timestamps_1h) == len(predictions_test_1h) else None,
        save_path='results'
    )
    print(test_report_1h)
    
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


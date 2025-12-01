"""
데이터 누수 검증 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os


class DataLeakageChecker:
    """데이터 누수 검증 클래스"""
    
    def __init__(self):
        pass
    
    def check_feature_target_correlation(self, 
                                        X: np.ndarray, 
                                        y: np.ndarray,
                                        feature_names: list,
                                        threshold: float = 0.95) -> Dict:
        """
        특징과 타겟 간의 상관관계 확인
        
        Args:
            X: 특징 데이터 (n_samples, window_size, n_features) 또는 (n_samples, n_features)
            y: 타겟 데이터 (n_samples,)
            feature_names: 특징 이름 리스트
            threshold: 의심스러운 상관관계 임계값
        
        Returns:
            검증 결과 딕셔너리
        """
        results = {
            'high_correlation_features': [],
            'max_correlation': 0.0,
            'suspicious': False
        }
        
        # X를 평탄화 (window_size, n_features) -> (n_samples * window_size, n_features)
        if len(X.shape) == 3:
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            # 각 시퀀스의 마지막 시점만 사용 (예측 시점에 가장 가까운 데이터)
            X_last = X[:, -1, :]  # (n_samples, n_features)
        else:
            X_last = X
        
        # 멀티타겟인지 확인 (y가 2D이고 shape[1] == 2이면 멀티타겟)
        is_multitarget = y.ndim == 2 and y.shape[1] == 2
        
        # 멀티타겟인 경우 1시간 타겟(인덱스 1) 사용
        if is_multitarget:
            y_target = y[:, 1]  # 1시간 타겟
        else:
            y_target = y.flatten() if y.ndim > 1 else y
        
        # 각 특징과 타겟 간의 상관관계 계산
        correlations = []
        for i, feat_name in enumerate(feature_names):
            if i >= X_last.shape[1]:
                continue
            feat_values = X_last[:, i]
            corr = np.corrcoef(feat_values, y_target)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append((feat_name, abs(corr)))
            
            if abs(corr) > threshold:
                results['high_correlation_features'].append({
                    'feature': feat_name,
                    'correlation': corr
                })
        
        if correlations:
            results['max_correlation'] = max([c[1] for c in correlations])
            results['suspicious'] = results['max_correlation'] > threshold
        
        return results, correlations
    
    def check_sequence_target_alignment(self,
                                       target_data: np.ndarray,
                                       window_size: int,
                                       prediction_horizon: int,
                                       sample_indices: list = None) -> Dict:
        """
        시퀀스 생성 시 타겟 인덱스가 정확한지 확인
        
        Args:
            target_data: 원본 타겟 데이터
            window_size: 윈도우 크기
            prediction_horizon: 예측 지평선
            sample_indices: 확인할 샘플 인덱스 리스트
        
        Returns:
            검증 결과
        """
        results = {
            'correct': True,
            'errors': []
        }
        
        if sample_indices is None:
            sample_indices = [0, 100, 500, 1000] if len(target_data) > 1000 else [0, 10, 50, 100]
        
        for seq_idx in sample_indices:
            if seq_idx >= len(target_data) - window_size - prediction_horizon + 1:
                continue
            
            # 시퀀스 i의 타겟 인덱스
            expected_target_idx = seq_idx + window_size + prediction_horizon - 1
            
            if expected_target_idx >= len(target_data):
                results['errors'].append(f"시퀀스 {seq_idx}: 타겟 인덱스 {expected_target_idx}가 데이터 범위를 벗어남")
                results['correct'] = False
            else:
                # 시퀀스의 마지막 입력 인덱스
                last_input_idx = seq_idx + window_size - 1
                # 타겟은 prediction_horizon 이후여야 함
                if expected_target_idx <= last_input_idx:
                    results['errors'].append(
                        f"시퀀스 {seq_idx}: 타겟 인덱스 {expected_target_idx}가 입력 범위({last_input_idx})와 겹침"
                    )
                    results['correct'] = False
        
        return results
    
    def check_future_data_usage(self, df: pd.DataFrame) -> Dict:
        """
        미래 데이터 사용 여부 확인 (shift(-1) 등)
        
        Args:
            df: 특징 데이터프레임
        
        Returns:
            검증 결과
        """
        results = {
            'suspicious_columns': [],
            'has_future_leakage': False
        }
        
        # 각 컬럼의 값이 미래와 완전히 일치하는지 확인
        for col in df.columns:
            try:
                # 숫자형인지 확인 (안전한 방법)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
            except:
                continue
            
            # 현재 값이 다음 시점의 값과 완전히 일치하는지 확인
            shifted = df[col].shift(-1)
            matches = (df[col][:-1] == shifted[:-1]).sum()
            match_ratio = matches / (len(df) - 1) if len(df) > 1 else 0
            
            if match_ratio > 0.9:  # 90% 이상 일치하면 의심
                results['suspicious_columns'].append({
                    'column': col,
                    'match_ratio': match_ratio
                })
                results['has_future_leakage'] = True
        
        return results
    
    def detailed_price_analysis(self,
                               predictions: np.ndarray,
                               actuals: np.ndarray,
                               timestamps: pd.DatetimeIndex = None,
                               save_path: str = 'results') -> Dict:
        """
        세부 가격 분석 (시간대별, 가격 범위별 오차)
        
        Args:
            predictions: 예측값
            actuals: 실제값
            timestamps: 타임스탬프 (선택사항)
            save_path: 결과 저장 경로
        
        Returns:
            분석 결과
        """
        os.makedirs(save_path, exist_ok=True)
        
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / (actuals + 1e-8) * 100
        
        results = {
            'overall_stats': {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'mean_abs_error': float(np.mean(abs_errors)),
                'mean_pct_error': float(np.mean(pct_errors)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors))
            }
        }
        
        # 1. 가격 범위별 오차 분석
        price_ranges = [
            (0, np.percentile(actuals, 25), 'Low (0-25%)'),
            (np.percentile(actuals, 25), np.percentile(actuals, 50), 'Medium-Low (25-50%)'),
            (np.percentile(actuals, 50), np.percentile(actuals, 75), 'Medium-High (50-75%)'),
            (np.percentile(actuals, 75), np.max(actuals), 'High (75-100%)')
        ]
        
        range_stats = []
        for min_price, max_price, label in price_ranges:
            mask = (actuals >= min_price) & (actuals < max_price) if max_price < np.max(actuals) else (actuals >= min_price)
            if mask.sum() == 0:
                continue
            
            range_errors = abs_errors[mask]
            range_pct_errors = pct_errors[mask]
            
            range_stats.append({
                'range': label,
                'count': int(mask.sum()),
                'mean_abs_error': float(np.mean(range_errors)),
                'mean_pct_error': float(np.mean(range_pct_errors)),
                'std_error': float(np.std(range_errors))
            })
        
        results['price_range_stats'] = range_stats
        
        # 2. 시간대별 오차 분석 (타임스탬프가 있는 경우)
        if timestamps is not None and len(timestamps) == len(actuals):
            hourly_stats = []
            for hour in range(24):
                hour_mask = timestamps.hour == hour
                if hour_mask.sum() == 0:
                    continue
                
                hour_errors = abs_errors[hour_mask]
                hour_pct_errors = pct_errors[hour_mask]
                
                hourly_stats.append({
                    'hour': hour,
                    'count': int(hour_mask.sum()),
                    'mean_abs_error': float(np.mean(hour_errors)),
                    'mean_pct_error': float(np.mean(hour_pct_errors))
                })
            
            results['hourly_stats'] = hourly_stats
        
        # 3. 오차 분포 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 오차 히스토그램
        axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, 0].set_xlabel('Error (Actual - Predicted)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 절대 오차 히스토그램
        axes[0, 1].hist(abs_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(np.mean(abs_errors), color='r', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(abs_errors):.2f}')
        axes[0, 1].set_xlabel('Absolute Error', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 가격 범위별 평균 오차
        if range_stats:
            ranges = [s['range'] for s in range_stats]
            mean_errors = [s['mean_abs_error'] for s in range_stats]
            axes[1, 0].bar(ranges, mean_errors, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=11)
            axes[1, 0].set_title('Error by Price Range', fontsize=12, fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 시간대별 평균 오차 (타임스탬프가 있는 경우)
        if 'hourly_stats' in results and results['hourly_stats']:
            hours = [s['hour'] for s in results['hourly_stats']]
            hour_errors = [s['mean_abs_error'] for s in results['hourly_stats']]
            axes[1, 1].plot(hours, hour_errors, marker='o', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Hour of Day', fontsize=11)
            axes[1, 1].set_ylabel('Mean Absolute Error', fontsize=11)
            axes[1, 1].set_title('Error by Hour of Day', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(0, 24, 2))
        else:
            axes[1, 1].text(0.5, 0.5, 'No timestamp data', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Error by Hour of Day', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        analysis_path = os.path.join(save_path, 'detailed_price_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        print(f"세부 가격 분석 그래프가 저장되었습니다: {analysis_path}")
        plt.close()
        
        return results
    
    def generate_report(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       feature_names: list,
                       target_data: np.ndarray,
                       window_size: int,
                       prediction_horizon: int,
                       predictions: np.ndarray,
                       actuals: np.ndarray,
                       df_features: pd.DataFrame = None,
                       timestamps: pd.DatetimeIndex = None,
                       save_path: str = 'results') -> str:
        """
        종합 데이터 누수 검증 리포트 생성
        
        Returns:
            리포트 텍스트
        """
        os.makedirs(save_path, exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("데이터 누수 검증 리포트")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 특징-타겟 상관관계 검증
        report_lines.append("1. 특징-타겟 상관관계 검증")
        report_lines.append("-" * 80)
        corr_results, correlations = self.check_feature_target_correlation(
            X, y, feature_names, threshold=0.95
        )
        
        if corr_results['suspicious']:
            report_lines.append("⚠️ 경고: 매우 높은 상관관계를 가진 특징이 발견되었습니다!")
            report_lines.append(f"   최대 상관관계: {corr_results['max_correlation']:.4f}")
            report_lines.append("   의심스러운 특징:")
            for feat_info in corr_results['high_correlation_features']:
                report_lines.append(f"     - {feat_info['feature']}: {feat_info['correlation']:.4f}")
        else:
            report_lines.append("✓ 특징-타겟 상관관계가 정상 범위입니다.")
            if correlations:
                top_corr = sorted(correlations, key=lambda x: x[1], reverse=True)[:5]
                report_lines.append("   상위 5개 상관관계:")
                for feat_name, corr in top_corr:
                    report_lines.append(f"     - {feat_name}: {corr:.4f}")
        report_lines.append("")
        
        # 2. 시퀀스-타겟 정렬 검증
        report_lines.append("2. 시퀀스-타겟 정렬 검증")
        report_lines.append("-" * 80)
        alignment_results = self.check_sequence_target_alignment(
            target_data, window_size, prediction_horizon
        )
        if alignment_results['correct']:
            report_lines.append("✓ 시퀀스와 타겟의 정렬이 올바릅니다.")
        else:
            report_lines.append("⚠️ 경고: 시퀀스-타겟 정렬에 문제가 있습니다!")
            for error in alignment_results['errors']:
                report_lines.append(f"   - {error}")
        report_lines.append("")
        
        # 3. 미래 데이터 사용 검증
        if df_features is not None:
            report_lines.append("3. 미래 데이터 사용 검증")
            report_lines.append("-" * 80)
            future_check = self.check_future_data_usage(df_features)
            if future_check['has_future_leakage']:
                report_lines.append("⚠️ 경고: 미래 데이터 사용이 의심됩니다!")
                for col_info in future_check['suspicious_columns']:
                    report_lines.append(
                        f"   - {col_info['column']}: 다음 시점과 {col_info['match_ratio']*100:.1f}% 일치"
                    )
            else:
                report_lines.append("✓ 미래 데이터 사용이 감지되지 않았습니다.")
            report_lines.append("")
        
        # 4. 세부 가격 분석
        report_lines.append("4. 세부 가격 분석")
        report_lines.append("-" * 80)
        price_analysis = self.detailed_price_analysis(
            predictions, actuals, timestamps, save_path
        )
        
        stats = price_analysis['overall_stats']
        report_lines.append(f"전체 통계:")
        report_lines.append(f"  평균 오차: {stats['mean_error']:.2f}")
        report_lines.append(f"  오차 표준편차: {stats['std_error']:.2f}")
        report_lines.append(f"  평균 절대 오차: {stats['mean_abs_error']:.2f}")
        report_lines.append(f"  평균 백분율 오차: {stats['mean_pct_error']:.2f}%")
        report_lines.append(f"  최대 오차: {stats['max_error']:.2f}")
        report_lines.append("")
        
        if 'price_range_stats' in price_analysis:
            report_lines.append("가격 범위별 오차:")
            for range_stat in price_analysis['price_range_stats']:
                report_lines.append(
                    f"  {range_stat['range']}: "
                    f"MAE={range_stat['mean_abs_error']:.2f}, "
                    f"MAPE={range_stat['mean_pct_error']:.2f}%, "
                    f"샘플 수={range_stat['count']}"
                )
            report_lines.append("")
        
        if 'hourly_stats' in price_analysis:
            report_lines.append("시간대별 평균 절대 오차 (상위 5개):")
            hourly_sorted = sorted(price_analysis['hourly_stats'], 
                                 key=lambda x: x['mean_abs_error'], reverse=True)[:5]
            for hour_stat in hourly_sorted:
                report_lines.append(
                    f"  {hour_stat['hour']}시: "
                    f"MAE={hour_stat['mean_abs_error']:.2f}, "
                    f"샘플 수={hour_stat['count']}"
                )
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # 리포트 저장
        report_path = os.path.join(save_path, 'data_leakage_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"데이터 누수 검증 리포트가 저장되었습니다: {report_path}")
        
        return report_text


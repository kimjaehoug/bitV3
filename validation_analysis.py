"""
검증 로스 수렴 분석 모듈
Train/Val 데이터 분포 차이 및 특징 유용성 분석
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os


class ValidationAnalysis:
    """검증 로스 수렴 분석 클래스"""
    
    def __init__(self):
        pass
    
    def analyze_train_val_distribution(self,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     X_val: np.ndarray,
                                     y_val: np.ndarray,
                                     feature_names: list) -> Dict:
        """
        Train/Val 데이터 분포 차이 분석
        
        Returns:
            분석 결과 딕셔너리
        """
        results = {
            'target_stats': {},
            'feature_stats': {},
            'distribution_differences': []
        }
        
        # 멀티타겟인지 확인 (y_train이 2D이고 shape[1] == 2이면 멀티타겟)
        is_multitarget = y_train.ndim == 2 and y_train.shape[1] == 2
        
        # 멀티타겟인 경우 5분 타겟(인덱스 1) 사용
        if is_multitarget:
            y_train_target = y_train[:, 1]  # 1시간 타겟
            y_val_target = y_val[:, 1]  # 1시간 타겟
        else:
            y_train_target = y_train.flatten() if y_train.ndim > 1 else y_train
            y_val_target = y_val.flatten() if y_val.ndim > 1 else y_val
        
        # 타겟 분포 차이
        results['target_stats'] = {
            'train_mean': float(y_train_target.mean()),
            'train_std': float(y_train_target.std()),
            'val_mean': float(y_val_target.mean()),
            'val_std': float(y_val_target.std()),
            'mean_diff': float(abs(y_train_target.mean() - y_val_target.mean())),
            'std_ratio': float(y_val_target.std() / y_train_target.std()) if y_train_target.std() > 0 else 0
        }
        
        # 특징별 분포 차이 (시퀀스의 마지막 시점만 사용)
        if len(X_train.shape) == 3:
            X_train_last = X_train[:, -1, :]  # (n_samples, n_features)
            X_val_last = X_val[:, -1, :]
        else:
            X_train_last = X_train
            X_val_last = X_val
        
        for i, feat_name in enumerate(feature_names):
            if i >= X_train_last.shape[1]:
                continue
            
            train_feat = X_train_last[:, i]
            val_feat = X_val_last[:, i]
            
            train_mean = train_feat.mean()
            train_std = train_feat.std()
            val_mean = val_feat.mean()
            val_std = val_feat.std()
            
            mean_diff = abs(train_mean - val_mean)
            std_ratio = val_std / train_std if train_std > 0 else 0
            
            # 큰 차이가 있는 특징 기록
            if mean_diff > 0.5 or abs(std_ratio - 1.0) > 0.5:
                results['distribution_differences'].append({
                    'feature': feat_name,
                    'train_mean': float(train_mean),
                    'val_mean': float(val_mean),
                    'mean_diff': float(mean_diff),
                    'train_std': float(train_std),
                    'val_std': float(val_std),
                    'std_ratio': float(std_ratio)
                })
        
        return results
    
    def analyze_feature_importance(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  feature_names: list) -> Dict:
        """
        특징 중요도 분석 (Train vs Val)
        
        Returns:
            특징별 상관관계 및 중요도
        """
        results = {
            'train_correlations': [],
            'val_correlations': [],
            'correlation_differences': []
        }
        
        # 시퀀스의 마지막 시점만 사용
        if len(X_train.shape) == 3:
            X_train_last = X_train[:, -1, :]
            X_val_last = X_val[:, -1, :]
        else:
            X_train_last = X_train
            X_val_last = X_val
        
        # 멀티타겟인지 확인 (y_train이 2D이고 shape[1] == 2이면 멀티타겟)
        is_multitarget = y_train.ndim == 2 and y_train.shape[1] == 2
        
        # 멀티타겟인 경우 5분 타겟(인덱스 1) 사용
        if is_multitarget:
            y_train_target = y_train[:, 1]  # 1시간 타겟
            y_val_target = y_val[:, 1]  # 1시간 타겟
        else:
            y_train_target = y_train.flatten() if y_train.ndim > 1 else y_train
            y_val_target = y_val.flatten() if y_val.ndim > 1 else y_val
        
        for i, feat_name in enumerate(feature_names):
            if i >= X_train_last.shape[1]:
                continue
            
            train_feat = X_train_last[:, i]
            val_feat = X_val_last[:, i]
            
            # Train 상관관계
            train_corr = np.corrcoef(train_feat, y_train_target)[0, 1]
            if np.isnan(train_corr):
                train_corr = 0.0
            
            # Val 상관관계
            val_corr = np.corrcoef(val_feat, y_val_target)[0, 1]
            if np.isnan(val_corr):
                val_corr = 0.0
            
            results['train_correlations'].append({
                'feature': feat_name,
                'correlation': float(train_corr)
            })
            
            results['val_correlations'].append({
                'feature': feat_name,
                'correlation': float(val_corr)
            })
            
            # 상관관계 차이
            corr_diff = abs(train_corr - val_corr)
            if corr_diff > 0.1:  # 상관관계가 크게 다른 특징
                results['correlation_differences'].append({
                    'feature': feat_name,
                    'train_corr': float(train_corr),
                    'val_corr': float(val_corr),
                    'difference': float(corr_diff)
                })
        
        return results
    
    def generate_report(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       feature_names: list,
                       save_path: str = 'results') -> str:
        """
        종합 분석 리포트 생성
        """
        os.makedirs(save_path, exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("검증 로스 수렴 분석 리포트")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. Train/Val 분포 차이 분석
        report_lines.append("1. Train/Val 데이터 분포 차이 분석")
        report_lines.append("-" * 80)
        dist_analysis = self.analyze_train_val_distribution(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        target_stats = dist_analysis['target_stats']
        report_lines.append("타겟 분포:")
        report_lines.append(f"  Train: 평균={target_stats['train_mean']:.4f}, std={target_stats['train_std']:.4f}")
        report_lines.append(f"  Val: 평균={target_stats['val_mean']:.4f}, std={target_stats['val_std']:.4f}")
        report_lines.append(f"  평균 차이: {target_stats['mean_diff']:.4f}")
        report_lines.append(f"  Std 비율: {target_stats['std_ratio']:.4f}")
        
        if target_stats['mean_diff'] > 0.1 or abs(target_stats['std_ratio'] - 1.0) > 0.3:
            report_lines.append("  ⚠️ 경고: Train과 Val의 타겟 분포가 크게 다릅니다!")
            report_lines.append("     이는 시계열 데이터의 분포 변화(concept drift)를 나타냅니다.")
            report_lines.append("     해결: 더 많은 데이터 수집 또는 변화율 사용")
        report_lines.append("")
        
        if dist_analysis['distribution_differences']:
            report_lines.append("분포 차이가 큰 특징 (상위 10개):")
            sorted_diffs = sorted(dist_analysis['distribution_differences'], 
                                key=lambda x: x['mean_diff'], reverse=True)[:10]
            for diff in sorted_diffs:
                report_lines.append(
                    f"  {diff['feature']}: "
                    f"평균 차이={diff['mean_diff']:.4f}, "
                    f"std 비율={diff['std_ratio']:.4f}"
                )
            report_lines.append("")
        
        # 2. 특징 중요도 분석
        report_lines.append("2. 특징 중요도 분석 (Train vs Val)")
        report_lines.append("-" * 80)
        importance_analysis = self.analyze_feature_importance(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        # Train에서 유용한 특징
        train_sorted = sorted(importance_analysis['train_correlations'],
                            key=lambda x: abs(x['correlation']), reverse=True)[:10]
        report_lines.append("Train에서 상관관계가 높은 특징 (상위 10개):")
        for corr_info in train_sorted:
            report_lines.append(f"  {corr_info['feature']}: {corr_info['correlation']:.4f}")
        report_lines.append("")
        
        # Val에서 유용한 특징
        val_sorted = sorted(importance_analysis['val_correlations'],
                          key=lambda x: abs(x['correlation']), reverse=True)[:10]
        report_lines.append("Val에서 상관관계가 높은 특징 (상위 10개):")
        for corr_info in val_sorted:
            report_lines.append(f"  {corr_info['feature']}: {corr_info['correlation']:.4f}")
        report_lines.append("")
        
        # 상관관계 차이가 큰 특징
        if importance_analysis['correlation_differences']:
            report_lines.append("Train/Val 상관관계 차이가 큰 특징:")
            sorted_corr_diffs = sorted(importance_analysis['correlation_differences'],
                                     key=lambda x: x['difference'], reverse=True)[:10]
            for diff in sorted_corr_diffs:
                report_lines.append(
                    f"  {diff['feature']}: "
                    f"Train={diff['train_corr']:.4f}, "
                    f"Val={diff['val_corr']:.4f}, "
                    f"차이={diff['difference']:.4f}"
                )
            report_lines.append("")
        
        # 3. Val Loss 수렴하지 않는 원인 분석
        report_lines.append("3. Val Loss 수렴하지 않는 원인 분석")
        report_lines.append("-" * 80)
        
        # 타겟 분포 차이
        if target_stats['mean_diff'] > 0.1:
            report_lines.append("원인 1: 타겟 분포 차이")
            report_lines.append(f"  Train과 Val의 타겟 평균이 {target_stats['mean_diff']:.4f}만큼 다릅니다.")
            report_lines.append("  → 해결: 변화율 사용 (이미 적용됨)")
            report_lines.append("")
        
        # 특징 분포 차이
        if len(dist_analysis['distribution_differences']) > 5:
            report_lines.append("원인 2: 특징 분포 차이")
            report_lines.append(f"  {len(dist_analysis['distribution_differences'])}개 특징의 분포가 크게 다릅니다.")
            report_lines.append("  → 해결: 더 일반적인 특징 사용 또는 정규화 강화")
            report_lines.append("")
        
        # 상관관계 차이
        if len(importance_analysis['correlation_differences']) > 5:
            report_lines.append("원인 3: 특징-타겟 상관관계 차이")
            report_lines.append(f"  {len(importance_analysis['correlation_differences'])}개 특징의 상관관계가 크게 다릅니다.")
            report_lines.append("  → 해결: Train과 Val 모두에서 유용한 특징 추가")
            report_lines.append("")
        
        # 데이터 부족
        if len(X_train) < 10000:
            report_lines.append("원인 4: 학습 데이터 부족")
            report_lines.append(f"  학습 데이터가 {len(X_train)}개로 부족할 수 있습니다.")
            report_lines.append("  → 해결: 더 많은 데이터 수집")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # 리포트 저장
        report_path = os.path.join(save_path, 'validation_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"검증 로스 수렴 분석 리포트가 저장되었습니다: {report_path}")
        
        return report_text


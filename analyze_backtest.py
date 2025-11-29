"""
백테스팅 결과 분석 스크립트
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_backtest_results():
    """백테스팅 결과 분석"""
    # 데이터 로드
    signals_df = pd.read_csv('results/backtest_signals.csv')
    portfolio_df = pd.read_csv('results/backtest_portfolio.csv')
    
    print("=" * 60)
    print("백테스팅 결과 상세 분석")
    print("=" * 60)
    
    # 1. 시그널 통계
    print("\n[1] 시그널 통계")
    print(f"  총 시그널 수: {len(signals_df)}")
    print(f"  Hold: {len(signals_df[signals_df['signal']=='hold'])} ({len(signals_df[signals_df['signal']=='hold'])/len(signals_df)*100:.1f}%)")
    print(f"  Buy: {len(signals_df[signals_df['signal']=='buy'])} ({len(signals_df[signals_df['signal']=='buy'])/len(signals_df)*100:.1f}%)")
    print(f"  Sell: {len(signals_df[signals_df['signal']=='sell'])} ({len(signals_df[signals_df['signal']=='sell'])/len(signals_df)*100:.1f}%)")
    
    # 2. 예측 변화율 통계
    print("\n[2] 예측 변화율 통계")
    print(f"  평균: {signals_df['predicted_change'].mean():.4f}%")
    print(f"  표준편차: {signals_df['predicted_change'].std():.4f}%")
    print(f"  최소: {signals_df['predicted_change'].min():.4f}%")
    print(f"  최대: {signals_df['predicted_change'].max():.4f}%")
    print(f"  절댓값 평균: {signals_df['predicted_change'].abs().mean():.4f}%")
    
    # 3. 실제 변화율 통계
    print("\n[3] 실제 변화율 통계")
    print(f"  평균: {signals_df['actual_change'].mean():.4f}%")
    print(f"  표준편차: {signals_df['actual_change'].std():.4f}%")
    print(f"  최소: {signals_df['actual_change'].min():.4f}%")
    print(f"  최대: {signals_df['actual_change'].max():.4f}%")
    print(f"  절댓값 평균: {signals_df['actual_change'].abs().mean():.4f}%")
    
    # 4. 방향 정확도
    print("\n[4] 방향 정확도")
    correct = ((signals_df['predicted_change'] > 0) & (signals_df['actual_change'] > 0)) | \
              ((signals_df['predicted_change'] < 0) & (signals_df['actual_change'] < 0))
    direction_accuracy = correct.sum() / len(signals_df) * 100
    print(f"  {correct.sum()}/{len(signals_df)} = {direction_accuracy:.2f}%")
    
    # 5. 신뢰도 임계값 분석
    print("\n[5] 신뢰도 임계값 분석 (2%)")
    above_threshold_pred = len(signals_df[signals_df['predicted_change'].abs() >= 2.0])
    above_threshold_actual = len(signals_df[signals_df['actual_change'].abs() >= 2.0])
    print(f"  2% 이상 예측 변화: {above_threshold_pred}회 ({above_threshold_pred/len(signals_df)*100:.1f}%)")
    print(f"  2% 이상 실제 변화: {above_threshold_actual}회 ({above_threshold_actual/len(signals_df)*100:.1f}%)")
    
    # 6. 예측 오차 분석
    print("\n[6] 예측 오차 분석")
    mae = (signals_df['predicted_change'] - signals_df['actual_change']).abs().mean()
    mse = ((signals_df['predicted_change'] - signals_df['actual_change']) ** 2).mean()
    rmse = np.sqrt(mse)
    print(f"  MAE: {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    
    # 7. 포트폴리오 분석
    print("\n[7] 포트폴리오 분석")
    initial_value = portfolio_df['portfolio_value'].iloc[0]
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    print(f"  초기 자본금: ${initial_value:,.2f}")
    print(f"  최종 자산: ${final_value:,.2f}")
    print(f"  총 수익률: {total_return:+.2f}%")
    print(f"  거래 횟수: {len(portfolio_df[portfolio_df['position'] > 0])}회 (포지션 보유)")
    
    # 8. 문제점 진단
    print("\n[8] 문제점 진단")
    if above_threshold_pred == 0:
        print("  ⚠️ 심각: 2% 이상의 변화를 예측한 적이 없습니다!")
        print("     → 모델이 큰 변화를 예측하지 못하고 있습니다.")
    if direction_accuracy < 50:
        print(f"  ⚠️ 경고: 방향 정확도가 {direction_accuracy:.1f}%로 랜덤 수준입니다.")
        print("     → 모델이 방향을 예측하지 못하고 있습니다.")
    if signals_df['predicted_change'].abs().mean() < 0.1:
        print(f"  ⚠️ 경고: 예측 변화율의 절댓값 평균이 {signals_df['predicted_change'].abs().mean():.4f}%로 매우 작습니다.")
        print("     → 모델이 거의 변화를 예측하지 않고 있습니다.")
    
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)

if __name__ == '__main__':
    analyze_backtest_results()


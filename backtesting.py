"""
백테스팅 모듈
과거 데이터를 사용하여 모델의 실제 거래 성능을 시뮬레이션
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
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
                 min_confidence: float = 0.02,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001):  # 0.1% 수수료
        """
        Args:
            model_path: 학습된 모델 파일 경로
            window_size: 슬라이딩 윈도우 크기
            min_confidence: 최소 신뢰도 (가격 변화율, 2% = 0.02)
            initial_capital: 초기 자본금
            commission: 거래 수수료 (기본값: 0.1%)
        """
        self.model_path = model_path
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.initial_capital = initial_capital
        self.commission = commission
        
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
            scaler_type='robust'
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
        특정 시점에서 예측을 위한 데이터 준비
        
        Args:
            df_features: 이미 피처 엔지니어링이 완료된 전체 데이터프레임
            start_idx: 시작 인덱스
        
        Returns:
            X: 입력 시퀀스 (1, window_size, n_features)
            current_price: 현재 가격
        """
        # 필요한 데이터 범위 확인
        if start_idx < self.window_size:
            return None, None
        
        # window_size 개의 데이터 선택 (start_idx 이전의 window_size 개)
        start_data_idx = max(0, start_idx - self.window_size)
        end_data_idx = start_idx
        
        # 스케일러가 기대하는 feature 개수 확인
        scaler_expected_features = self.preprocessor.scaler.n_features_in_
        
        # 스케일러가 기대하는 feature 순서대로 데이터 준비
        # self.feature_names에서 정확히 scaler_expected_features 개만 사용
        scaler_feature_values = []
        feature_names_to_use = self.feature_names[:scaler_expected_features] if len(self.feature_names) >= scaler_expected_features else self.feature_names
        
        for col in feature_names_to_use:
            if col in df_features.columns:
                scaler_feature_values.append(df_features[col].iloc[start_data_idx:end_data_idx].values)
            else:
                # 누락된 feature는 0으로 채움
                scaler_feature_values.append(np.zeros(end_data_idx - start_data_idx))
        
        # 부족한 feature는 0으로 채움
        while len(scaler_feature_values) < scaler_expected_features:
            scaler_feature_values.append(np.zeros(end_data_idx - start_data_idx))
        
        # (n_samples, n_features) 형태로 변환 (스케일러용)
        recent_data_for_scaler = np.column_stack(scaler_feature_values)
        
        # 정확히 scaler_expected_features 개인지 확인
        if recent_data_for_scaler.shape[1] != scaler_expected_features:
            if recent_data_for_scaler.shape[1] > scaler_expected_features:
                recent_data_for_scaler = recent_data_for_scaler[:, :scaler_expected_features]
            else:
                padding = np.zeros((recent_data_for_scaler.shape[0], scaler_expected_features - recent_data_for_scaler.shape[1]))
                recent_data_for_scaler = np.hstack([recent_data_for_scaler, padding])
        
        # window_size보다 적으면 패딩
        if len(recent_data_for_scaler) < self.window_size:
            padding = np.zeros((self.window_size - len(recent_data_for_scaler), len(self.feature_names)))
            recent_data_for_scaler = np.vstack([padding, recent_data_for_scaler])
        
        # 스케일링 (스케일러는 self.feature_names 순서대로 기대)
        feature_flat = recent_data_for_scaler.reshape(-1, recent_data_for_scaler.shape[-1])
        feature_scaled_scaler = self.preprocessor.scaler.transform(feature_flat)
        feature_scaled_scaler = feature_scaled_scaler.reshape(1, self.window_size, -1)
        
        # 모델이 기대하는 feature만 선택 (리얼타임과 동일한 로직)
        model_feature_names = getattr(self.preprocessor, 'model_feature_columns', None)
        if model_feature_names is None:
            model_feature_names = self.feature_names
        
        # 스케일러 feature에서 모델 feature 인덱스 찾기 (리얼타임과 동일)
        scaler_expected_features = self.preprocessor.scaler.n_features_in_
        model_expected_features = len(model_feature_names) if model_feature_names else scaler_expected_features
        
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
                feature_scaled = feature_scaled_scaler[:, :, model_feature_indices]
            else:
                # 인덱스 매칭 실패 시 앞부분만 사용
                feature_scaled = feature_scaled_scaler[:, :, :model_expected_features]
        else:
            # feature 개수가 같으면 그대로 사용
            feature_scaled = feature_scaled_scaler
        
        # 모델이 기대하는 feature 개수에 맞춤
        if feature_scaled.shape[2] != self.num_features:
            if feature_scaled.shape[2] > self.num_features:
                feature_scaled = feature_scaled[:, :, :self.num_features]
            else:
                # 부족한 경우 0으로 패딩
                padding = np.zeros((1, self.window_size, self.num_features - feature_scaled.shape[2]))
                feature_scaled = np.concatenate([feature_scaled, padding], axis=2)
        
        # 현재 가격 (원본 데이터에서 가져오기 - 'close' 컬럼이 있어야 함)
        # start_idx 시점의 가격을 사용 (예측 시점)
        if 'close' in df_features.columns:
            current_price = float(df_features['close'].iloc[start_idx])
        else:
            # close가 없으면 해당 시점의 가격 사용
            current_price = float(df_features.iloc[start_idx, 0])
        
        return feature_scaled, current_price
    
    def _generate_signal(self, current_price: float, predicted_price: float) -> str:
        """
        매수/매도 시그널 생성
        
        Args:
            current_price: 현재 가격
            predicted_price: 예측 가격
        
        Returns:
            'buy', 'sell', 'hold'
        """
        price_change_pct = (predicted_price - current_price) / current_price
        
        if price_change_pct > self.min_confidence:
            return 'buy'
        elif price_change_pct < -self.min_confidence:
            return 'sell'
        else:
            return 'hold'
    
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
        print(f"최소 신뢰도: {self.min_confidence*100:.2f}%")
        print()
        
        # 데이터 수집
        print("데이터 수집 중...")
        df_raw = self.fetcher.fetch_ohlcv(
            since=start_date,
            limit=10000,
            timeframe=timeframe
        )
        
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
        
        # 초기 상태
        cash = self.initial_capital
        position = 0.0  # 보유 중인 BTC 수량
        portfolio_value = cash
        entry_price = 0.0
        
        # 거래 기록 초기화
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        
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
                # 모델 예측 (멀티타겟)
                y_pred_scaled = self.model.predict(X, verbose=0)  # (1, 3)
                
                # 디버깅: 모델 원시 출력 확인 (처음 몇 개만)
                if i < start_idx + 5:
                    print(f"디버깅 [인덱스 {i}]: 모델 원시 출력 (스케일링 전) - 3분: {y_pred_scaled[0, 0]:.6f}, 5분: {y_pred_scaled[0, 1]:.6f}, 15분: {y_pred_scaled[0, 2]:.6f}")
                
                # 역변환 (5분 타겟 사용)
                y_pred_changes = self.preprocessor.target_scaler.inverse_transform(y_pred_scaled)
                
                # 디버깅: 역변환 후 변화율 확인 (처음 몇 개만)
                if i < start_idx + 5:
                    print(f"디버깅 [인덱스 {i}]: 역변환 후 변화율 - 3분: {y_pred_changes[0, 0]:.6f}, 5분: {y_pred_changes[0, 1]:.6f}, 15분: {y_pred_changes[0, 2]:.6f}")
                
                change_5m = np.clip(y_pred_changes[0, 1], -0.5, 0.5)  # 5분 타겟
                
                # 디버깅: 클리핑 후 변화율 확인 (처음 몇 개만)
                if i < start_idx + 5:
                    print(f"디버깅 [인덱스 {i}]: 클리핑 후 5분 변화율: {change_5m:.6f}")
                
                predicted_price = current_price * (1 + change_5m)
                
                # 시그널 생성
                signal = self._generate_signal(current_price, predicted_price)
                
                # 실제 가격 (5분 후)
                if i + 5 < len(df_features):
                    actual_price = float(df_features['close'].iloc[i + 5])
                    actual_change = (actual_price - current_price) / current_price
                    
                    # 예측 정확도 확인
                    predicted_change = change_5m
                    if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0):
                        successful_predictions += 1
                    total_predictions += 1
                else:
                    actual_price = current_price
                    actual_change = 0.0
                
                # 거래 실행
                if signal == 'buy' and position == 0:
                    # 매수
                    btc_amount = cash / current_price
                    commission_cost = cash * self.commission
                    position = btc_amount * (1 - self.commission)
                    cash = 0.0
                    entry_price = current_price
                    
                    self.trades.append({
                        'timestamp': df_features.index[i],
                        'type': 'buy',
                        'price': current_price,
                        'amount': position,
                        'commission': commission_cost
                    })
                
                elif signal == 'sell' and position > 0:
                    # 매도
                    cash = position * current_price * (1 - self.commission)
                    profit = cash - (entry_price * position / (1 - self.commission))
                    position = 0.0
                    
                    self.trades.append({
                        'timestamp': df_features.index[i],
                        'type': 'sell',
                        'price': current_price,
                        'amount': position,
                        'profit': profit,
                        'commission': current_price * position * self.commission
                    })
                
                # 포트폴리오 가치 계산
                if position > 0:
                    portfolio_value = position * current_price
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
                    'signal': signal,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'predicted_change': change_5m * 100,
                    'actual_change': actual_change * 100
                })
                
            except Exception as e:
                print(f"⚠️ 경고: 인덱스 {i}에서 예측 실패: {e}")
                continue
        
        # 최종 포트폴리오 가치
        final_price = float(df_features['close'].iloc[-1])
        if position > 0:
            final_portfolio_value = position * final_price * (1 - self.commission)
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
        
        # 거래 통계
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        # 승률
        profitable_trades = [t for t in sell_trades if 'profit' in t and t['profit'] > 0]
        win_rate = len(profitable_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0.0
        
        # 평균 수익/손실
        if len(sell_trades) > 0:
            profits = [t.get('profit', 0) for t in sell_trades]
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
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
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
        print(f"  매수 거래: {results['buy_trades']}회")
        print(f"  매도 거래: {results['sell_trades']}회")
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
        
        # 매수/매도 시그널 표시
        signals_df = pd.DataFrame(results['signals'])
        if len(signals_df) > 0:
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            buy_signals = signals_df[signals_df['signal'] == 'buy']
            sell_signals = signals_df[signals_df['signal'] == 'sell']
            
            if len(buy_signals) > 0:
                axes[1].scatter(buy_signals['timestamp'], 
                              portfolio_df.loc[buy_signals['timestamp'], 'price'].values,
                              color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            if len(sell_signals) > 0:
                axes[1].scatter(sell_signals['timestamp'],
                              portfolio_df.loc[sell_signals['timestamp'], 'price'].values,
                              color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
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
    parser.add_argument('--capital', type=float, default=10000.0, help='초기 자본금')
    parser.add_argument('--confidence', type=float, default=0.02, help='최소 신뢰도 (기본값: 0.02 = 2%%)')
    
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
        min_confidence=args.confidence,
        initial_capital=args.capital
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


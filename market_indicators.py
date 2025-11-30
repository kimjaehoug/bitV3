"""
고급 시장 지표 분석 모듈
- 오더북 불균형 (Order Book Imbalance)
- 청산 클러스터 (Liquidation Clusters)
- 변동성 압축 (Volatility Squeeze)
- OI 급증 (Open Interest Surge)
- CVD 전환 (Cumulative Volume Delta)
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class MarketIndicators:
    """고급 시장 지표 분석 클래스"""
    
    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        """
        Args:
            exchange: ccxt 거래소 객체 (None이면 새로 생성)
        """
        if exchange is None:
            self.exchange = ccxt.binance({
                'options': {
                    'defaultType': 'future',  # 선물 거래
                },
                'enableRateLimit': True,
            })
        else:
            self.exchange = exchange
        
        self.symbol = 'BTC/USDT'
        
        # 이전 데이터 저장 (CVD 계산용)
        self.last_cvd = 0.0
        self.last_timestamp = None
        
        # 변동성 압축 계산용 데이터 저장
        self.volatility_history = []
        self.max_history_length = 100
    
    def get_orderbook_imbalance(self, depth: int = 20) -> Dict:
        """
        오더북 불균형 계산
        
        Args:
            depth: 분석할 오더북 깊이 (기본값: 20)
        
        Returns:
            {
                'bid_volume': float,  # 매수 호가 총량
                'ask_volume': float,   # 매도 호가 총량
                'imbalance_ratio': float,  # 불균형 비율 (-1 ~ 1)
                'imbalance_strength': str,  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
                'weighted_bid_price': float,  # 가중 평균 매수 가격
                'weighted_ask_price': float,  # 가중 평균 매도 가격
            }
        """
        try:
            # 오더북 가져오기
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=depth)
            
            bids = orderbook['bids']  # 매수 호가 [(가격, 수량), ...]
            asks = orderbook['asks']  # 매도 호가 [(가격, 수량), ...]
            
            # 가중 평균 계산 (가까운 호가일수록 높은 가중치)
            bid_volume = 0.0
            ask_volume = 0.0
            weighted_bid_price = 0.0
            weighted_ask_price = 0.0
            total_bid_weight = 0.0
            total_ask_weight = 0.0
            
            # 매수 호가 분석 (가까운 호가일수록 높은 가중치)
            for i, (price, volume) in enumerate(bids):
                weight = 1.0 / (i + 1)  # 거리 기반 가중치
                bid_volume += volume
                weighted_bid_price += price * volume * weight
                total_bid_weight += volume * weight
            
            # 매도 호가 분석
            for i, (price, volume) in enumerate(asks):
                weight = 1.0 / (i + 1)  # 거리 기반 가중치
                ask_volume += volume
                weighted_ask_price += price * volume * weight
                total_ask_weight += volume * weight
            
            # 가중 평균 가격 계산
            if total_bid_weight > 0:
                weighted_bid_price = weighted_bid_price / total_bid_weight
            if total_ask_weight > 0:
                weighted_ask_price = weighted_ask_price / total_ask_weight
            
            # 불균형 비율 계산 (-1 ~ 1)
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance_ratio = (bid_volume - ask_volume) / total_volume
            else:
                imbalance_ratio = 0.0
            
            # 불균형 강도 판단
            if imbalance_ratio >= 0.3:
                imbalance_strength = 'strong_buy'
            elif imbalance_ratio >= 0.1:
                imbalance_strength = 'buy'
            elif imbalance_ratio <= -0.3:
                imbalance_strength = 'strong_sell'
            elif imbalance_ratio <= -0.1:
                imbalance_strength = 'sell'
            else:
                imbalance_strength = 'neutral'
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance_ratio': imbalance_ratio,
                'imbalance_strength': imbalance_strength,
                'weighted_bid_price': weighted_bid_price,
                'weighted_ask_price': weighted_ask_price,
                'spread': weighted_ask_price - weighted_bid_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"오더북 불균형 계산 실패: {e}")
            return {
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'imbalance_ratio': 0.0,
                'imbalance_strength': 'neutral',
                'weighted_bid_price': 0.0,
                'weighted_ask_price': 0.0,
                'spread': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_liquidation_clusters(self, lookback_hours: int = 24) -> Dict:
        """
        청산 클러스터 분석 (강제 청산 = 강한 방향성)
        
        Args:
            lookback_hours: 분석할 시간 범위 (기본값: 24시간)
        
        Returns:
            {
                'long_liquidations': float,  # 롱 청산 금액
                'short_liquidations': float,  # 숏 청산 금액
                'liquidation_ratio': float,  # 청산 비율 (-1 ~ 1)
                'liquidation_strength': str,  # 'strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish'
                'total_liquidations': float,  # 총 청산 금액
            }
        """
        try:
            # 바이낸스 선물 거래소의 청산 데이터 가져오기
            # 참고: ccxt가 직접 지원하지 않으므로 공개 API 사용
            # 실제로는 바이낸스 공개 API나 다른 소스를 사용해야 함
            
            # 여기서는 거래량과 가격 변동을 기반으로 추정
            # 실제 청산 데이터는 바이낸스 공개 API나 전문 데이터 제공자 필요
            
            # 대안: 최근 가격 변동과 거래량을 기반으로 청산 추정
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # 최근 OHLCV 데이터로 급격한 가격 변동 감지
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe='5m',
                limit=288  # 24시간 (5분봉 * 288 = 24시간)
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 급격한 하락 = 롱 청산 가능성
            # 급격한 상승 = 숏 청산 가능성
            price_changes = df['close'].pct_change()
            volumes = df['volume']
            
            # 급격한 하락 구간 (롱 청산 추정)
            long_liquidation_signals = (price_changes < -0.01) & (volumes > volumes.quantile(0.8))
            long_liquidation_volume = volumes[long_liquidation_signals].sum()
            
            # 급격한 상승 구간 (숏 청산 추정)
            short_liquidation_signals = (price_changes > 0.01) & (volumes > volumes.quantile(0.8))
            short_liquidation_volume = volumes[short_liquidation_signals].sum()
            
            total_liquidation_volume = long_liquidation_volume + short_liquidation_volume
            
            if total_liquidation_volume > 0:
                liquidation_ratio = (short_liquidation_volume - long_liquidation_volume) / total_liquidation_volume
            else:
                liquidation_ratio = 0.0
            
            # 청산 강도 판단
            if liquidation_ratio >= 0.3:
                liquidation_strength = 'strong_bullish'  # 숏 청산 많음 = 상승 압력
            elif liquidation_ratio >= 0.1:
                liquidation_strength = 'bullish'
            elif liquidation_ratio <= -0.3:
                liquidation_strength = 'strong_bearish'  # 롱 청산 많음 = 하락 압력
            elif liquidation_ratio <= -0.1:
                liquidation_strength = 'bearish'
            else:
                liquidation_strength = 'neutral'
            
            return {
                'long_liquidations': float(long_liquidation_volume),
                'short_liquidations': float(short_liquidation_volume),
                'liquidation_ratio': liquidation_ratio,
                'liquidation_strength': liquidation_strength,
                'total_liquidations': float(total_liquidation_volume),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"청산 클러스터 분석 실패: {e}")
            return {
                'long_liquidations': 0.0,
                'short_liquidations': 0.0,
                'liquidation_ratio': 0.0,
                'liquidation_strength': 'neutral',
                'total_liquidations': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_volatility_squeeze(self, lookback_periods: int = 20) -> Dict:
        """
        변동성 압축 후 폭발 감지
        
        Args:
            lookback_periods: 분석할 기간 (기본값: 20)
        
        Returns:
            {
                'current_volatility': float,  # 현재 변동성
                'avg_volatility': float,  # 평균 변동성
                'volatility_ratio': float,  # 변동성 비율
                'squeeze_status': str,  # 'squeeze', 'normal', 'expansion'
                'expansion_potential': str,  # 'high', 'medium', 'low'
            }
        """
        try:
            # 최근 OHLCV 데이터 가져오기
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe='5m',
                limit=lookback_periods + 10
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # ATR (Average True Range) 계산 (변동성 지표)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # 현재 변동성
            current_volatility = atr.iloc[-1]
            
            # 평균 변동성
            avg_volatility = atr.iloc[-lookback_periods:].mean()
            
            # 변동성 비율
            if avg_volatility > 0:
                volatility_ratio = current_volatility / avg_volatility
            else:
                volatility_ratio = 1.0
            
            # 압축 상태 판단
            if volatility_ratio < 0.7:
                squeeze_status = 'squeeze'  # 변동성 압축
            elif volatility_ratio > 1.5:
                squeeze_status = 'expansion'  # 변동성 확장
            else:
                squeeze_status = 'normal'
            
            # 폭발 가능성 판단 (압축 후 확장 가능성)
            if squeeze_status == 'squeeze':
                # 압축이 심할수록 폭발 가능성 높음
                compression_level = 1.0 - volatility_ratio
                if compression_level > 0.4:
                    expansion_potential = 'high'
                elif compression_level > 0.2:
                    expansion_potential = 'medium'
                else:
                    expansion_potential = 'low'
            else:
                expansion_potential = 'low'
            
            return {
                'current_volatility': float(current_volatility),
                'avg_volatility': float(avg_volatility),
                'volatility_ratio': float(volatility_ratio),
                'squeeze_status': squeeze_status,
                'expansion_potential': expansion_potential,
                'compression_level': float(1.0 - volatility_ratio) if volatility_ratio < 1.0 else 0.0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"변동성 압축 분석 실패: {e}")
            return {
                'current_volatility': 0.0,
                'avg_volatility': 0.0,
                'volatility_ratio': 1.0,
                'squeeze_status': 'normal',
                'expansion_potential': 'low',
                'compression_level': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_open_interest_surge(self, lookback_periods: int = 20) -> Dict:
        """
        OI (Open Interest) 급증 감지 - 큰 돈이 한쪽에 베팅됨
        
        Args:
            lookback_periods: 분석할 기간 (기본값: 20)
        
        Returns:
            {
                'current_oi': float,  # 현재 미체결약정
                'avg_oi': float,  # 평균 미체결약정
                'oi_change_pct': float,  # OI 변화율
                'oi_surge_status': str,  # 'surge', 'normal', 'decline'
                'funding_rate': float,  # 펀딩 수수료율
            }
        """
        try:
            # 바이낸스 선물 거래소의 OI 데이터 가져오기
            # ccxt가 직접 지원하지 않으므로 공개 API 사용
            # 실제로는 바이낸스 공개 API 필요
            
            # 대안: 펀딩 수수료율과 거래량을 기반으로 추정
            # 펀딩 수수료율이 높으면 롱 포지션이 많음 (숏이 롱에게 수수료 지불)
            # 펀딩 수수료율이 낮으면 숏 포지션이 많음 (롱이 숏에게 수수료 지불)
            
            # 펀딩 수수료율 가져오기
            try:
                funding_rate_info = self.exchange.fetch_funding_rate(self.symbol)
                funding_rate = funding_rate_info.get('fundingRate', 0.0)
            except:
                # 펀딩 수수료율을 가져올 수 없으면 0으로 설정
                funding_rate = 0.0
            
            # 최근 거래량 데이터로 OI 추정
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe='5m',
                limit=lookback_periods + 10
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 거래량 증가 추세로 OI 급증 추정
            volumes = df['volume']
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.iloc[-lookback_periods:].mean()
            
            # OI 급증은 거래량 증가와 함께 나타남
            volume_change_pct = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
            
            # 펀딩 수수료율 기반 OI 추정
            # 펀딩 수수료율이 높으면 롱 포지션 많음
            if funding_rate > 0.01:  # 0.01% 이상
                oi_direction = 'long_dominant'
            elif funding_rate < -0.01:  # -0.01% 이하
                oi_direction = 'short_dominant'
            else:
                oi_direction = 'balanced'
            
            # OI 급증 상태 판단
            if volume_change_pct > 0.5 and abs(funding_rate) > 0.01:
                oi_surge_status = 'surge'
            elif volume_change_pct < -0.3:
                oi_surge_status = 'decline'
            else:
                oi_surge_status = 'normal'
            
            return {
                'current_oi': float(current_volume),  # 추정값
                'avg_oi': float(avg_volume),  # 추정값
                'oi_change_pct': float(volume_change_pct),
                'oi_surge_status': oi_surge_status,
                'oi_direction': oi_direction,
                'funding_rate': float(funding_rate),
                'funding_rate_pct': float(funding_rate * 100),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"OI 급증 분석 실패: {e}")
            return {
                'current_oi': 0.0,
                'avg_oi': 0.0,
                'oi_change_pct': 0.0,
                'oi_surge_status': 'normal',
                'oi_direction': 'balanced',
                'funding_rate': 0.0,
                'funding_rate_pct': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_cvd_turnover(self, lookback_periods: int = 20) -> Dict:
        """
        CVD (Cumulative Volume Delta) 전환 감지
        - 매수 거래량과 매도 거래량의 누적 차이
        
        Args:
            lookback_periods: 분석할 기간 (기본값: 20)
        
        Returns:
            {
                'cvd': float,  # 현재 CVD 값
                'cvd_change': float,  # CVD 변화량
                'cvd_trend': str,  # 'bullish', 'bearish', 'neutral'
                'cvd_turnover': bool,  # CVD 전환 여부
            }
        """
        try:
            # 최근 거래 데이터 가져오기
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe='5m',
                limit=lookback_periods + 10
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 가격 변화 방향으로 매수/매도 거래량 추정
            # 상승 캔들 = 매수 거래량 우세
            # 하락 캔들 = 매도 거래량 우세
            price_change = df['close'] - df['open']
            
            # 매수 거래량 (상승 캔들)
            buy_volume = df['volume'].where(price_change > 0, 0)
            # 매도 거래량 (하락 캔들)
            sell_volume = df['volume'].where(price_change < 0, 0)
            
            # Volume Delta (매수 - 매도)
            volume_delta = buy_volume - sell_volume
            
            # CVD (누적 Volume Delta)
            cvd = volume_delta.cumsum()
            
            current_cvd = cvd.iloc[-1]
            previous_cvd = cvd.iloc[-2] if len(cvd) > 1 else 0.0
            cvd_change = current_cvd - previous_cvd
            
            # CVD 추세 판단
            if current_cvd > 0:
                cvd_trend = 'bullish'  # 매수 우세
            elif current_cvd < 0:
                cvd_trend = 'bearish'  # 매도 우세
            else:
                cvd_trend = 'neutral'
            
            # CVD 전환 감지 (이전과 다른 부호)
            previous_trend = 'bullish' if previous_cvd > 0 else ('bearish' if previous_cvd < 0 else 'neutral')
            cvd_turnover = (cvd_trend != previous_trend) and (cvd_trend != 'neutral')
            
            # CVD 변화율
            if abs(previous_cvd) > 0:
                cvd_change_pct = (cvd_change / abs(previous_cvd)) * 100
            else:
                cvd_change_pct = 0.0
            
            return {
                'cvd': float(current_cvd),
                'cvd_change': float(cvd_change),
                'cvd_change_pct': float(cvd_change_pct),
                'cvd_trend': cvd_trend,
                'cvd_turnover': cvd_turnover,
                'buy_volume': float(buy_volume.iloc[-5:].sum()),  # 최근 5개 캔들 매수 거래량
                'sell_volume': float(sell_volume.iloc[-5:].sum()),  # 최근 5개 캔들 매도 거래량
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"CVD 전환 분석 실패: {e}")
            return {
                'cvd': 0.0,
                'cvd_change': 0.0,
                'cvd_change_pct': 0.0,
                'cvd_trend': 'neutral',
                'cvd_turnover': False,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'timestamp': datetime.now()
            }
    
    def get_all_indicators(self) -> Dict:
        """
        모든 시장 지표를 한 번에 가져오기
        
        Returns:
            모든 지표를 포함한 딕셔너리
        """
        try:
            indicators = {
                'orderbook_imbalance': self.get_orderbook_imbalance(),
                'liquidation_clusters': self.get_liquidation_clusters(),
                'volatility_squeeze': self.get_volatility_squeeze(),
                'oi_surge': self.get_open_interest_surge(),
                'cvd_turnover': self.get_cvd_turnover(),
                'timestamp': datetime.now()
            }
            
            return indicators
            
        except Exception as e:
            print(f"시장 지표 수집 실패: {e}")
            return {
                'orderbook_imbalance': {},
                'liquidation_clusters': {},
                'volatility_squeeze': {},
                'oi_surge': {},
                'cvd_turnover': {},
                'timestamp': datetime.now()
            }
    
    def get_trading_signal_from_indicators(self) -> Dict:
        """
        시장 지표를 종합하여 거래 신호 생성
        
        Returns:
            {
                'signal': str,  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
                'confidence': float,  # 신뢰도 (0 ~ 1)
                'reasons': list,  # 신호 근거 리스트
            }
        """
        try:
            indicators = self.get_all_indicators()
            
            signals = []
            confidence_scores = []
            
            # 1. 오더북 불균형
            ob = indicators['orderbook_imbalance']
            if ob['imbalance_strength'] == 'strong_buy':
                signals.append('오더북: 강한 매수 압력')
                confidence_scores.append(0.3)
            elif ob['imbalance_strength'] == 'buy':
                signals.append('오더북: 매수 압력')
                confidence_scores.append(0.15)
            elif ob['imbalance_strength'] == 'strong_sell':
                signals.append('오더북: 강한 매도 압력')
                confidence_scores.append(-0.3)
            elif ob['imbalance_strength'] == 'sell':
                signals.append('오더북: 매도 압력')
                confidence_scores.append(-0.15)
            
            # 2. 청산 클러스터
            lc = indicators['liquidation_clusters']
            if lc['liquidation_strength'] == 'strong_bullish':
                signals.append('청산: 숏 청산 많음 (상승 압력)')
                confidence_scores.append(0.25)
            elif lc['liquidation_strength'] == 'bullish':
                signals.append('청산: 숏 청산 우세')
                confidence_scores.append(0.1)
            elif lc['liquidation_strength'] == 'strong_bearish':
                signals.append('청산: 롱 청산 많음 (하락 압력)')
                confidence_scores.append(-0.25)
            elif lc['liquidation_strength'] == 'bearish':
                signals.append('청산: 롱 청산 우세')
                confidence_scores.append(-0.1)
            
            # 3. 변동성 압축 후 폭발
            vs = indicators['volatility_squeeze']
            if vs['squeeze_status'] == 'squeeze' and vs['expansion_potential'] == 'high':
                signals.append('변동성: 압축 후 폭발 가능성 높음')
                confidence_scores.append(0.2)
            
            # 4. OI 급증
            oi = indicators['oi_surge']
            if oi['oi_surge_status'] == 'surge':
                if oi['oi_direction'] == 'long_dominant':
                    signals.append('OI: 롱 포지션 급증')
                    confidence_scores.append(0.15)
                elif oi['oi_direction'] == 'short_dominant':
                    signals.append('OI: 숏 포지션 급증')
                    confidence_scores.append(-0.15)
            
            # 5. CVD 전환
            cvd = indicators['cvd_turnover']
            if cvd['cvd_turnover']:
                if cvd['cvd_trend'] == 'bullish':
                    signals.append('CVD: 매수 우세로 전환')
                    confidence_scores.append(0.2)
                elif cvd['cvd_trend'] == 'bearish':
                    signals.append('CVD: 매도 우세로 전환')
                    confidence_scores.append(-0.2)
            elif cvd['cvd_trend'] == 'bullish':
                signals.append('CVD: 매수 우세 지속')
                confidence_scores.append(0.1)
            elif cvd['cvd_trend'] == 'bearish':
                signals.append('CVD: 매도 우세 지속')
                confidence_scores.append(-0.1)
            
            # 종합 신호 계산
            total_confidence = sum(confidence_scores)
            abs_confidence = abs(total_confidence)
            
            # 신호 강도 판단
            if total_confidence >= 0.5:
                signal = 'strong_buy'
            elif total_confidence >= 0.2:
                signal = 'buy'
            elif total_confidence <= -0.5:
                signal = 'strong_sell'
            elif total_confidence <= -0.2:
                signal = 'sell'
            else:
                signal = 'neutral'
            
            # 신뢰도 정규화 (0 ~ 1)
            confidence = min(abs_confidence, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasons': signals,
                'raw_confidence': total_confidence,
                'indicators': indicators
            }
            
        except Exception as e:
            print(f"거래 신호 생성 실패: {e}")
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'reasons': [],
                'raw_confidence': 0.0,
                'indicators': {}
            }


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("시장 지표 분석 모듈 테스트")
    print("=" * 60)
    
    indicators = MarketIndicators()
    
    # 개별 지표 테스트
    print("\n[1] 오더북 불균형")
    ob = indicators.get_orderbook_imbalance()
    print(f"  불균형 비율: {ob['imbalance_ratio']:.4f}")
    print(f"  강도: {ob['imbalance_strength']}")
    
    print("\n[2] 청산 클러스터")
    lc = indicators.get_liquidation_clusters()
    print(f"  청산 비율: {lc['liquidation_ratio']:.4f}")
    print(f"  강도: {lc['liquidation_strength']}")
    
    print("\n[3] 변동성 압축")
    vs = indicators.get_volatility_squeeze()
    print(f"  변동성 비율: {vs['volatility_ratio']:.4f}")
    print(f"  상태: {vs['squeeze_status']}")
    print(f"  폭발 가능성: {vs['expansion_potential']}")
    
    print("\n[4] OI 급증")
    oi = indicators.get_open_interest_surge()
    print(f"  OI 변화율: {oi['oi_change_pct']*100:.2f}%")
    print(f"  상태: {oi['oi_surge_status']}")
    print(f"  펀딩 수수료율: {oi['funding_rate_pct']:.4f}%")
    
    print("\n[5] CVD 전환")
    cvd = indicators.get_cvd_turnover()
    print(f"  CVD: {cvd['cvd']:.2f}")
    print(f"  추세: {cvd['cvd_trend']}")
    print(f"  전환: {cvd['cvd_turnover']}")
    
    # 종합 신호
    print("\n[6] 종합 거래 신호")
    signal = indicators.get_trading_signal_from_indicators()
    print(f"  신호: {signal['signal']}")
    print(f"  신뢰도: {signal['confidence']:.2f}")
    print(f"  근거:")
    for reason in signal['reasons']:
        print(f"    - {reason}")


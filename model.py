"""
시계열 예측을 위한 간단하고 효과적인 모델 구조
학습이 잘 되는 구조로 재설계
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import numpy as np


class PatchCNNBiLSTM:
    """간단하고 효과적인 시계열 예측 모델"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 num_features: int,
                 patch_size: int = 5,
                 cnn_filters: list = [32, 64],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        # dropout_rate는 사용하지 않지만 호환성을 위해 유지
        """
        Args:
            input_shape: (window_size, num_features)
            num_features: 특징 개수
            patch_size: CNN 패치 크기
            cnn_filters: CNN 필터 개수 리스트 (간소화)
            lstm_units: LSTM 유닛 개수
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
        """
        self.input_shape = input_shape
        self.num_features = num_features
        self.patch_size = patch_size
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
    
    def _create_attention(self, query_value, attention_dim: int = 64, name_prefix: str = 'attention'):
        """
        작은 어텐션 메커니즘 생성
        급격한 변화와 방향성을 포착하기 위한 어텐션
        
        Args:
            query_value: 입력 텐서
            attention_dim: 어텐션 차원
            name_prefix: 레이어 이름 접두사 (중복 방지)
        """
        # Query, Key, Value 생성
        query = layers.Dense(attention_dim, use_bias=False, name=f'{name_prefix}_query')(query_value)
        key = layers.Dense(attention_dim, use_bias=False, name=f'{name_prefix}_key')(query_value)
        value = layers.Dense(attention_dim, use_bias=False, name=f'{name_prefix}_value')(query_value)
        
        # Attention scores 계산 (scaled dot-product attention)
        attention_scores = layers.Lambda(
            lambda x: tf.matmul(x[0], x[1], transpose_b=True) / tf.sqrt(tf.cast(attention_dim, tf.float32)),
            name=f'{name_prefix}_scores'
        )([query, key])
        
        # Softmax로 attention weights 계산
        attention_weights = layers.Softmax(name=f'{name_prefix}_weights')(attention_scores)
        
        # Attention 적용
        attention_output = layers.Lambda(
            lambda x: tf.matmul(x[0], x[1]),
            name=f'{name_prefix}_output'
        )([attention_weights, value])
        
        return attention_output, attention_weights
    
    def _direction_aware_loss(self, y_true, y_pred):
        """
        멀티타겟 방향성 인식 손실 함수
        30분, 1시간 변화율을 동시에 예측
        각 타겟에 대해 개별적으로 손실 계산 후 가중 평균
        """
        # y_true와 y_pred는 (batch_size, 2) 형태 (30분, 1시간)
        # 각 타겟에 대해 개별 손실 계산
        losses = []
        
        for i in range(2):  # 30분, 1시간
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            # 균형잡힌 손실 함수: 데이터 분포를 반영한 자연스러운 학습
            epsilon = 1e-6
            
            # 1. 기본 MSE 손실 - 가장 중요
            # 절대 오차를 정확히 측정
            absolute_mse = tf.reduce_mean(tf.square(y_true_i - y_pred_i))
            
            # 2. Huber Loss (큰 오차에 덜 민감하게)
            # 이상치에 덜 민감한 손실 함수
            delta = 0.01  # Huber loss threshold (1%)
            error = y_pred_i - y_true_i
            abs_error = tf.abs(error)
            huber_loss = tf.where(
                abs_error < delta,
                0.5 * tf.square(error),  # 작은 오차: MSE
                delta * abs_error - 0.5 * tf.square(delta)  # 큰 오차: 선형
            )
            huber_mse = tf.reduce_mean(huber_loss)
            
            # 3. 대칭적 손실 (양수/음수 균형) - 강화
            # 양수 예측과 음수 예측의 오차를 균형있게 평가
            positive_mask = tf.cast(y_true_i > 0, tf.float32)
            negative_mask = tf.cast(y_true_i < 0, tf.float32)
            
            # 양수 예측 오차
            positive_error = tf.reduce_mean(positive_mask * tf.square(y_pred_i - y_true_i))
            # 음수 예측 오차
            negative_error = tf.reduce_mean(negative_mask * tf.square(y_pred_i - y_true_i))
            # 균형 손실: 양수와 음수 오차의 차이가 작아지도록 (더 강하게)
            balance_loss = tf.square(positive_error - negative_error)
            
            # 3.5. 편향 제거 패널티 (예측 평균이 0에 가까워지도록)
            # 예측의 평균이 0에서 크게 벗어나면 강한 패널티
            pred_mean = tf.reduce_mean(y_pred_i)
            # 실제 타겟의 평균도 고려 (타겟이 편향되어 있을 수 있음)
            true_mean = tf.reduce_mean(y_true_i)
            # 예측 평균이 실제 평균과 크게 다르면 패널티
            mean_bias = tf.abs(pred_mean - true_mean)
            bias_penalty = tf.square(mean_bias) * 50.0  # 강한 패널티
            
            # 3.6. 양수 예측 장려 (양수 타겟에 대해 양수 예측을 하면 보너스)
            # 양수 타겟에 대해 양수 예측을 하면 손실 감소
            positive_target_positive_pred = tf.reduce_mean(
                positive_mask * tf.cast(y_pred_i > 0, tf.float32) * tf.square(y_pred_i - y_true_i)
            )
            positive_target_negative_pred = tf.reduce_mean(
                positive_mask * tf.cast(y_pred_i <= 0, tf.float32) * tf.square(y_pred_i - y_true_i)
            )
            # 양수 타겟에 대해 음수 예측하면 추가 패널티
            positive_prediction_penalty = tf.maximum(0.0, positive_target_negative_pred - positive_target_positive_pred) * 20.0
            
            # 4. 변화율 크기별 가중치 (큰 변화율에 더 집중하되, 작은 변화율도 정확히)
            change_magnitude = tf.abs(y_true_i) + epsilon
            # 작은 변화율(0.01% 이하): 가중치 1.0
            # 중간 변화율(0.01~0.1%): 가중치 2.0
            # 큰 변화율(0.1% 이상): 가중치 5.0
            magnitude_weights = tf.where(
                change_magnitude < 0.0001,  # 0.01% 이하
                1.0,
                tf.where(
                    change_magnitude < 0.001,  # 0.01~0.1%
                    2.0,
                    5.0  # 0.1% 이상
                )
            )
            weighted_mse = tf.reduce_mean(magnitude_weights * tf.square(y_true_i - y_pred_i))
            
            # 5. 방향 일치 보너스 (방향이 맞으면 손실 감소)
            # 예측 방향이 맞으면 손실을 약간 감소시켜 방향 정확도 향상
            true_sign = tf.sign(y_true_i)
            pred_sign = tf.sign(y_pred_i)
            direction_match = tf.cast(tf.equal(true_sign, pred_sign), tf.float32)
            # 방향이 맞으면 오차를 10% 감소
            direction_bonus = 1.0 - 0.1 * direction_match
            direction_weighted_mse = tf.reduce_mean(direction_bonus * tf.square(y_true_i - y_pred_i))
            
            # 최종 손실: 기본 MSE + Huber Loss + 균형 손실 + 가중 MSE + 방향 보너스 + 편향 제거 + 양수 예측 장려
            # 자연스러운 학습을 위한 균형잡힌 조합
            mse_loss_i = (0.3 * absolute_mse + 
                         0.2 * huber_mse + 
                         0.15 * weighted_mse + 
                         0.15 * direction_weighted_mse + 
                         0.1 * balance_loss + 
                         0.08 * bias_penalty + 
                         0.02 * positive_prediction_penalty)
            
            # 방향성 손실 (배치 크기가 2보다 클 때만)
            batch_size = tf.shape(y_true_i)[0]
            direction_loss_i = tf.cond(
                batch_size > 2,
                lambda: self._compute_direction_loss_improved(
                    tf.expand_dims(y_true_i, axis=1),
                    tf.expand_dims(y_pred_i, axis=1)
                ),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )
            
            # 각 타겟의 최종 손실 (MSE + 방향성 손실)
            alpha = 0.7
            total_loss_i = alpha * mse_loss_i + (1 - alpha) * direction_loss_i
            losses.append(total_loss_i)
        
        # 각 타겟에 가중치 부여 (1시간이 가장 중요, 30분은 보조)
        weights = [0.4, 0.6]  # 30분, 1시간
        total_loss = sum(w * loss for w, loss in zip(weights, losses))
        
        # NaN 방지
        total_loss = tf.where(tf.math.is_nan(total_loss), losses[1], total_loss)  # 1시간 손실을 기본값으로
        
        return total_loss
    
    def _compute_direction_loss_improved(self, y_true, y_pred):
        """강화된 방향성 손실 계산 - 방향 정확도와 세부적인 변화 포착에 집중"""
        # 실제 방향: 다음 값 - 현재 값
        true_diff = y_true[1:] - y_true[:-1]
        # 예측 방향: 다음 예측값 - 현재 예측값
        pred_diff = y_pred[1:] - y_pred[:-1]
        
        # 값 클리핑 (NaN 방지)
        true_diff = tf.clip_by_value(true_diff, -1e6, 1e6)
        pred_diff = tf.clip_by_value(pred_diff, -1e6, 1e6)
        
        # 실제 방향 부호 (1: 상승, -1: 하락, 0: 변화 없음)
        true_sign = tf.sign(true_diff)
        pred_sign = tf.sign(pred_diff)
        
        # 방향 일치 여부 (부호가 같으면 1, 다르면 0)
        direction_match = tf.cast(
            tf.equal(true_sign, pred_sign),
            tf.float32
        )
        
        # 방향성 손실: 1 - 방향 일치율 (0에 가까울수록 좋음)
        direction_loss = 1.0 - tf.reduce_mean(direction_match)
        
        # 방향이 틀렸을 때 강한 패널티 (차이의 크기를 고려)
        wrong_direction_mask = 1.0 - direction_match
        
        # 정규화된 차이 오류 (방향이 틀렸을 때만)
        true_diff_norm = tf.abs(true_diff) / (tf.abs(true_diff) + 1.0)  # 0~1 정규화
        pred_diff_norm = tf.abs(pred_diff) / (tf.abs(pred_diff) + 1.0)
        diff_error = tf.abs(true_diff_norm - pred_diff_norm)
        
        # 방향이 틀렸을 때만 패널티 적용
        direction_penalty = tf.reduce_mean(wrong_direction_mask * (diff_error + 0.5))
        
        # 큰 변화 포착을 위한 가중 MSE (큰 변화에 더 큰 가중치)
        # 큰 변화를 예측하지 못하면 강한 패널티
        abs_true_diff = tf.abs(true_diff)
        # 큰 변화에 더 큰 가중치 (제곱 가중치)
        # 예: 변화가 0.01이면 가중치 1.0, 변화가 0.1이면 가중치 10.0
        large_change_weights = 1.0 + 100.0 * tf.square(abs_true_diff)  # 큰 변화에 훨씬 큰 가중치
        large_change_weights = tf.clip_by_value(large_change_weights, 1.0, 1000.0)  # 가중치 범위 제한
        
        # 큰 변화에 대한 가중 MSE
        large_change_mse = tf.reduce_mean(large_change_weights * tf.square(true_diff - pred_diff))
        
        # 예측 크기 부족 패널티: 실제로 큰 변화가 있었는데 예측이 작으면 패널티
        pred_diff_abs = tf.abs(pred_diff)
        magnitude_underestimate = tf.maximum(0.0, abs_true_diff - pred_diff_abs)  # 예측이 실제보다 작을 때만
        magnitude_penalty = tf.reduce_mean(tf.square(magnitude_underestimate)) * 5.0
        
        # 방향성 손실, 패널티, 큰 변화 MSE, 크기 부족 패널티 결합
        total_direction_loss = 0.15 * direction_loss + 0.4 * direction_penalty + 0.35 * large_change_mse + 0.1 * magnitude_penalty
        
        # 안전한 손실 계산 (NaN 방지)
        total_direction_loss = tf.where(
            tf.math.is_nan(total_direction_loss) | tf.math.is_inf(total_direction_loss),
            0.5,  # 기본값
            total_direction_loss
        )
        
        return total_direction_loss
    
    def build_model(self) -> keras.Model:
        """
        방향 정확도 개선을 위한 단순화된 구조
        - CNN으로 로컬 패턴 포착
        - BiLSTM으로 장기 패턴 학습
        - Attention으로 중요한 시점 집중
        - 방향성 손실 함수로 방향 예측 개선
        """
        # 입력 레이어
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # 1. Fine-grained Multi-scale CNN - 세부적인 변화 포착 강화
        # 매우 작은 커널: 즉각적인 변화 (1, 2) - 세부 노이즈 포착
        # 작은 커널: 단기 변화 (3)
        # 중간 커널: 단기 패턴 (5)
        # 큰 커널: 중기 패턴 (7)
        
        # 즉각적인 변화 포착 (kernel_size=1, 2)
        cnn_1 = layers.Conv1D(
            filters=24,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_1'
        )(inputs)
        cnn_1 = layers.BatchNormalization(name='bn_cnn_1')(cnn_1)
        
        cnn_2 = layers.Conv1D(
            filters=24,
            kernel_size=2,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_2'
        )(inputs)
        cnn_2 = layers.BatchNormalization(name='bn_cnn_2')(cnn_2)
        
        # 단기 변화 포착 (kernel_size=3)
        cnn_3 = layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_3'
        )(inputs)
        cnn_3 = layers.BatchNormalization(name='bn_cnn_3')(cnn_3)
        
        # 단기 패턴 포착 (kernel_size=5)
        cnn_5 = layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_5'
        )(inputs)
        cnn_5 = layers.BatchNormalization(name='bn_cnn_5')(cnn_5)
        
        # 중기 패턴 포착 (kernel_size=7)
        cnn_7 = layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_7'
        )(inputs)
        cnn_7 = layers.BatchNormalization(name='bn_cnn_7')(cnn_7)
        
        # Multi-scale CNN 출력 결합
        cnn = layers.Concatenate(name='cnn_multi_scale')([cnn_1, cnn_2, cnn_3, cnn_5, cnn_7])
        cnn = layers.Dropout(0.1, name='dropout_cnn')(cnn)
        
        # 1.5. Fine-grained Feature Extraction - 세부 패턴 강화
        # 추가 CNN 레이어로 세부적인 변화 포착
        cnn_fine = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='cnn_fine'
        )(cnn)
        cnn_fine = layers.BatchNormalization(name='bn_cnn_fine')(cnn_fine)
        cnn_fine = layers.Dropout(0.1, name='dropout_cnn_fine')(cnn_fine)
        
        # Residual connection: 원본 정보 보존
        # cnn과 cnn_fine의 차원이 다르므로 projection 필요
        if cnn.shape[-1] != cnn_fine.shape[-1]:
            cnn_proj = layers.Conv1D(
                filters=cnn_fine.shape[-1],
                kernel_size=1,
                padding='same',
                activation='linear',
                name='cnn_proj'
            )(cnn)
            cnn = layers.Add(name='cnn_residual')([cnn_proj, cnn_fine])
        else:
            cnn = layers.Add(name='cnn_residual')([cnn, cnn_fine])
        cnn = layers.LayerNormalization(name='ln_cnn')(cnn)
        
        # 2. Multi-layer BiLSTM - 장기 패턴 학습 강화
        # 첫 번째 BiLSTM 레이어
        lstm_1 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=keras.regularizers.l2(1e-6),
                name='bilstm_1'
            ),
            name='bidirectional_1'
        )(cnn)
        lstm_1 = layers.LayerNormalization(name='ln_lstm_1')(lstm_1)
        
        # 두 번째 BiLSTM 레이어 (더 깊은 패턴 학습)
        lstm_2 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                kernel_regularizer=keras.regularizers.l2(1e-6),
                name='bilstm_2'
            ),
            name='bidirectional_2'
        )(lstm_1)
        lstm_2 = layers.LayerNormalization(name='ln_lstm_2')(lstm_2)
        
        # Residual connection between LSTM layers
        # 차원이 같으므로 직접 더하기
        lstm_output = layers.Add(name='lstm_residual')([lstm_1, lstm_2])
        lstm_output = layers.LayerNormalization(name='ln_lstm_output')(lstm_output)
        
        # 3. Multi-head Attention - 세부적인 변화 포착 강화
        attention_dim = self.lstm_units * 2  # BiLSTM이므로 2배
        
        # 첫 번째 Attention head (전체 패턴)
        attention_1, attention_weights_1 = self._create_attention(
            lstm_output, attention_dim=attention_dim, name_prefix='attention_1'
        )
        attention_mean_1 = layers.GlobalAveragePooling1D(name='attention_mean_1')(attention_1)
        attention_max_1 = layers.GlobalMaxPooling1D(name='attention_max_1')(attention_1)
        
        # 두 번째 Attention head (세부 패턴 - 더 작은 차원)
        attention_2, attention_weights_2 = self._create_attention(
            lstm_output, attention_dim=attention_dim // 2, name_prefix='attention_2'
        )
        attention_mean_2 = layers.GlobalAveragePooling1D(name='attention_mean_2')(attention_2)
        attention_max_2 = layers.GlobalMaxPooling1D(name='attention_max_2')(attention_2)
        
        # LSTM의 마지막 출력 (전체 컨텍스트)
        lstm_last = layers.Lambda(
            lambda x: x[:, -1, :],
            output_shape=lambda input_shape: (input_shape[0], input_shape[2]),
            name='lstm_last'
        )(lstm_output)
        
        # LSTM의 평균 출력 (전체 컨텍스트 평균)
        lstm_mean = layers.GlobalAveragePooling1D(name='lstm_mean')(lstm_output)
        
        # LSTM의 최대 출력 (최대 활성화)
        lstm_max = layers.GlobalMaxPooling1D(name='lstm_max')(lstm_output)
        
        # CNN의 마지막 출력 (세부 패턴)
        cnn_last = layers.Lambda(
            lambda x: x[:, -1, :],
            output_shape=lambda input_shape: (input_shape[0], input_shape[2]),
            name='cnn_last'
        )(cnn)
        
        # CNN의 평균 출력
        cnn_mean = layers.GlobalAveragePooling1D(name='cnn_mean')(cnn)
        
        # 특징 결합 (더 많은 정보 활용)
        combined = layers.Concatenate(name='combined_features')(
            [attention_mean_1, attention_max_1,  # Attention head 1
             attention_mean_2, attention_max_2,  # Attention head 2 (세부 패턴)
             lstm_last, lstm_mean, lstm_max,      # LSTM 다양한 집계
             cnn_last, cnn_mean]                  # CNN 세부 패턴
        )
        
        # 4. 공유 Dense 레이어
        shared_dense = layers.Dense(
            self.lstm_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='shared_dense'
        )(combined)
        shared_dense = layers.BatchNormalization(name='bn_shared')(shared_dense)
        shared_dense = layers.Dropout(0.25, name='dropout_shared')(shared_dense)
        
        # 5. 멀티타겟 예측 브랜치 (30분, 1시간 변화율)
        # 각 타겟에 대한 공유 레이어
        target_dense = layers.Dense(
            self.lstm_units // 2,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name='target_dense'
        )(shared_dense)
        target_dense = layers.BatchNormalization(name='bn_target')(target_dense)
        target_dense = layers.Dropout(0.2, name='dropout_target')(target_dense)
        
        # 30분 후 변화율 예측
        # 출력 레이어: 균형잡힌 초기화 (편향 없이 작은 값으로 시작)
        change_30m = layers.Dense(
            1,
            activation='linear',
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            bias_initializer=keras.initializers.Zeros(),  # 편향 없음
            name='change_30m'
        )(target_dense)
        
        # 1시간 후 변화율 예측
        change_1h = layers.Dense(
            1,
            activation='linear',
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            bias_initializer=keras.initializers.Zeros(),  # 편향 없음
            name='change_1h'
        )(target_dense)
        
        # 멀티타겟 출력 (30분, 1시간 변화율)
        outputs = layers.Concatenate(name='multi_target_output')([change_30m, change_1h])
        
        # 모델 생성
        model = keras.Model(
            inputs=inputs, 
            outputs=outputs, 
            name='Direction_Aware_Predictor'
        )
        
        # 방향성 인식 손실 함수 사용
        direction_loss = self._direction_aware_loss
        
        # 컴파일 (방향성 손실 함수 사용)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0
            ),
            loss=direction_loss,  # 방향성 인식 손실 함수
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def get_model(self) -> keras.Model:
        """모델 반환"""
        if self.model is None:
            self.build_model()
        return self.model
    
    def summary(self):
        """모델 구조 출력"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


if __name__ == "__main__":
    # 테스트 코드
    model_builder = PatchCNNBiLSTM(
        input_shape=(60, 50),
        num_features=50,
        patch_size=5,
        cnn_filters=[32, 64],
        lstm_units=128,
        dropout_rate=0.2
    )
    
    model = model_builder.build_model()
    model.summary()
    
    # 샘플 데이터로 테스트
    import numpy as np
    sample_input = np.random.randn(1, 60, 50)
    sample_output = model.predict(sample_input, verbose=0)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")

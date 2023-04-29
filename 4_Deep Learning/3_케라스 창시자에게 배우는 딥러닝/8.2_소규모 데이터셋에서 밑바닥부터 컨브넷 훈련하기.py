'''
1) 70% 성능: 작은 컨브넷 훈련
2) 80~85% 성능: 과대적합을 줄이기 위한 강력한 방법인 데이터증식
3) 97.5% 성능: 사전 훈련된 네트워크로 특성을 추출
4) 98.5% 성능: 사전 훈련된 네트워크를 세밀하게 튜닝
'''

from tensorflow import keras
from tensorflow.keras import layers


def func_make_model():

    inputs = keras.Input(shape=(180, 180, 3))

    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, actionation='rele')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

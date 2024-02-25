"""
import and dependancy
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import _keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


"""
GPU check and CUDA to use
1) Tensorflow 선택할 경우, 
2) GPU를 사용하도록 설정
"""

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# GPU를 사용하도록 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 성장을 허용(메모리 제한을 조절 가능: memery - 6GB로 제한)
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)]  
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 메모리 성장을 설정할 수 없는 경우
        print(e)


"""
Download the training dataset
1) check dataset size
"""
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

# train-dataset
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()

val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

# 데이터셋 형태 확인
print("훈련용 데이터셋 형태:", train_cache)
for image, label in train_cache.take(1):
    print("이미지 형태:", image.shape)
    print("라벨 형태", label.shape)
    
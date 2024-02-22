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
div2k_data.as_dataset()
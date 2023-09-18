"""
import
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


"""
훈련용 데이터셋 다운로드
"""
# Download DIV2K from TF Datasets
# Using bicubic 4x degradation type
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

# Taking train data from div2k_data object
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()
# Validation data
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()


"""
Data argumentation: image 자르고, 돌리고하고 훈련데이터 생성. 
"""
def flip_left_right(lowres_img, highres_img):
    """
    flips images to left and right
    - outputs random values from a uniform distribution in between 0 to 1
    1) rn < 0.5, return: lowres_img and highres_img
    2) rn >= 0.5, return: flip img
    """

    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (tf.image.flip_left_right(lowres_img), tf.image.flip_left_right(highres_img)),
        )


def random_rotate(lowres_img, highres_img):
    """
    rotate img by 90 degrees
    - outputs random value from a uniform distribution in between 0 to 4
    """

    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    """
    Crop images
    1) lowres_img: 24 x 24
    2) highres_img: 96 x 96
    """

    lowres_crop_size = hr_crop_size // scale
    lowres_img_shape = tf.shape(lowres_img)[:2]         # height, width

    lowres_height = tf.random.uniform(shape=(),
                                      maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32)

    lowres_width = tf.random.uniform(shape=(),
                                     maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32)

    highres_height = lowres_height * scale
    highres_width = lowres_width * scale

    lowres_img_cropped = lowres_img[
                         lowres_height: lowres_height + lowres_crop_size,
                         lowres_width: lowres_width + lowres_crop_size]

    highres_img_cropped = highres_img[
                         highres_height: highres_height + hr_crop_size,
                         highres_width: highres_width + hr_crop_size]

    return lowres_img_cropped, highres_img_cropped


"""
dataset을 data augmentation을 이용하여 생성.
lowres: 24 x 24 
"""
def dataset_object(dataset_cache, training=True):

    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres),
        num_parallel_calls=AUTOTUNE
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)

    # Batching Data
    ds = ds.batch(16)

    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat()
    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)


"""
데이터를 image에서 확인하기
lowres: 24 x 24 
"""
lowres, highres = next(iter(train_ds))

# High Resolution Images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(highres[i].numpy().astype("uint8"))
    plt.title(highres[i].shape)
    plt.axis("off")

# Low Resolution Images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(lowres[i].numpy().astype("uint8"))
    plt.title(lowres[i].shape)
    plt.axis("off")
plt.show()


"""
모델 build.
"""
class EDSRModel(tf.keras.Model):
    """
    Batch normalization layer 로 인하여 특성을 정규화하므로 출력값의 유연성을 제한할 수도 있기에
    해당 Model에서는 Batch normalization layer 삭제
    """
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        ## Compute gradients: self.trainable_variables 에 해당 변수들이 자동으로 포함
        trainable_vars = self.trainable.variables
        gradients = tape.gradient(loss, trainable_vars)

        ## update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}



























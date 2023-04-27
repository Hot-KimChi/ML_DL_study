''''''

'''
간단한 컨브넷 만들기
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def func_make_model():
    inputs = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model = func_make_model()
print(model.summary())


'''
MNIST 숫자 이미지에 이 컨브넷 훈련
'''

from tensorflow.keras.datasets import mnist
def func_dataset_object():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    return train_images, train_labels, test_images, test_labels


train_images, train_labels, test_images, test_labels = func_dataset_object()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'테스트 정확도: {test_acc:.3f}')
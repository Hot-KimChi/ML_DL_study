'''
---------------------------------------------------------------------------
1) 70% 성능: 작은 컨브넷 훈련
2) 80~85% 성능: 과대적합을 줄이기 위한 강력한 방법인 데이터증식
3) 97.5% 성능: 사전 훈련된 네트워크로 특성을 추출
4) 98.5% 성능: 사전 훈련된 네트워크를 세밀하게 튜닝
'''


'''
---------------------------------------------------------------------------
make dataset from directory
'''
import os, shutil, pathlib

org_dir = pathlib.Path('D:\PycharmProjects\dogs-vs-cats\\train')
new_dir = pathlib.Path('D:\PycharmProjects\dogs-vs-cats\dataset')

def func_make_subset(subset_name, start_index, end_index):
    for category in ('cat', 'dog'):
        dir = new_dir / subset_name / category
        os.makedirs(dir)

        fnames = [f'{category}.{i}.jpg'
                  for i in range(start_index, end_index)]

        for fname in fnames:
            print(fname)
            shutil.copyfile(src=org_dir/fname, dst=dir/fname)


func_make_subset('train', start_index=0, end_index=1000)
func_make_subset('validation', start_index=1000, end_index=1500)
func_make_subset('test', start_index=1500, end_index=2500)


'''
---------------------------------------------------------------------------
모델 만들기
'''
from tensorflow import keras
from keras import layers
import tensorflow as tf
def func_make_model():

    inputs = keras.Input(shape=(180, 180, 3))

    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
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

model = func_make_model()
print(model.summary())


'''
---------------------------------------------------------------------------
모델 훈련 설정하기
'''
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


'''
---------------------------------------------------------------------------
데이터 전처리
1) 사진 파일 load
2) JPEG 콘텐츠를 RGB 픽셀값으로 디코딩
3) 부동 소수점 타입의 텐서로 변환
4) 동일한 크기의 이미지로 변환
5) 배치로 묶는다.
--> image_dataset_from_directory 로 위의 단계 자돟화 가능
'''
train_dir = 'D:\PycharmProjects\dogs-vs-cats\dataset\\train'
validation_dir = 'D:\PycharmProjects\dogs-vs-cats\dataset\\validation'
test_dir = 'D:\PycharmProjects\dogs-vs-cats\dataset\\test'

from keras.utils import image_dataset_from_directory

batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

validation_ds = image_dataset_from_directory(
    validation_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )


'''
---------------------------------------------------------------------------
Dataset 반환하는 데이터와 레이블 크기 확인하기
'''
for data_batch, labels_batch in train_ds:
    print('데이터 배치 크기:', data_batch.shape)
    print('데이터 배치 크기:', labels_batch.shape)
    break


'''
---------------------------------------------------------------------------
Dataset 활용하여 모델 훈련하기
'''
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='convert_from_scratch.keras',
        save_best_only=True,
        monitor='val_loss')
]

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=validation_ds,
    callbacks=callbacks)


'''
---------------------------------------------------------------------------
결과 그래프로 그리기
'''
import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

cnt_epochs = range(1, len(accuracy)+1)

plt.plot(cnt_epochs, accuracy, 'bo', label='training_accuracy')
plt.plot(cnt_epochs, val_accuracy, 'b', label='validation_accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(cnt_epochs, loss, 'bo', label='training_loss')
plt.plot(cnt_epochs, val_loss, 'b', label='validation_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


'''
---------------------------------------------------------------------------
테스트 세트에서 모델 평가하기(0.735%)
- 비교적 훈련 샘플의 개수(2000개) 적기 때문에 과대적합 발생.
'''
test_model = keras.models.load_model('convert_from_scratch.keras')
test_loss, test_acc = test_model.evaluate(test_ds)
print(f'테스트 정확도: {test_acc:.3f}')


'''
---------------------------------------------------------------------------
- 과대적합을 해결하기 위하여, 데이터 증식 사용
- 데이터 증식 시, 모델에 같은 입력 데이터가 2번 주입되지 않지만, 입력 데이터들 사이에 상호 연관성이 크다.
    - 과대적합을 더 억제하기 위해, 밀집 연결 분류기 직전에 Dropout 층 추가
'''
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')

print(augmented_images.shape)


'''
---------------------------------------------------------------------------
- 이미지 증식과 드롭아웃을 포함한 컨브넷 만들기
'''
def func_make_model_aug_drop():

    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)

    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

model_aug_drop = func_make_model_aug_drop()
print(model_aug_drop.summary())

model_aug_drop.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='convert_from_scratch_with_augmentation.keras',
        save_best_only=True,
        monitor='val_loss')
]

history = model_aug_drop.fit(
    train_ds,
    epochs=100,
    validation_data=validation_ds,
    callbacks=callbacks)

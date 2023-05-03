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
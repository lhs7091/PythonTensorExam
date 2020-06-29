import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import PIL

np.set_printoptions(linewidth=1000)


def save_model(h5_path, model_path):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='softmax', input_shape=[784]))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    mnist = input_data.read_data_sets('mnist')
    model.fit(mnist.train.images, mnist.train.labels,
              validation_data=[mnist.validation.images, mnist.validation.labels],
              epochs=15, batch_size=128, verbose=0)

    # 케라스 모델과 변수 모두 저장
    model.save(h5_path)

    # -------------------------------------- #

    # 저장한 파일로부터 모델 변환 후 다시 저장
    converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
    #converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(h5_path)

    flat_data = converter.convert()

    with open(model_path, 'wb') as f:
        f.write(flat_data)


save_model('./model/mnist.h5', './model/mnist.tflite')


def read_testset(dir_path):
    filenames = os.listdir(dir_path)
    # filenames = ['0_003.png', '1_005.png', '2_013.png']

    # images를 배열이 들어있는 리스트로 생성하면 에러
    images = np.zeros([len(filenames), 784], dtype=np.float32)
    labels = np.zeros([len(filenames)], dtype=np.int32)

    for i, filename in enumerate(filenames):
        img = PIL.Image.open(os.path.join(dir_path, filename))
        # gray 스케일 변환
        img = img.convert("L")
        # 원본이 mnist와 맞지 않는다면 필요
        img = img.resize([28, 28])
        # PIL로 읽어오면 0~255까지의 정수. 0~1 사이의 실수로 변환.
        # 생략하면 소프트맥스 결과가 하나가 1이고 나머지는 모두 0. 확률 개념이 사라짐.
        # 255로 나누면 원본과 비교해서 오차가 있긴 하지만, 정말 의미 없는 수준이라서 무시해도 된다.
        img = np.uint8(img) / 255
        # 2차원을 mnist와 동일한 1차원으로 변환. np로 변환 후에 reshape 호출
        images[i] = img.reshape(-1)

        # 레이블. 이름의 맨 앞에 정답 있다.
        finds = filename.split('_')         # 0_003.png
        labels[i] = int(finds[0])           # 0

    return images, labels


def load_model(h5_path, dir_path):
    model = tf.keras.models.load_model(h5_path)

    mnist = input_data.read_data_sets('mnist')
    print('mnist :', model.evaluate(mnist.test.images, mnist.test.labels))

    # 파일로 변환한 mnist 숫자 이미지 파일 읽
    images, labels = read_testset(dir_path)
    print('files :', model.evaluate(images, labels))

    # 에뮬레이터 결과와 비교 목적
    preds = model.predict(images)
    print(preds)


load_model('./model/mnist.h5', './mnist/new_data')


# 파일 이미지 출력
def show_image_values(file_path):
    img = PIL.Image.open(file_path)

    img = img.convert("L")
    img = img.resize([28, 28])
    img = np.uint8(img) / 255

    print(img)


# pc와 에뮬레이터에 같은 파일을 넣고 실제 값 출력해서 비교.
# 똑같이 나왔다. 변환이 잘 되었다는 뜻.
show_image_values('./mnist/new_data/2_003.png')


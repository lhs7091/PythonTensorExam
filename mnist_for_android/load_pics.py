from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
# mnist 데이터셋 중에서 test 데이터셋 사용
# 'mnist' 폴더에 파일 다운로드
def get_dataset():
    mnist = input_data.read_data_sets('mnist')
    return mnist.test.images, mnist.test.labels
# 파일을 저장할 폴더 확인 후 생성
def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
# 파일 이름은 한 자리 숫자가 앞에 오고, 세 자리 일련번호가 뒤에 온다.
# 일련번호 최대 숫자는 999이기 때문에 1천장을 넘어가면 형식이 어긋난다.
def get_file_path(digit, order, dir_path):
    filename = '{}_{:03d}.png'.format(digit, order)
    return os.path.join(dir_path, filename)

# mnist의 test 데이터셋으로부터 지정한 개수만큼 파일로 변환해서 저장
def make_digit_images_from_mnist(count, dir_path):
    images, labels = get_dataset()
    make_directory(dir_path)
    if count > len(images):
        count = len(images)
    for i in range(count):
        file_path = get_file_path(labels[i], i, dir_path)
        # i는 일련번호 + 배열 인덱스
        plt.imsave(file_path, images[i].reshape(28, 28), cmap='gray')


make_digit_images_from_mnist(30, './mnist/new_data')


# 난수 샘플링 적용
def make_random_images_from_mnist(count, dir_path):
    images, labels = get_dataset()
    make_directory(dir_path)
    if count > len(images):
        count = len(images)
    # 중복되지 않는 인덱스 추출
    series = np.arange(len(images))
    indices = np.random.choice(series, count, replace=False)

    for i, idx in enumerate(indices):
        # i는 일련번호, idx는 배열의 인덱스
        file_path = get_file_path(labels[idx], i, dir_path)
        plt.imsave(file_path, images[idx].reshape(28, 28), cmap='gray')


make_random_images_from_mnist(30, './mnist/new_data')


# 숫자별로 지정한 개수만큼 동일하게 추출. 추출 개수는 digit_count * 10.
def make_random_images_from_mnist_by_digit(digit_count, dir_path):
    images, labels = get_dataset()
    make_directory(dir_path)

    if digit_count > len(images) // 10:
        digit_count = len(images) // 10

    # 동일 개수를 보장해야 하므로 전체 인덱스 필요
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        # 0~9까지 10개 key를 사용하고, value는 0에서 시작
        extracted_digit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        extracted_total = 0
        for idx in indices:
            digit = labels[idx]

            # 숫자별 최대 개수를 채웠다면, 처음으로.
            if extracted_digit[digit] >= digit_count:
                continue

            # 현재 숫자 증가
            extracted_digit[digit] += 1
            file_path = get_file_path(digit, extracted_total, dir_path)
            plt.imsave(file_path, images[idx].reshape(28, 28), cmap='gray')

            # 추출 숫자 전체와 추출해야 할 개수 비교
            extracted_total += 1
            if extracted_total >= digit_count * 10:
                break


make_random_images_from_mnist_by_digit(digit_count=2, dir_path='./mnist/new_data')
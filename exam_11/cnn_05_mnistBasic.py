import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/lhs/PycharmProjects/PythonTensorExam/exam_11/mist_data", one_hot=True)

image = mnist.train.images[0].reshape(28,28)
"""
plt.imshow(image, cmap='gray')
plt.show()
"""

sess = tf.InteractiveSession()

image = image.reshape(-1,28,28,1) #(None, x, y, one color)
W = tf.Variable(tf.random_normal([3,3,1,5]), dtype=tf.float32) #(x, y, one color, 5 filters)
conv2d = tf.nn.conv2d(image, W, strides=[1,2,2,1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_image = conv2d.eval()
conv2d_image = np.swapaxes(conv2d_image, 0, 3)
for i, one_img in enumerate(conv2d_image):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')

pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(pool)
pool_image = pool.eval()
pool_image = np.swapaxes(pool_image, 0, 3)
for i, one_img in enumerate(pool_image):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')

plt.show()
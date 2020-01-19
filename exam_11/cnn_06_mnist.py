# MNIST and Vonvolutional Neural Network

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

start_time = time.time()
config = tf.ConfigProto(intra_op_parallelism_threads=2,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count = {'CPU': 2})

mnist = input_data.read_data_sets("/Users/lhs/PycharmProjects/PythonTensorExam/exam_11/mist_data", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input placeholder
X = tf.placeholder(tf.float32, [None, 784])
X_image = tf.reshape(X, [-1, 28, 28, 1]) # image 28*28 one color(channel)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 image shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01)) # 3*3 input_channel 1, output_channel 32
L1 = tf.nn.conv2d(X_image, W1, strides=[1,1,1,1], padding='SAME')
#print(L1) #-> Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.nn.relu(L1)
#print(L1) #->Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#print(L1) #->Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2 image shape=(?,14,14,32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# Final FC 7*7*64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2_flat, W3)+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print("start to training. please wait for a second")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost = c / total_batch

        print("Epoch : ", "%04d"%(epoch+1), "cost = ", "{:.9f}".format(avg_cost))

    print("Training is completed")

    # test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    print('작업 수행된 시간 : %f 초' % (time.time() - start_time))
    plt.show()


"""
Accuracy :  0.9876
1458.467590 초 = 24분 18초
1345.370280 초 = 22분 25초
1378.027277 초
"""

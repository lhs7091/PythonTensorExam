# Classifying diabetes(당뇨병)
import tensorflow as tf
import numpy as np

xy = np.loadtxt('/Users/lhs/PycharmProjects/PythonTensorExam/exam_05/databases.csv', delimiter=',', dtype=np.float)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholder for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W)+b)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# true if hypothesis>=0.5 else false
predicted = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}

    for step in range(10001):
        sess.run(train, feed_dict=feed)

        if step % 2000 == 0:
            print(step, sess.run(cost, feed_dict=feed))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print()
    print(h, "  ", c, "  ", a)

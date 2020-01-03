# data loading

import tensorflow as tf
import numpy as np

# for reproducibility
tf.set_random_seed(777)

xy = np.loadtxt('/Users/lhs/PycharmProjects/PythonTensorExam/exam_04/test-data.csv', delimiter=',', dtype=np.float)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
hypothesis = tf.matmul(X, W) + b

# simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in a graph.
sess.run(tf.global_variables_initializer())

# Set up feed_dict variables inside the loop.
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, "cost : ", cost_val, "\nPrediction:\n", hy_val)

# Ask my Score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 80, 96]]}))
print("Other score will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 100],[100,98,97]]}))

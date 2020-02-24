import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1.+tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

#cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)+tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5, else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initilaize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        _, cost_val, w_val = sess.run([optimizer, cost, W], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})

    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
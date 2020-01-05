# Softmax Classifier
import tensorflow as tf
import matplotlib.pyplot as plt


# for reproducibility
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", shape=[None, 4])
Y = tf.placeholder("float", shape=[None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activation
# softmax = exp(logits)/ reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cost_list = []
    step_list = []
    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
        cost_list.append(cost_val)
        step_list.append(step)
        if step % 200 == 0:
            print(step, cost_val)

    #plt.plot(cost_list, step_list)
    #plt.show()

    print('---------------------')
    # testing & one-hot coding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(sess.run(tf.argmax(a, 1)))

    print('---------------------')
    # testing & one-hot coding
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))
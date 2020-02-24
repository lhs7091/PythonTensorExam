# Softmax Classifier
import tensorflow as tf
import numpy as np

# for reproducibility
tf.set_random_seed(777)

xy = np.loadtxt('/Users/lhs/PycharmProjects/PythonTensorExam/exam_06/data-04-zoo.csv', delimiter=',', dtype=np.float)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
nb_classes = 7 # 0~7

#print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
#print(Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
#print(Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) # your expectation
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # compare your expectation with correct results
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

        if step % 200 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    # test
    test_data = np.array([[5]])
    print(test_data.shape)
    pred = sess.run(prediction, feed_dict={X:[[0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1]], Y: test_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip (pred, test_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
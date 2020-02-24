"""
NN for MNIST
3layers : 0.945

deep NN for MNIST
7layers : 0.933

if a layer is increased more and more, the result of accuracy becomes bad because of overfitting.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("/Users/lhs/PycharmProjects/PythonTensorExam/exam_10/training_image_set", one_hot=True)
nb_classes = 10

#MNIST data image of shape 28*28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([784,256]), name='weight1')
    b1 = tf.Variable(tf.random_normal([256]), name='bias')
    layer1 = tf.nn.relu(tf.matmul(X, W1)+b1)
    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
    b2 = tf.Variable(tf.random_normal([256]), name='bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Layer2", layer2)

with tf.name_scope("Layer3"):
    W3 = tf.Variable(tf.random_normal([256, 256]), name='weight3')
    b3 = tf.Variable(tf.random_normal([256]), name='bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("Layer3", layer3)

with tf.name_scope("Layer4"):
    W4 = tf.Variable(tf.random_normal([256, 256]), name='weight4')
    b4 = tf.Variable(tf.random_normal([256]), name='bias4')
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

    tf.summary.histogram("W4", W4)
    tf.summary.histogram("b4", b4)
    tf.summary.histogram("Layer4", layer4)

with tf.name_scope("Layer5"):
    W5 = tf.Variable(tf.random_normal([256, 256]), name='weight5')
    b5 = tf.Variable(tf.random_normal([256]), name='bias5')
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

    tf.summary.histogram("W5", W5)
    tf.summary.histogram("b5", b5)
    tf.summary.histogram("Layer5", layer5)

with tf.name_scope("Layer6"):
    W6 = tf.Variable(tf.random_normal([256, 256]), name='weight6')
    b6 = tf.Variable(tf.random_normal([256]), name='bias6')
    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

    tf.summary.histogram("W6", W6)
    tf.summary.histogram("b6", b6)
    tf.summary.histogram("Layer6", layer6)

with tf.name_scope("Layer7"):
    W7 = tf.Variable(tf.random_normal([256, nb_classes]), name='weight7')
    b7 = tf.Variable(tf.random_normal([nb_classes]), name='bias7')
    hypothesis = tf.matmul(layer6, W7) + b7

    tf.summary.histogram("W7", W7)
    tf.summary.histogram("b7", b7)
    tf.summary.histogram("Layer7", hypothesis)

with tf.name_scope("Cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    tf.summary.scalar("Cost", cost)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

#Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

with tf.name_scope("Accuracy"):
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples/batch_size)

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/Users/lhs/PycharmProjects/PythonTensorExam/exam_10/logs/ReLU_logs_r0_03")
    writer.add_graph(sess.graph)  # Show the graph

    # training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, summary, cost_val = sess.run([optimizer, merged_summary, cost], feed_dict={X:batch_xs, Y:batch_ys})
            writer.add_summary(summary, global_step=i)
            avg_cost+=cost_val/num_iterations

            print("Epch:{:04d}, cost{:.9f}".format(epoch+1, avg_cost))

    print("Learning Completed")

    # test the model using tet sets
    print("Accuracy : ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}),)

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}),)
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap="Greys", interpolation="nearest",)
    plt.show()

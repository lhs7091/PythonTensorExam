import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# check out information of mnist data
mnist = input_data.read_data_sets("./mist_data", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
nb_classes = 10


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout(keep_prob) rate is set 0.5 ~ 0.7
            # but on training, should be set 1
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.x = tf.placeholder(tf.float32, [None, 784])
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            self.y = tf.placeholder(tf.float32, [None, nb_classes])

            # Convolution Layer 1
            conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, padding='SAME')
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # Convolution Layer 2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, padding='SAME')
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # Convolution Layer 3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            # Pooling Layer 3
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2, padding='SAME')
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            # Fully Connected(Dense) Layer with Relu
            """
            Densely-connected layer class.
            This layer implements the operation: outputs = activation(inputs * kernel + bias) 
            Where activation is the activation function passed as the activation argument (if not None), 
            kernel is a weights matrix created by the layer, 
            and bias is a bias vector created by the layer (only if use_bias is True).
            """
            flat = tf.reshape(dropout3, [-1,4*4*128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # Logits(no activation) Layer: L5 final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.x:x_test, self.training:training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x:x_test, self.y:y_test, self.training:training})

    def train(self, x_test, y_test, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x:x_test, self.y:y_test, self.training:training})


# initialize
sess = tf.Session()

models = []
num_model = 2
for m in range(num_model):
    models.append(Model(sess, "model"+str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Start')

# Test my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models)) #a new array of given shape and type, filled with zeros.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models): #enumerate(iterable[, start]) -> iterator for index, value of iterable
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c/total_batch

    print('Epoch:', '%04d'%(epoch+1), 'cost:', avg_cost_list)

print('Learning Finish')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):
    print((m_idx+1), 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

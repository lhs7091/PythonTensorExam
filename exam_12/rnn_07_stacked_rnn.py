import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

tf.set_random_seed(777)

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length]) # x data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # y label

# one-hot encoding
x_one_hot = tf.one_hot(X, num_classes)
print(x_one_hot)

# Make a lstm cell with hidden size (each unit output vector size)
cell = rnn.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

# outputs = unfolding size X hidden size, state = hidden state
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)
print(outputs)

# (optional) softmax layer
x_for_soft = tf.reshape(outputs, [-1, rnn_hidden_size])
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(x_for_soft, softmax_w)+softmax_b

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, rnn_hidden_size])

# all weights are 1(equal weights)
weights = tf.ones([batch_size, sequence_length])

# activation function
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))


import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

tf.set_random_seed(777)

sample = ("if you want to build a ship, don't drum up people together to "
          "collect wood and don't assign them task and work , but rather "
          "teach them to long for the endless immensity of the sea")
idx2char = list(set(sample)) # idx → char
char2idx = {c:i for i, c in enumerate(idx2char)} # char → idx

x_data = []
y_data = []

# hyper parameter
dic_size = len(char2idx) # RNN input size(one hot size)
rnn_hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size(rnn or softmax, .. etc)
sequence_length = 10 # number of LSTM unfolding (unit#)
learning_rate = 0.1

for i in range(0, len(sample)-sequence_length):
    x_str = sample[i:i+sequence_length]
    y_str = sample[i+1:i+sequence_length+1]
    print(i, x_str, "→", y_str)

    x = [char2idx[c] for c in x_str] # x string to index
    y = [char2idx[c] for c in y_str] # y string to index

    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

X = tf.placeholder(tf.int32, [None, sequence_length]) # x data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # y label
x_one_hot = tf.one_hot(X, num_classes)
print(x_one_hot)

# Make a lstm cell with hidden size (each unit output vector size)
cell = rnn.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

# outputs = unfolding size X hidden size, state = hidden state
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)

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
with tf.name_scope("Cost"):
    mean_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights))
    tf.summary.scalar("mean_loss", mean_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)
    prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/Users/lhs/PycharmProjects/PythonTensorExam/exam_12/logs/xor_logs_r0_01")
    writer.add_graph(sess.graph) # show the graph
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results, summary = sess.run(
            [train_op, mean_loss, outputs, merged_summary], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=i)

        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([idx2char[t] for t in index]), l)

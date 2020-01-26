import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

tf.set_random_seed(777)

sample = "if you want you"
idx2char = list(set(sample)) # idx → char
char2idx = {c:i for i, c in enumerate(idx2char)} # char → idx

#print(idx2char) → ['i', 'a', 'y', ' ', 'w', 'o', 't', 'u', 'n', 'f']
#print(char2idx) → {'i': 0, 'a': 1, 'y': 2, ' ': 3, 'w': 4, 'o': 5, 't': 6, 'u': 7, 'n': 8, 'f': 9}

sample_idx = [char2idx[c] for c in sample] # char → idx
#print(sample_idx) → [5, 8, 1, 7, 2, 3, 1, 0, 9, 6, 4, 1, 7, 2, 3]
x_data = [sample_idx[:-1]]  # data sample(0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # data sample(1 ~ n) hello: ello

# hyper parameter
dic_size = len(char2idx) # RNN input size(one hot size)
rnn_hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size(rnn or softmax, .. etc)
batch_size = 1 # one sample data, one batch
sequence_length = len(sample)-1 # number of LSTM unfolding (unit#)
learning_rate = 0.1

x = tf.placeholder(tf.int32, [None, sequence_length]) # x data
y = tf.placeholder(tf.int32, [None, sequence_length]) # y label
x_one_hot = tf.one_hot(x, num_classes)

cell = rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        l, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})
        result = sess.run(prediction, feed_dict={x:x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        if i % 100 == 0:
            print(i, "loss:", l, "prediction:",''.join(result_str))




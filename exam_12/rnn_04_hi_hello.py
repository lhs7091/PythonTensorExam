from tensorflow.contrib import rnn as rnn
import tensorflow.contrib as contrib
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']

#Teach hello : hihell -> ihello
x_data = [[0,1,0,2,3,3]] #hihell
x_one_hot = [[[1,0,0,0,0],  #h 0
              [0,1,0,0,0],  #i 1
              [1,0,0,0,0],  #h 0
              [0,0,1,0,0],  #e 2
              [0,0,0,1,0],  #l 3
              [0,0,0,1,0]]] #l 3
y_data = [[1,0,2,3,3,4]] #ihello

# Define parameters
num_classes = 5 # 5 characters
input_dim = 5   # one_hot size
hidden_size = 5 # output from the LSTM. 5 to directly predict one_hot
batch_size = 1  # one sentence
sequence_length = 6 #|ihello| == 6
learning_rate = 0.1

x = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X one-hot
y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

# FC layer
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
"""
fc_w = tf.get_variable("tf_w", [hidden_size, num_classes])
fc_b = tf.get_variable("tf_b", [num_classes])
outputs = tf.matmul(x_for_fc, fc_w)+fc_b
"""
outputs = tf.contrib.layers.fully_connected(inputs=x_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = contrib.seq2seq.sequence_loss(logits=outputs, targets=y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={x:x_one_hot, y:y_data})
        result = sess.run(prediction, feed_dict={x:x_one_hot})
        print(i, "loss: ", l, "prediction: ", result, "Correct Value", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tprediction str: ", ''.join(result_str))

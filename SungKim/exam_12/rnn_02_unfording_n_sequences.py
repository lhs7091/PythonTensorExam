import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

hidden_size = 2
cell = rnn.BasicLSTMCell(num_units=hidden_size)
x_data = np.array([[h,e,l,l,o]], dtype=np.float32)
print(x_data.shape)
print()
pp.pprint(x_data)

output, state = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
print()
pp.pprint(output.eval())
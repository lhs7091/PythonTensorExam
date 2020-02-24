import tensorflow as tf
import numpy as np

# hyper parameters
timesteps = seq_length = 7 # 7days
data_dim = 5 # open, high, low, volume, close
output_dim = 1
hidden_dim = 5 # if you use fully connected befor you get output, you can define any values you want
learning_rate = 0.01


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# read data from csv file
file_path = "/Users/lhs/PycharmProjects/PythonTensorExam/exam_12/data-stock-daily.csv"
xy = np.loadtxt(file_path, delimiter=',')
xy = xy[::-1] # chronological order(ordering events in accordence with time sequence)
xy = MinMaxScaler(xy) # Normalization
x = xy
y = xy[:, [-1]]

dataX = []
dataY = []

for i in range(0, len(y)-seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    print(_x,'→',_y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and test
train_size = int(len(dataY)*0.7)
test_size = len(dataY)-train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input placeholder
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

from tensorflow.contrib import rnn
cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
from tensorflow.contrib import layers
Y_pred = layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None) # we use last cell's outputs

# cost/loss function
loss = tf.reduce_mean(tf.square(Y_pred-Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1001):
        _, l = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
        if i % 100 ==0:
            print(i, l)

    test_prediction = sess.run(Y_pred, feed_dict={X:testX})


import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(test_prediction)
plt.show()


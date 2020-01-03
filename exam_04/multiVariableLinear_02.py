import tensorflow as tf

x_data = [[73., 80., 75.],[93., 88., 93.],[89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# placeholder for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, W_val, b_val, _ = sess.run([cost, hypothesis, train, W, b], feed_dict={X: x_data, Y: y_data})
    if step % 200 ==0:
        print(step, "cost: ", cost_val, "\Prediction:\n", hy_val)
        print(W_val, b_val)



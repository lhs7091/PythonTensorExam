# output when W = 5.0

import tensorflow as tf

# tf graph input
x = [1,2,3]
y = [2,9,17]

# set wrong model rate
W = tf.Variable(-5.0)

# Linear model W * X
hypothesis = W * x

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-y))

# Minimize gradient descent magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a Session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
    if step % 10 == 0:
        print(step, sess.run(W))
        sess.run(train)
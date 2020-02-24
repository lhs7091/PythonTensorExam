# optional : compute_gradient and apply_gradient

import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]
# set wrong model weight
W = tf.Variable(5.)
#W = tf.Variable(tf.random_normal([1]))

# Linear model
hypothesis = W * X

# manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2
# cost/lost function
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)

# get gradient
gvs = optimizer.compute_gradients(cost)
# apply gradient
apply_gradients = optimizer.apply_gradients(gvs)

# Launch in a Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)

# Our Hypothesis for liner model X * W
hypothesis = X * W

# cost/loss function
# reduce_mean = average
# square = 제곱
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost,W],feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    if i%10 == 0:
        print(cost_val[len(cost_val)-1], W_val[len(W_val)-1])

# show the cost function
plt.plot(W_val, cost_val)
plt.show()

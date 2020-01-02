import tensorflow as tf

#X and y data
#x_train = [1,2,3]
#y_train = [3,5,7]

# we can use X and Y in place of x_train and y_train now
# placeholders for a tensor that will be always fed using feed_dict
# see http://stackoverflow.com/questions/36693740

# Variable -> tensorflow가 학습하는 변수(일반적인 언어에서 쓰는 변수와는 다른 개)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#Our hypothesis Wx+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Lounch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})

    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)


# Testing our model
print(sess.run(hypothesis, feed_dict={X:[7.2]}))

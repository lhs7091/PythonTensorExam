import tensorflow as tf

#X and y data
x_train = [1,2,3]
y_train = [3,5,7]

# Variable -> tensorflow가 학습하는 변수(일반적인 언어에서 쓰는 변수와는 다른 개)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our hypothesis Wx+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Lounch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

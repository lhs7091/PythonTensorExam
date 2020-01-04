Logistic Classification

# Classification : Binary(0, 1 encoding)
    Spam email detection : spam(1) or ham(0)
    facebook : show(1) or hide(0)
    credit card fraudulent transaction detection : legitimate(0)/fraud(1)

    we know Y is 0 or 1.
    H(x) = Wx + b,
    Hypothesis can give values more than 1 or less than 0
    so we need a function for making values between 0 to 1.
    Sigmoid : Curved in two directions like the letter "S" or the Greek ς(sigma)
    g(z) = 1/(1+e^-z)
    z = Wx

    H(x) = 1/(1+e^-Wx)

# Cost function and Gradient descent
    we want to find global minimum values but it may find local minimum values and stop.

    New cost function for logistic
    cost(W) = 1/m Σc(H(x),y)

    c(H(x),y) = -log(H(x)) y:1
              = -log(1-H(x)) y:0
    -> c(H(x),y) = -ylog(H(x))-(1-y)log(1-H(x))

# Minimize cost - Gradient descent algorithm
cost(W) = -1/mΣylog(H(x))-(1-y)log(1-H(x))

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# if you need data for logistic classification, please check folling website.
https://www.kaggle.com



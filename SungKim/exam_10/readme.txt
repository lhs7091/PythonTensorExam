BackPropagation(chain rule)
    if you principle more than 2, 3 layers such as w1, w2, w3, w4,....w9, wx,
    it will be converged to 0.
    then, the result of deep learning will be fault or error.
        -> vanishing gradient
    in this case, we can use ReLU instead of XSigmoid

Weight Initialization
    Need to set the initial weight values wisely
        not all 0's
        challenging issue
        "A fast learning algorithm for deep belief nets" -> restricted boatman machine(RBM)
    How can we use RBM to initialize weight?
        Apply the RBM idea on adjacent two layers as a pre-training step
        Continues the first process to all layers
        This will set weights
        Example Deep Belief Network -> Weight initialized by RBM
    Good news
        No need to use complicated RBM for weight initializations
        Simple methods are OK
            Xavier initialization(2010)
            he's initialization(2015)
                Make sure the weigths are 'Just Right', not too small, not too big
                Using number of input(fan_in) and output(fan_out)
                W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) (2010)
                W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2) (2015)

Dropout
    Am I overfitting?
        Very hight accuracy on the training data set (eg:0.99)
        Poor accuracy on the training data set (eg:0.85)
    Solution for overfitting
        More training data
        Reduce the number of features
        Regularization
            Let's not have too big numbers in the weight
            l2reg = 0.001 * tf.reduce_sum(tf.square(W))
Dropout : A simple way to prevent neural networks from overfitting
    Tensorflow implementation
    dropout_rate = tf.placeholder(“float”)
    _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
    B1 = tf.nn.dropout(_L1, dropout_rate)

    Train:
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})

    Evaluation:
        print(“Accuracy: ”, accuracy,eval({X:mnist.test.images, Y:mnist.test.labels, dropout_rate:1}))

Ensemble
    Instead of using a single neural network, we use several neural networks and average their outputs.
    The ensemble improves the classification accuracy slightly on the test-set, but the difference is so small that it is possibly random.
    Furthermore, the ensemble mis-classifies some images that are correctly classified by some of the individual networks.


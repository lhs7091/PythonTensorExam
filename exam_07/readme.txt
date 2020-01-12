learning rate : overshooting

small learning rate : takes too long, stops at local minimum
big learning rate : diverge

Try several learning rate
    observe the cost function
    check it goes down in a reasonable rate

Data preprocessing


Over fitting
    our model is very good with training data set(with memorization)
    not good at test data set or in real use

solutions of Over fitting
    More training data
    Reduce number of the features
    Regularization : let's not have too big numbers in the weight


Evaluation using training set?
    30% of fore side and 70% of test set in one model
    So training 20% of fore side and input a number or a case at 70% of test set in one model
    you've already known how values or result you get
    so you can decide that is good model or not

Online learning


Training each/batch
In the neural network terminology:
    One epoch = one forward pass and one backward pass of all the training examples
    Batch size = the number of training examples in one forward/backward pass, The higher the batch size, the more memory space you’ll need.
    Number of iterations = number of passes, each pass using [batch size] number of examples.
        To be clear, one pass = one forward pass + one backward pass(we do not count the forward pass and backward pass as two different passes).

    For example, if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[2]:


# for reproducibility
tf.set_random_seed(777)

train_data_set = pd.read_csv('./train.csv')
for i in range(len(train_data_set)):
    if train_data_set["Sex"][i]=="male":
        train_data_set["Sex"][i]=0
    else:
        train_data_set["Sex"][i]=1

x_data = train_data_set[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]
y_data = train_data_set["Survived"]


# In[3]:


x_data = np.array(x_data, dtype=np.float)
y_data = np.array(y_data, dtype=np.float)
y_data = y_data.reshape(891,1)


# In[4]:


X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# In[5]:


# parameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100
total_batch = int(len(x_data) / batch_size)


# In[6]:


keep_prob = tf.placeholder(tf.float32)


# In[7]:


W1 = tf.get_variable("W1", shape=[5, 5],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([5]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[5, 5],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([5]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[5, 5],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([5]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[5, 5], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([5]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[5, 2],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L4, W5) + b5


# In[8]:


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.squared_difference(Y, hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1) # your expectation
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1)) # compare your expectation with correct results
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[9]:


# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[10]:


for step in range(20001):
    _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data, keep_prob:1})

    if step % 2000 == 0:
        print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

print('Learning Finished!')


# In[11]:


test_data_set = pd.read_csv('./test.csv')
for i in range(len(test_data_set)):
    if test_data_set["Sex"][i]=="male":
        test_data_set["Sex"][i]=0
    else:
        test_data_set["Sex"][i]=1
test_data = test_data_set[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]


# In[12]:


test_data = np.array(test_data, dtype=np.float)
test_data.shape


# In[13]:


result = np.loadtxt('./gender_submission.csv', delimiter=',', dtype=np.int)
result = result[:, [-1]]
result


# In[14]:


pred = sess.run(prediction, feed_dict={X:test_data, keep_prob:1})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
count = 0
for p, r in zip (pred, result.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == r, p, r))
    if p==r:
        count+=1
print('true',count)
print("Acc: {:.2%}".format(count/418))


# In[ ]:





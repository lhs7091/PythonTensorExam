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


x_data


# In[4]:


y_data


# In[5]:


X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.int32, shape=[None, 1])


# In[6]:


Y_one_hot = tf.one_hot(Y, 2)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 2])


# In[7]:


W = tf.Variable(tf.random_normal([5, 2]), name='weight')
b = tf.Variable(tf.random_normal([2]), name='bias')


# In[8]:


logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)


# In[9]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# In[10]:


prediction = tf.argmax(hypothesis, 1) # your expectation
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # compare your expectation with correct results
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[11]:


x_data = np.array(x_data, dtype=np.float)


# In[12]:


y_data = np.array(y_data, dtype=np.float)
y_data = y_data.reshape(891,1)


# In[13]:


x_data.shape


# In[14]:


y_data.shape


# In[15]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

    if step % 2000 == 0:
        print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))


# In[16]:


pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


# In[17]:


test_data_set = pd.read_csv('./test.csv')
for i in range(len(test_data_set)):
    if test_data_set["Sex"][i]=="male":
        test_data_set["Sex"][i]=0
    else:
        test_data_set["Sex"][i]=1
test_data = test_data_set[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]


# In[18]:


test_data = np.array(test_data, dtype=np.float)
test_data.shape


# In[19]:


result = np.loadtxt('./gender_submission.csv', delimiter=',', dtype=np.int)
result = result[:, [-1]]


# In[20]:


result


# In[21]:


pred = sess.run(prediction, feed_dict={X:test_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
count = 0
for p, r in zip (pred, result.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == r, p, r))
    if p==r:
        count+=1
print('true',count)
print("Acc: {:.2%}".format(count/418))


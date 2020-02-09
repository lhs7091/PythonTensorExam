import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

del train['Name']
del train['Ticket']
del train['Fare']
del train['Embarked']

train = train.fillna(value=0.0)

for i in range(train.shape[0]):
    if train.at[i, 'Sex'] == 'male':
        train.at[i, 'Sex'] = 1
    else:
        train.at[i, 'Sex'] = 0

train['Age_group'] = 0
for i in range(train.shape[0]):
    for j in range(70, 0, -10):
        if train.at[i, 'Age'] > j:
            train.at[i, 'Age_group'] = int(j/10)
            break
del train['Age'] # it's unnecessary anymore

print(list(set(train['Cabin'].values))[:10]) # sample of 'Cabin' values
train['Cabin_section'] = '0'
for i in range(train.shape[0]):
    if train.at[i, 'Cabin'] != 0:
        train.at[i, 'Cabin_section'] = train.at[i, 'Cabin'][0]
CABIN_SECTION = list(set(train['Cabin_section'].values)) # will be reused for test data
print(CABIN_SECTION) # 'Cabin_Section' values
for i in range(train.shape[0]):
    train.at[i, 'Cabin_section'] = CABIN_SECTION.index(train.at[i, 'Cabin_section'])
del train['Cabin'] # it's unnecessary anymore

pclass = np.eye(train['Pclass'].values.max()+1)[train['Pclass'].values]
age_group = np.eye(train['Age_group'].values.max()+1)[train['Age_group'].values]
cabin_section = np.eye(train['Cabin_section'].values.max()+1) \
                    [train['Cabin_section'].values.astype(int)] # prevent IndexError

X = train[['Sex', 'SibSp', 'Parch']].values
X = np.concatenate([X, age_group], axis=1)
X = np.concatenate([X, pclass], axis=1)
X = np.concatenate([X, cabin_section], axis=1)
X = X.astype(float)

y = train['Survived'].values
y = y.astype(float).reshape(-1, 1)

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=0)

del test['Name']
del test['Ticket']
del test['Fare']
del test['Embarked']

test = test.fillna(value=0.0)

test['Age_group'] = 0
test['Cabin_section'] = '0'
for i in range(test.shape[0]):
    if test.at[i, 'Sex'] == 'male':
        test.at[i, 'Sex'] = 1
    else:
        test.at[i, 'Sex'] = 0

    for j in range(70, 0, -10):
        if test.at[i, 'Age'] > j:
            test.at[i, 'Age_group'] = int(j/10)
            break

    if test.at[i, 'Cabin'] != 0:
        test.at[i, 'Cabin_section'] = test.at[i, 'Cabin'][0]
    test.at[i, 'Cabin_section'] = CABIN_SECTION.index(test.at[i, 'Cabin_section'])

del test['Cabin'] # it's unnecessary anymore
del test['Age'] # it's unnecessary anymore

pclass_test = np.eye(test['Pclass'].values.max()+1)[test['Pclass'].values]
age_group_test = np.eye(test['Age_group'].values.max()+1)[test['Age_group'].values]
cabin_section_test = np.eye(test['Cabin_section'].values.max()+1) \
                    [test['Cabin_section'].values.astype(int)] # prevent IndexError

X_test = test[['Sex', 'SibSp', 'Parch']].values
X_test = np.concatenate([X_test, age_group_test], axis=1)
X_test = np.concatenate([X_test, pclass_test], axis=1)
X_test = np.concatenate([X_test, cabin_section_test], axis=1)
X_test = X_test.astype(float)

id_test = test['PassengerId'].values
id_test = id_test.reshape(-1, 1)

seed = 7 # for reproducible purpose
input_size = X_train.shape[1] # number of features
learning_rate = 0.001 # most common value for Adam
epochs = 8500 # I've tested previously that this is the best epochs to avoid overfitting

graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(seed)
    np.random.seed(seed)

    X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')

    #W1 = tf.Variable(tf.random_normal(shape=[input_size, input_size], seed=seed), name='W1')
    W1 = tf.get_variable("W1", shape=[input_size, input_size], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal(shape=[input_size], seed=seed), name='b1')
    L1 = tf.add(tf.matmul(X_input, W1), b1)
    L1 = tf.nn.dropout(L1, keep_prob=0.7)

    W2 = tf.get_variable("W2", shape=[input_size, input_size], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal(shape=[input_size], seed=seed), name='b2')
    L2 = tf.add(tf.matmul(L1, W2), b2)
    L2 = tf.nn.dropout(L1, keep_prob=0.7)

    W3 = tf.get_variable("W3", shape=[input_size, input_size], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal(shape=[input_size], seed=seed), name='b3')
    L3 = tf.add(tf.matmul(L2, W3), b3)
    L3 = tf.nn.dropout(L1, keep_prob=0.7)

    #W2 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W2')
    W4 = tf.get_variable("W4", shape=[input_size, 1], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b4')
    sigm = tf.nn.sigmoid(tf.add(tf.matmul(L3, W4), b4), name='pred')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                  logits=sigm, name='loss'))
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred')  # 1 if >= 0.5
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')

    init_var = tf.global_variables_initializer()
train_feed_dict = {X_input: X_train, y_input: y_train}
dev_feed_dict = {X_input: X_dev, y_input: y_dev}
test_feed_dict = {X_input: X_test}  # no y_input since the goal is to predict it

sess = tf.Session(graph=graph)
sess.run(init_var)
cur_loss = sess.run(loss, feed_dict=train_feed_dict)
train_acc = sess.run(acc, feed_dict=train_feed_dict)
test_acc = sess.run(acc, feed_dict=dev_feed_dict)
print('step 0: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc))
for step in range(1, epochs+1):
    sess.run(train_steps, feed_dict=train_feed_dict)
    cur_loss = sess.run(loss, feed_dict=train_feed_dict)
    train_acc = sess.run(acc, feed_dict=train_feed_dict)
    test_acc = sess.run(acc, feed_dict=dev_feed_dict)
    if step%100 != 0: # print result every 100 steps
        continue
    print('step {3}: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       cur_loss, 100*train_acc, 100*test_acc, step))
'''
step 8500: loss 0.63441, train_acc 79.78%, test_acc 78.89%
L1 step 8500: loss 0.62886, train_acc 80.77%, test_acc 75.56%
initial value step 8500: loss 0.62095, train_acc 82.77%, test_acc 80.00%

initial value
'''

y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
                          columns=['PassengerId', 'Survived'])

print(prediction.head())

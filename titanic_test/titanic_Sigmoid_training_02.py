import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# set the dataset with train, test data
train_data_test = [train, test]

'''
separate Mr, Miss, Mrs from name
and delete name because it's string type
'''
for dataset in train_data_test:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3,
                 "Rev":3, "Col":3, "Major":3, "Mile":3, "Countess":3, "Capt":3, "Ms":3,
                 "Jonkheer":3, "Sir":3, "Don":3, "Mme":3,
                 "Lady":3}

for dataset in train_data_test:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'].fillna(value=3.0, inplace=True)

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

'''
change Sex string value to int type 
'''

sex_mapping={"male":0, "female":1}
for dataset in train_data_test:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

'''
fill with int type value in age, null
'''
train["Age"].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test["Age"].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)

'''
fill out missing embarked with s embark
because almost people get in from S embarked
and change int type for training
'''
for dataset in train_data_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embark_mapping={"S":0, "C":1, "Q":2}
for dataset in train_data_test:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)

'''
fill out missing cabin columns 
and change int type for training
fill missing Fare with median fare for each Pclass
'''
cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2, "G":2.4, "T":2.8}
for dataset in train_data_test:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train['Cabin'].fillna(value=0.0, inplace=True)
test['Cabin'].fillna(value=0.0, inplace=True)

'''
fill out missing Fare columns
it will be input average of Fare as Pclass
'''
test['Fare'].fillna(value=0, inplace=True)

'''
Define Dataset of train and test
'''
training_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Title']
X_train = train[training_columns].astype(float)
X_train = np.array(X_train)
X_train = X_train.reshape(891,9)

Y_train = train['Survived'].values
Y_train = Y_train.astype(float).reshape(-1, 1)

test_train = test[training_columns].values
test_train = np.array(test_train)
test_train = test_train.reshape(418,9)

'''
Training
'''
#test_train = test_train
print(X_train.shape)
print(Y_train.shape)
print(test_train.shape)

seed = 7 # for reproducible purpose
input_size = 9 # number of features
learning_rate = 0.001 # most common value for Adam
epochs = 10000 # I've tested previously that this is the best epochs to avoid overfitting

#X_train, X_dev, y_train, y_dev = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)

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
train_feed_dict = {X_input: X_train, y_input: Y_train}
#dev_feed_dict = {X_input: X_dev, y_input: y_dev}
test_feed_dict = {X_input: test_train}  # no y_input since the goal is to predict it

sess = tf.Session(graph=graph)
sess.run(init_var)
cur_loss = sess.run(loss, feed_dict=train_feed_dict)
train_acc = sess.run(acc, feed_dict=train_feed_dict)

print('step 0: loss {0:.5f}, train_acc {1:.2f}%'.format(cur_loss, 100*train_acc))
for step in range(1, epochs+1):
    sess.run(train_steps, feed_dict=train_feed_dict)
    cur_loss = sess.run(loss, feed_dict=train_feed_dict)
    train_acc = sess.run(acc, feed_dict=train_feed_dict)
    if step%100 != 0: # print result every 100 steps
        continue
    print('step {0}: loss {1:.5f}, train_acc {2:.2f}%'.format(step, cur_loss, 100*train_acc))


'''
Save Test result (PassengerId, predicted Survived)
'''
id_test = test['PassengerId'].values
id_test = id_test.reshape(-1, 1)

y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)

prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
                          columns=['PassengerId', 'Survived'])
prediction.to_csv('./submission.csv', index=False)

print(prediction.head())

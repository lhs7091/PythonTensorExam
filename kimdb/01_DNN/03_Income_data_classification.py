'''
please download dataset for keras
https://archive.ics.uci.edu/ml/index.php
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *
import seaborn as sns
pd.options.display.max_columns = 15

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status','occupation', 'relationship',
 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '5k']

quantitative = ['age', 'fnlwgt', 'education-num','capital-gain', 'capital-loss', 'hours-per-week']

df = pd.read_csv('./data/adult.data', names=names, index_col=False)

Y = df['5k'].values.tolist()
Y = [1 if i == ' <=50K' else 0 for i in Y]
Y = to_categorical(Y)
#print(Y)

# separate quantitative and qualitative values
X = df.drop(quantitative, axis=1)
X = X.drop(['5k'], axis=1)

# encoding by one-hot
X = pd.get_dummies(X, drop_first=True)
X = pd.concat([X, df[quantitative]], axis=1)

# in case of age, fnlwgt, number is much bigger than one-hot encoding
# so we may need normalization
scaler = MinMaxScaler()

X[quantitative] = scaler.fit_transform(X[quantitative])

# divide train set and test set by train data
X_train = X[:-1000]
X_test = X[-1000:]

Y_train = Y[:-1000]
Y_test = Y[-1000:]
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# modeling
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(100,)))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

# define optimizer and error fuction
# metrics : calculate accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# training
# validation_split = 0.1 -> 0.1%는 back propogation 중 test실시해서 calculate accuracy
hist = model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

print(hist.history)

# check overfitting
'''
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(hist.history['acc'], color='r')
plt.plot(hist.history['val_acc'], color='b')
plt.title('acc')

plt.subplot(1,2,2)
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')
'''

# evaluation and prediction
score = model.evaluate(X_test, Y_test)
print(score)

pred = model.predict(X_test)
print(pred[:10])
print(Y_test[:10])


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *
import seaborn as sns
pd.options.display.max_columns = 5

names = ['sl', 'sw', 'pl', 'pw', 'class']
df = pd.read_csv('./data/iris.data', names=names, index_col=False)

print(df.head())
print(df.info())

# one-hot encoding on class
Y = LabelEncoder().fit_transform(df['class'])
Y = to_categorical(Y)
print(Y)

X = df.drop('class', axis=1)

# set training data and test data
X_train = X[:-5]
X_test = X[-5:]

Y_train = Y[:-5]
Y_test = Y[-5:]

# define model
# the more we use dense layer, the deeper we can do training -> deep learning
model = Sequential()
model.add(Dense(256, input_shape=(4,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())

# define optimizer and Error Function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, epochs=60, validation_split=0.1)


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')

plt.subplot(1,2,2)
plt.plot(hist.history['acc'], color='r')
plt.plot(hist.history['val_acc'], color='b')
plt.title('acc')

# evaluation
score = model.evaluate(X_test, Y_test)

print(score)

# prediction
pred = model.predict(X_test)
print(pred)
print(Y_test)
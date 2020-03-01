import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *

label = ['feat.{}'.format(i) for i in range(73)]

df = pd.read_csv('./data/eighthr.data', names=label)

# object type -> float type
df = df.apply(pd.to_numeric, errors='coerce')

# remove nan values
df.dropna(inplcea=True)

Y = df['feat.72']
Y = to_categorical(Y)

df.drop(['feat.72'], axis=1, inplace=True)

# training / test set
X_train = np.asarray(df[:-100].values.tolist(), dtype=np.float64)
X_test = np.asarray(df[-100:].values.tolist(), dtype=np.float64)

y_train = Y[:-100]
y_test = Y[-100:]

X_train = X_train[:1700]
y_train = y_train[:1700]

# (sampling, timestamp, feature)
X_train = X_train.reshape(-1, 10, 72)
y_train = y_train.reshape(-1, 10, 2)
X_test = X_test.reshape(-1, 10, 72)
y_test = y_test.reshape(-1, 10, 2)

# modeling
model = Sequential()
model.add(LSTM(128, input_shape=(10,72), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(2, activation='softmax'))

# compling
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# training
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

# evaluation
score = model.evaluate(X_test, y_test)
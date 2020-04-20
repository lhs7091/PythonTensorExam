import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
plt.style.use('bmh')
pd.set_option('display.max_columns', None)

# https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast

df = pd.read_csv('./data/Bias_correction_ucl.csv', date_parser='Date')
X = df.drop(columns=['Date','Next_Tmax', 'Next_Tmin'])
y = df[['Next_Tmax', 'Next_Tmin']]

X = np.asarray(X)
y = np.asarray(y)

# (7752, 23),(7752, 2) -> ((7752, 23, 1),(7752, 2))
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modeling
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1],X.shape[2],)))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='linear'))

# loss/optimizer
model.compile(loss='mse', optimizer='adam')
model.summary()

# training
model.fit(X_train, y_train, epochs=10, batch_size=1)

# prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
plt.figure((7,7))
plt.plot(X_train[0], train_predict[0])
plt.show()

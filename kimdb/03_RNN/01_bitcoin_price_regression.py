import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py # for graph
import plotly.graph_objs as go # for graph
import requests
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
plt.style.use('bmh')

# bitcoin url
bitcoin_url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1405699200&end=9999999999&period=86400'

request = requests.get(bitcoin_url)
js = request.json()

df = pd.DataFrame(js)

# scaling
scaler = MinMaxScaler()
df[['close']] = scaler.fit_transform(df[['close']])
price = df['close'].values.tolist()
'''
Many to many

Many to one
X(5 prices by order)
so (sample, 5,1) (samples, timestamp, features)
 0.0034278368698297933,
 0.0032309273586996947,
 0.0034803460727978203,
 0.0029552540431175573,
 0.0029552540431175573,

Y(6th price)
 0.003204672757215681,
so (sample, 1) feautres

'''

window_size = 5
X = []
Y = []

for i in range(len(price)-window_size):
    X.append([price[i+j] for j in range(window_size)])
    Y.append(price[window_size+1])

X = np.asarray(X)
Y = np.asarray(Y)

train_test_split = 1000

X_train = X[:train_test_split, :]
y_train = Y[:train_test_split]
X_test = X[train_test_split:, :]
y_test = Y[train_test_split:]

# X_train.shape, y_train.shape
# ((1000, 5), (1000,)) -> ((1000, 5, 1), (1000, ))
X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
X_test = np.reshape(X_test, (X_test.shape[0], window_size, 1))

# modeling
model = Sequential()
model.add(LSTM(128, input_shape=(5,1,)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# loss/optimizer
model.compile(loss='mse', optimizer='adam')
model.summary()

# taining
model.fit(X_train, y_train, epochs=10, batch_size=1,)

#
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

plt.figure(figsize=(10,10))
plt.plot(price)

# prediction of X_train
split_pt = train_test_split + window_size
plt.plot(np.arange(window_size, split_pt, 1), train_predict, color='g')

# real values of X_train = X_test
plt.plot(np.arange(split_pt, split_pt+len(test_predict), 1), test_predict, color='r')

trace = go.Scatter(x=np.arange(window_size, split_pt,1), y=train_predict.reshape(1000), mode='lines', name='train')
trace2 = go.Scatter(x=np.arange(split_pt, split_pt+len(test_predict), 1), y=test_predict.reshape(85), mode='lines', name='test')
trace3 = go.Scatter(x=np.arange(1, len(price), 1), y=price, mode='lines', name='price')
data=[trace, trace2, trace3]
py.offline.plot(data)



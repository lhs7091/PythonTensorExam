import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

pd.options.display.max_columns = 15

dataset = load_boston()
columns = dataset.feature_names

X = pd.DataFrame(dataset.data, columns=columns)
y = pd.DataFrame(dataset.target, columns=['MEDV'])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define model
model = Sequential()
model.add(Dense(256, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

model.summary()

# define optimizer and Error Function
model.compile(loss='mse', optimizer='adam')
hist = model.fit(X_train, y_train, epochs=1000, validation_split=0.2)

plt.figure(figsize=(7,7))
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')
plt.show()

score = model.evaluate(X_test, y_test)
print(score)

pred = model.predict(X_test[-5:])
print(pred)
print(y_test[-5:])



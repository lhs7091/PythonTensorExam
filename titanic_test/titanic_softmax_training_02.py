#Acc: 67.46%

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_data_set = pd.read_csv('./train.csv')
for i in range(len(train_data_set)):
    if train_data_set["Sex"][i]=="male":
        train_data_set["Sex"][i]=0
    else:
        train_data_set["Sex"][i]=1

x_data = train_data_set[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]
y_data = train_data_set["Survived"].values.tolist()

y_data = to_categorical(y_data, )

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(891, activation='relu', input_shape=(5,)))
model.add(Dense(512, activation='relu', ))
model.add(Dense(256, activation='relu', ))
model.add(Dense(128, activation='relu', ))
model.add(Dense(32, activation='relu', ))
model.add(Dense(2, activation='softmax', ))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, y_train, epochs=500, validation_split=0.1)

score = model.evaluate(X_test, y_test)
print(score)

# np.argmax(list, axis=-1) -> [max,min] it gives 0, for [min,max] it gives 1
pred = model.predict(X_test,)
class_preds = np.argmax(pred, axis=-1)
class_y = np.argmax(y_test, axis=-1)

count = 0
for i in range(class_y.size):
    p = class_preds[i]
    r = class_y[i]
    if p == r:
        count += 1
print('true', count)
print("Acc: {:.2%}".format(count / class_y.size))

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(hist.history['acc'], color='r')
plt.plot(hist.history['val_acc'], color='b')
plt.title('acc')

plt.subplot(1,2,2)
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')
plt.show()

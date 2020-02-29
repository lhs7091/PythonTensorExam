import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test)  = cifar10.load_data()

#X_train.shape

fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(6,6,i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i])
#plt.show()

# Scaler
X_train = X_train/255.0
X_test = X_test/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modeling
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=4, padding='same', strides=1, activation='relu', input_shape=(32,32,3,)))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu', ))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu', ))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#loss/optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=150, epochs=1, validation_split=0.2)

score = model.evaluate(X_test, y_test)
print(score)

# prediction by X_test
pred = model.predict(X_test)

label = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(20,10))
for i, idx in enumerate(np.random.choice(X_test.shape[0], size=32)):
    ax = fig.add_subplot(4,8, i+1, xticks=[], yticks=[])
    ax.imshow(X_test[i])

    pred_idx = np.argmax(pred[idx])
    true_idx = np.argmax(y_test[idx])

    ax.set_title("{}_{}".format(label[pred_idx], label[true_idx]), color='green' if pred_idx == true_idx else 'red')

plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()

"""
2conv2d layers
in case of epoch = 1
- loss: 1.6249 - acc: 0.4066 - val_loss: 1.3907 - val_acc: 0.5040
score(accuracy) 0.4967

3conv2d layers
in case of epoch = 15
- loss: 0.1630 - acc: 0.9431 - val_loss: 1.5319 - val_acc: 0.6812
score = 0.6875

3conv2d layers and drop out
in case of epoch = 15
- loss: 0.7310 - acc: 0.7398 - val_loss: 0.7467 - val_acc: 0.7425
score = 0.7334

4conv2d layers and drop out
in case of epoch = 15
- loss: 0.7996 - acc: 0.7138 - val_loss: 0.8577 - val_acc: 0.7017
score = 0.6974

5conv2d layers and drop out and opt=sgd
in case of epoch = 15
 - loss: 1.8750 - acc: 0.3150 - val_loss: 1.8179 - val_acc: 0.3354
score = 0.3499

5conv2d layers and drop out 
and kernel_initializer='he_normal'
in case of epoch = 15
 - loss: 0.8080 - acc: 0.7099 - val_loss: 0.8043 - val_acc: 0.7118
score = 0.7029

5conv2d layers and drop out 
and kernel_initializer='he_normal'
in case of epoch = 100
 - loss: 0.4253 - acc: 0.8487 - val_loss: 0.7780 - val_acc: 0.7596
score = 0.7521 
 
"""



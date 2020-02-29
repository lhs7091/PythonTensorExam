#%matplotlib inline
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from keras import Model
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

plt.imshow(X_train[0], cmap='binary')
#plt.show()

# modeling
# now X_train shape is (60000, 28, 28)
# but we need (60000, 28, 28, 1) shape
# 60000-> pictures, 28->row, 28->columns, 1->color(black/white)

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaling
X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', strides=1, activation='relu', input_shape=(28,28,1,)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=36, kernel_size=(5,5), padding='valid', strides=1, activation='relu',))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# loss/optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=200, epochs=1, validation_split=0.2)

# evaluation my training
score = model.evaluate(X_test, y_test)
print(score)

# output the layer information of conv2d_1
L1 = model.get_layer('conv2d_1')
#print(L1.get_weights())


def plot_weight(w):
    w_min = np.min(w)
    w_max = np.max(w)

    num_grid = math.ceil(math.sqrt(w.shape[3]))

    fig, axis = plt.subplots(num_grid, num_grid)

    for i, ax in enumerate(axis.flat):
        if i < w.shape[3]:
            img = w[:,:,0,i]
            ax.imshow(img, vmin=w_min, vmax=w_max)

    plt.show()


'''L1 = model.get_layer('conv2d_1')
w1 = L1.get_weights()[0]
plot_weight(w1)'''

'''# load completed learning set
temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('conv2d_1').output)
output = temp_model.predict(X_test[:1])
print(output)'''


def plot_output(output):
    num_grid = math.ceil(math.sqrt(output.shape[3]))

    fix, axis = plt.subplots(num_grid, num_grid)

    for i, ax in enumerate(axis.flat):
        if i < output.shape[3]:
            img = output[0, :, :, i]
            ax.imshow(img, cmap='binary')

    plt.show()


# real prediction
temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('dense_2').output)
output = temp_model.predict(X_test)
r = random.randint(0, y_test.shape[0])
print('label:', np.argmax(y_test[r]))
print('prediction:', np.argmax(output[r]))

plt.imshow(X_test[r].reshape(28,28), cmap='binary')
plt.show()



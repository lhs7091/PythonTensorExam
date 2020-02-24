import matplotlib.pyplot as plt
import numpy as np
import warnings
from keras.layers import *
from keras.models import *
from keras.utils import *
from collections import Counter
# %matplotlib inline

# model data 생성
x = np.random.uniform(-np.pi, np.pi, 500)
y = np.random.uniform(-1, 1, 500)

'''plt.scatter(x,y)
plt.show()'''

# [x,y]의 형태로 행렬 생성
X = np.asarray([[x[i], y[i]] for i in range(500)])
print(X)
print(X.shape)

# basement line
sine = np.sin(x)
plt.scatter(x, sine)

# our data
'''plt.scatter(X[:,0], X[:,1])
plt.show()'''

# traget
Y = sine < X[:,1]
print(Y)

# check our target
'''plt.scatter(X[:,0][Y == 0], X[:,1][Y == 0])
plt.scatter(X[:,0][Y == 1], X[:,1][Y == 1])
plt.show()'''


model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd')

'''
training model
'''

# model.fit(X, Y, epochs=20)
# -> The Error is occured because of the shape
# now Y.shape is (1,0) but we need (2,0) Y is kind of type that [0.3, 0.7]
Y = to_categorical(Y)
print(Y)
model.fit(X, Y, epochs=1000)

# evaluate accuracy of our training
score = model.evaluate(X, Y)
print(score)

# prediction

a = [-2,0] # green
b = [2,0] # yello
pred_x = np.vstack((a,b))

print(pred_x)

prediction = model.predict(pred_x)
print(prediction)
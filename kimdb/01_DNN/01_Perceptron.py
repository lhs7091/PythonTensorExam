import matplotlib.pyplot as plt
import numpy as np
import warnings
from keras.layers import *
from keras.models import *
from keras.utils import *
from collections import Counter

warnings.filterwarnings('ignore')

x = np.linspace(1, 10, 1000)
y = 2*x+1

'''
Ready for training model(Perceptron)
Sequential() : initialize neural model
Dense() : hidden layer
    for example, Dense(10) means next the number of neural net is 10
'''
model = Sequential()

# Dense(output size, activation function, input shape(number of features))
model.add(Dense(1, activation='linear', input_shape=(1,)))

# for debugging model shape, parameter check
model.summary()
# param = 2 -> number of weight and bias

# what kind of error function we use? mean square eeee
# what kind of optimizer we use? gradient descent
model.compile(loss='mse', optimizer='sgd')

'''
training model
'''
# epochs=? -> how many training set
model.fit(x, y, epochs=20)

'''plt.plot(x,y)
plt.show()'''

# predict if x is 12, 14 then y values?
pred_x = []
pred_x = np.append(pred_x, 12)
pred_x = np.append(pred_x, 14)

pred_y = model.predict(pred_x)

print(pred_y)

plt.plot(x,y)
plt.scatter(pred_x, pred_y)
plt.show()

# check weight and bias
print(model.get_weights())
# [array([[2.0074224]], dtype=float32), array([0.94331545], dtype=float32)]
# weight = 2.007~~~, bias = 0.9433~~~


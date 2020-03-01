import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
import re

file = './data/text.txt'
text=''
with open(file) as f:
    lines = f.readlines()
    text = text.join([l for l in lines if re.match(r'^[A-Z].*:', l)])

array = []
for t in text.split('\n')[:1000]:
    array.append(t)

token = Tokenizer(lower=False, filters='.,?;"-_')
token.fit_on_texts(array)

# indexing of words
token.word_index

# change words to number by word_index
sequence = token.texts_to_sequences(array)

#
sequence = pad_sequences(sequence, maxlen=10)
X = sequence
Y = np.vstack((X[1:], X[0]))

# reshape
X = X.reshape(-1, 10, 1)
Y = Y.reshape(-1, 10, 1)

Y = to_categorical(Y)

# modeling
'''
direction of precessing
    -> -> ->
bidirectional(possible to reverse)
'''
model = Sequential()
model.add(Bidirectional(SimpleRNN(128, return_sequences=True), input_shape=(10,1)))
model.add(Bidirectional(SimpleRNN(128, return_sequences=True)))
model.add(Dense(1440, activation='softmax'))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
model.fit(X, Y, epochs=1, batch_size=1)

# model save
model.save('./data/temp_model')

# load
model = load_model('./data/temp_model')

# prediction
pred = model.predict(X[:1])
pred = np.argmax(pred, axis=2)

idx_word = {}
for w in token.word_index:
    idx_word[token.word_index[w]] = w

temp=''
for l in pred:
    for w in l:
        if w != 0:
            temp += idx_word[w]
            temp += ' '

for i in range(10):
    temp=''
    ran = random.randrange(0, len(X))
    pred = model.predict(np.expand_dims(X[ran], axis=0))

    pred = np.argmax(pred, axis=2)
    for l in pred:
        for w in l:
            if w != 0:
                temp += idx_word[w]
                temp += ' '
        temp += '\n'

    print(array[ran])
    print(temp)


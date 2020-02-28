import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df = pd.read_excel('./data/Concrete_Data.xls')
#print(df.head())
#print(df.columns)

# columns is too long to deal with datas
# so we change simply word for columns
df.rename(columns={
    'Cement (component 1)(kg in a m^3 mixture)': 'cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'blast',
       'Fly Ash (component 3)(kg in a m^3 mixture)': 'fly',
       'Water  (component 4)(kg in a m^3 mixture)': 'water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)' :'super',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coarse',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fine',
       'Age (day)':'age',
       'Concrete compressive strength(MPa, megapascals) ':'strength'
}, inplace=True)

#print(df.head())

# split the dataset X, y
X = df.drop(['strength'], axis=1)
y = df['strength']

# Scaling by minmax
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#print(X.shape)

#sns.pairplot(df)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(8,)))
model.add(Dense(128, activation='relu', ))
model.add(Dense(32, activation='relu', ))
model.add(Dense(1, activation='relu', ))

model.compile(loss='mse', optimizer='adam')
model.summary()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
hist = model.fit(X_train, y_train, epochs=1000, validation_split=0.1)

score = model.evaluate(X_test, y_test)
print(score)

pred = model.predict(X_test[-5:])
print(pred)
print(y_test[-5:])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')
plt.show()
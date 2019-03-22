import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt 
import numpy as np

seq = 10

x = np.arange(0, 6*np.pi, 0.01)
y = np.sin(x) + np.cos(x)*x

fig = plt.figure(1)
plt.plot(y, 'r')

train = np.array(y).astype(float)
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
data = []
for i in range(len(train) - seq - 1):
    data.append(train[i: i + seq + 1])
data = np.array(data).astype('float64')

x = data[:, :-1]
y = data[:, -1]
split = int(data.shape[0] * 0.5)

train_x = x[: split]
train_y = y[: split]

test_x = x  # [split:]
test_y = y  # [split:]

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

model = Sequential()
model.add(LSTM(input_dim=1, output_dim=6, return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')

model.fit(train_x, train_y, batch_size=50, nb_epoch=100, validation_split=0.1)
predict_y = model.predict(test_x)
predict_y = np.reshape(predict_y, (predict_y.size,))

predict_y = scaler.inverse_transform([[i] for i in predict_y])
test_y = scaler.inverse_transform(test_y)
fig2 = plt.figure(2)
plt.plot(predict_y, 'g')
plt.plot(test_y, 'r')
plt.show()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

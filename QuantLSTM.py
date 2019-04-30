from pandas import Series
from pandas import concat
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

# frame a sequence as a supervised learning problem


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]


def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # reshape seems redundant ?
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # add sequentail maodle
    model = Sequential()

    # first layer is lstm
    model.add(LSTM(neurons, batch_input_shape=(
        batch_size, X.shape[1], X.shape[2]), stateful=True))

    # then is dense layer
    model.add(Dense(1))

    # here is learinng start trigger, the third param is metrics  == accuracy on default
    # comile suppors mse by default just us mse
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # then for each part of datas use fit to learn
        # here epochs is time circle
        # batch_size is data sie in each iterate
        # validation_data to show diff and loss
        # we can use this to fit :
        # data = np.random.random((1000, 32))
        # labels = np.random.random((1000, 10))
        # val_data = np.random.random((100, 32))
        # val_labels = np.random.random((100, 10))
        # model.fit(data, labels, epochs=10, batch_size=32,
        #          validation_data=(val_data, val_labels))
        model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()
    return model
# accuturaly we can use  tf.keras.Model .evaluate and  tf.keras.model.predict to evalutae model and preidct datas
# for example:
#data = np.random.random((1000, 32))
#labels = np.random.random((1000, 10))
#model.evaluate(data, labels, batch_size=32)
#model.evaluate(dataset, steps=30)
#result = model.predict(data, batch_size=32)
#print(result.shape)

# make a one-step forecast


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]
## multi input test  todo  to validate
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
main_in = Input(shape = (100,), dtype = 'float32', name = 'price' )
sum_bus = Embedding(output_dim = 512, input_dim = 10000, input_length = 100)(main_in)
lstm_out = LSTM(32)(sum_bus)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
## multi input test end

series = read_csv('dahua.csv', header=0, parse_dates=[
                  0], index_col=0, date_parser=parser)
print("series head : \n")
print(series.head())

raw_values = series.values
# use open price as input to test
raw_values = raw_values[:, 0]
print(raw_values)

# use diff sequence such that each value is based on current state rather than relaies on the state before
diff_values = difference(raw_values, 1)
print(diff_values)

supervised_input = timeseries_to_supervised(diff_values, 1)
print(supervised_input)
supervised_input_values = supervised_input.values
print(supervised_input_values)

# todo this should refer to google instruct
train, test = supervised_input_values[0:-12], supervised_input_values[0: -12]

# do reshap and scale to [-1, 1]
print(train.shape[0])
print(train.shape[1])
train = train.reshape(train.shape[0], train.shape[1])
print(train)
scaler, train_scaled, test_scaled = scale(train, test)
print(train_scaled)

# build model
lstm_modle = fit_lstm(train_scaled, 1, 3000, 4)

# reshape to predict
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_modle.predict(train_reshaped, batch_size=1)

predictions = list()
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_modle, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) - i + 1)
    predictions.append(yhat)
    excepted = raw_values[len(train) + i + 1]
    print('Predicted = %f, Real = %f' % (yhat, excepted))

rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('RMSE : %.3f' % rmse)

pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()

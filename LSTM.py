from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('shampoo.csv', header=0, parse_dates=[
                  0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
# series.plot()
# pyplot.show()

X = series.values
train, test = X[0:-12], X[-12:]

history = [x for x in train]
print("\n history: \n")
print(history)

predictions = list()
print("\n list: \n")
print(history[-1])

for i in range(len(test)):
    predictions.append(history[-1])
    history.append(test[i])

print('predition: ')
print(predictions)

print('\nhistorys:')
print(history)

rmse = sqrt(mean_squared_error(test, predictions))

print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
# pyplot.plot(test)
# pyplot.plot(predictions)
# pyplot.show()


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    # axis = 1 add to right ignore index,  axis =0 add to after combine index
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


xxx = series.values
print(xxx)
supervised = timeseries_to_supervised(xxx, 1)
print(supervised.head())


def difference(dataSet, interval=1):
    diff = list()
    for i in range(interval, len(dataSet)):
        value = dataSet[i] - dataSet[i-interval]
        diff.append(value)
    return Series(diff)


def invers_difference(history, yhat, interval=1):
    return yhat + history[-interval]


diffed = difference(series, 1)
print(diffed.head())

inverted = list()

for i in range(len(diffed)):
    value = invers_difference(series, diffed[i], len(series) - i)
    inverted.append(value)

inverted = Series(inverted)
print(inverted.head())

X = series.values
X = X.reshape(len(X), 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scalered_X = scaler.transform(X)

print(scalered_X)

inverted_X = scaler.inverse_transform(scalered_X)
inverted_series = Series(inverted_X[:, 0])
print(inverted_series)
print(inverted_X)
# get 0th number in every row
scaled_series = Series(scalered_X[:,0])
print(scaled_series.head())

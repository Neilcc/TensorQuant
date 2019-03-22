import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tushare as ts

data = pd.read_csv('../data_stocks.csv')

data = data.drop(['DATE'], 1)

n = data.shape[0]
p = data.shape[1]
plt.show()

plt.plot(data['SP500'])
plt.draw()
data = data.values


print(ts.__version__)
dahua = ts.get_hist_data('002236', start='2019-02-01', end='2019-03-20')
dahua2 = {}
dahua2['open'] = dahua['open']
print dahua2
# filehua = open ('dahua.csv', 'w')
# filehua.write(dahua)
# filehua.close()

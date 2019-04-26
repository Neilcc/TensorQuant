from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

series = read_csv('dahua.csv', header = 0, parse_dates=[0],index_col=0, date_parser=parser)

print("series head : \n")

print(series.head())
series.plot()
pyplot.show()

x = series

print("series content: \n")
print(x)
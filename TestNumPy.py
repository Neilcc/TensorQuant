import numpy as np
import matplotlib.pyplot as mp
from pandas import read_csv

if __name__ == '__main__':

    mp.figure('pie', facecolor='lightgray')
    x = np.arange(10, 100)
    y = np.sin(x)
    mp.plot(x, y)
    mp.show()

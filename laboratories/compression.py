from scipy.fftpack import dct
import pandas as pd
import os
import sys

from laboratories import Interpolation


class Compression:
    def __init__(self):
        self.input = ['./output', 'out.csv']
        df = self.load_data()
        df = df[df != 'NaN'].dropna()
        a = dct(dct(df.T).T)
        print(max(abs(df-a)))
        # max(abs(df - a)) < 5 cm
        print(a)

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(*self.input), sep=" ", header=None)
        except OSError:
            print("Could not open/read file:", self.input[1])
            sys.exit()
        df.columns = ["Lat", "Long", "depth"]
        return df

# TODO:
# TODO: user input -> size of block of data
# TODO: user input -> decompression accuracy (abs)
#  after decompression this parameter is limit for error
# TODO: user input -> ZIP method yes or no
# TODO: time of computing
# TODO: progress bar
# TODO: compression ratio

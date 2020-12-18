import pandas as pd
import os
import sys

from scipy.fftpack import dct
from src.utils import get_project_root

root = get_project_root()


class Compression:
    def __init__(self, **kwargs):
        self.input = ("", "")
        setattr(self, 'input', kwargs.get('input'))
        print("| --------------------------------------------------------------- |")
        print("| -----------------------Compression process--------------------- |")
        print("| --------------------------------------------------------------- |")
        print(f"Reading data from: {root}/{self.input[0]}/{self.input[1]}")
        self.df = self.load_data()
        self.compress()

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(*self.input), sep=" ", header=None)
        except OSError:
            print("Could not open/read file:", self.input[1])
            sys.exit()
        df = df.dropna(axis=1)
        df.columns = ["Long", "Lat", "depth"]
        return df

    def compress(self):
        pass

# TODO:
# TODO: user input -> size of block of data
# TODO: user input -> decompression accuracy (abs)
#  after decompression this parameter is limit for error
# TODO: user input -> ZIP method yes or no
# TODO: time of computing
# TODO: progress bar
# TODO: compression ratio

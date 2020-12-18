import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import tqdm

from scipy.fftpack import dct, idct
from src.utils import get_project_root
from pathlib import Path

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
        df.columns = ["Long", "Lat", "depth"]
        return df

    @staticmethod
    def compress(grid, size_block, padding_h, padding_w, coords):
        h, w = grid.shape
        if h % size_block != 0:
            padding_h_zeros = np.zeros((h, padding_h))
            padding_h_zeros[:] = np.nan
            grid = np.hstack([grid, padding_h_zeros])
        if w % size_block != 0:
            padding_w_zeros = np.zeros((padding_w, w + padding_h))
            padding_w_zeros[:] = np.nan
            grid = np.vstack([grid, padding_w_zeros])
        h, w = grid.shape
        compression = np.zeros(grid.shape, dtype='float32')
        grid_ar = np.array(grid)
        for i in tqdm.tqdm(range(0, h, size_block), desc=coords):
            for j in range(0, w, size_block):
                compression[i:i + size_block, j:j + size_block] = dct(grid_ar[i:i + size_block, j:j + size_block])
        return compression

    @staticmethod
    def decompress(compressed, size_block, padding_h, padding_w, coords):
        decompression = np.zeros(compressed.shape, dtype='float32')
        h, w = compressed.shape
        for i in tqdm.tqdm(range(0, h, size_block), desc=coords):
            for j in range(0, w, size_block):
                decompression[i:i + size_block, j:j + size_block] = idct(compressed[i:i + size_block, j:j + size_block])
        if padding_h != 0 and padding_w != 0:
            decompression_new = decompression[:-padding_w, :-padding_h]
        elif padding_h != 0 and padding_w == 0:
            decompression_new = decompression[:-padding_w, :]
        elif padding_h == 0 and padding_w != 0:
            decompression_new = decompression[:, :-padding_h]
        else:
            decompression_new = decompression[:, :]
        return decompression_new

    def get_meshgrids(self, data):
        x_lim, y_lim = self.get_limits(data[:, 0], data[:, 1])
        xx, yy = np.meshgrid(np.arange(*x_lim), np.arange(*y_lim))
        zz = np.reshape(data[:, 2], yy.shape)
        return xx, yy, zz

    @staticmethod
    def get_limits(x, y):
        x_min, x_max = np.min(x), np.max(np.round(x, 2))
        step = x[1] - x[0]
        y_min, y_max = np.min(y), np.max(np.round(y, 2))
        return (x_min, x_max, step), (y_min, y_max, step)

# TODO:
# TODO: user input -> size of block of data
# TODO: user input -> decompression accuracy (abs)
#  after decompression this parameter is limit for error
# TODO: user input -> ZIP method yes or no
# TODO: time of computing
# TODO: progress bar
# TODO: compression ratio

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
        self.input = ["", ""]
        self.block_size = 0
        self.acc = 0.0
        self.to_zip = ''
        self.output = ["", ""]

        valid_keys = ["input", "block_size", "acc", "to_zip", "output"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        print("| --------------------------------------------------------------- |")
        print("| -----------------------Compression process--------------------- |")
        print("| --------------------------------------------------------------- |")
        print(f"Reading data from: {root}/{self.input[0]}/{self.input[1]}")
        self.df = self.load_data()
        self.to_zip = False if self.to_zip == 'N' else True

        data = self.df.to_numpy()

        xx, yy, zz = self.get_meshgrids(data)

        padding_h_x, padding_w_x = self.get_paddings(xx)
        padding_h_y, padding_w_y = self.get_paddings(yy)
        padding_h_z, padding_w_z = self.get_paddings(zz)

        xx_comp = self.compress(xx, self.block_size, padding_h_x, padding_w_x, 'X compression')
        yy_comp = self.compress(yy, self.block_size, padding_h_y, padding_w_y, 'Y compression')
        zz_comp = self.compress(zz, self.block_size, padding_h_z, padding_w_z, 'Z compression')

        xx_decomp = self.decompress(xx_comp, self.block_size, padding_h_x, padding_w_x, 'X decompression')
        yy_decomp = self.decompress(yy_comp, self.block_size, padding_h_y, padding_w_y, 'Y decompression')
        zz_decomp = self.decompress(zz_comp, self.block_size, padding_h_z, padding_w_z, 'Z decompression')

        z_abs = abs(zz - zz_decomp)

        df_kom = pd.DataFrame({'x': xx_comp.flatten(), 'y': yy_comp.flatten(), 'z': zz_comp.flatten()})

        if self.to_zip:
            df_kom.to_csv(os.path.join(*self.output, '.gz'), compression='gzip', sep=' ', header=False, index=False)
            cr_value = Path(*self.input).stat().st_size / Path(os.path.join(*self.output, '.gz')).stat().st_size
            print(f'Compression ratio CSV: {cr_value}')
        else:
            df_kom.to_csv(os.path.join(*self.output), sep=' ', header=False, index=False)
            cr_value = Path(*self.input).stat().st_size / Path(os.path.join(*self.output)).stat().st_size
            print(f'Compression ratio ZIP: {cr_value}')

        self.plot('Original', zz)
        self.plot('Compressed', zz_comp)
        self.plot('Decompressed', zz_decomp)

        self.plot('Errors zz', z_abs, 'magma')

        plt.show()

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(*self.input), names=["x", "y", "z"], sep='\s+')
        except OSError:
            print("Could not open/read file:", self.input[1])
            sys.exit()
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

    def get_paddings(self, aa):
        return self.block_size - (aa.shape[0] % self.block_size), self.block_size - (aa.shape[1] % self.block_size)

    def change_acc(self):
        # TODO: make accuracy level for comp/decomp
        pass

    @staticmethod
    def plot(title, dataset, c=None):
        plt.figure()
        plt.title(title)
        p = plt.imshow(dataset, cmap=c)
        plt.colorbar(p)

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


class Interpolation:
    def __init__(self, data_path="./data", filename='wraki utm.txt', **kwargs):
        self.df = self.load_data(data_path, filename)
        self.data = self.df.to_numpy()

        self.spacing = None
        self.is_square = None
        self.is_circle = None
        self.window_size = None
        self.num_min_points = None
        self.window_type = None

        valid_keys = ["spacing", "window_type", "window_size", "num_min_points"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        self.is_circle = True if self.window_type else False
        self.is_square = False if self.window_type else True

        self.grid = self.get_grid()
        self.zz = self.interp_moving_average()

        self.plot()

    def load_data(self, data_path, filename):
        try:
            df = pd.read_csv(os.path.join(data_path, filename), sep=" ", header=None)
        except OSError:
            print("Could not open/read file:", filename)
            sys.exit()
        df = df.dropna(axis=1)
        df.columns = ["Lat", "Long", "depth"]
        return df

    def interp_moving_average(self):
        if self.is_circle:
            tree = KDTree(self.data[:, :2])
            xx, yy = self.grid
            zz = np.empty((xx.shape[0], xx.shape[1]))
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    ids = tree.query_radius([[xx[i, j], yy[i, j]]], r=self.window_size)
                    zz[i, j] = np.nan if len(ids[0]) == 0 else np.mean(self.data[ids[0], 2])
                print_progress_bar(round((i / xx.shape[0])*100))
            print()
            return zz
        if self.is_square:
            pass

    def interp_idw(self):
        pass

    def interp_kriging(self):
        pass

    def get_grid(self, pos=False):
        mins, maxes = self.df.min(), self.df.max()
        x = np.arange(mins['Lat'], maxes['Lat'], self.spacing)
        y = np.arange(mins['Long'], maxes['Long'], self.spacing)
        mesh = np.meshgrid(x, y)
        return mesh if not pos else np.vstack(list(map(np.ravel, mesh)))

    def plot(self):
        fig = plt.figure()
        fig.suptitle('Interpolation')

        # First subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self.grid[0], self.grid[1], self.zz)

        ax = fig.add_subplot(1, 2, 2,)
        p = plt.imshow(self.zz)
        plt.colorbar(p)
        plt.show()


def print_progress_bar(iteration, total=100, prefix='Here', suffix='Now', decimals=0, length=50, fill='█', zfill='-'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    pbar = fill * filled_length + zfill * (length - filled_length)
    print('\r%s' % ('{0} |{1}| {2}% {3}'.format(prefix, pbar, percent, suffix)), end='')

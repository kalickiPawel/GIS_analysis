import pandas as pd
import numpy as np
import os
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
        self.plot(surf=True)

    def load_data(self, data_path, filename):
        df = pd.read_csv(os.path.join(data_path, filename), sep=" ", header=None)
        df = df.dropna(axis=1)
        df.columns = ["Lat", "Long", "depth"]
        return df

    def interp_moving_average(self):
        tree = KDTree(self.data[:, :2])
        xx, yy = self.grid
        zz = np.empty((xx.shape[0], xx.shape[1]))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                ids = tree.query_radius([[xx[i, j], yy[i, j]]], r=self.window_size)
                if len(ids[0]) == 0:
                    zz[i, j] = np.nan
                    continue
                zz[i, j] = np.mean(self.data[ids[0], 2])
        return zz

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

    def plot(self, surf=False):
        fig = plt.figure()
        if surf:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.grid[0], self.grid[1], self.zz)
        else:
            p = plt.imshow(self.zz)
            plt.colorbar(p)
        plt.show()

# TODO:
    # 7. -> progress bar

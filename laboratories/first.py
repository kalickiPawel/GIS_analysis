import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


class Interpolation:
    def __init__(self, data_path="./data", filename='wraki utm.txt', **kwargs):
        self.data = self.load_data(data_path, filename)

        self.spacing = None
        self.is_square = None
        self.is_circle = None
        self.window_size = None
        self.num_min_points = None

        valid_keys = ["spacing", "is_square", "is_circle", "window_size", "num_min_points"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        self.grid = self.get_grid()
        self.zz = self.interp_moving_average()
        xx, yy = self.grid
        self.plot_surf(xx, yy, self.zz)
        self.plot()

    def load_data(self, data_path, filename):
        df = pd.read_csv(os.path.join(data_path, filename), sep=" ", header=None)
        df = df.dropna(axis=1)
        df.columns = ["Lat", "Long", "depth"]
        return df

    def interp_moving_average(self):
        xy_data = self.data[['Lat', 'Long']].to_numpy()
        data = self.data[['Lat', 'Long', 'depth']].to_numpy()
        tree = KDTree(xy_data)
        radius_length = self.window_size
        xx, yy = self.grid
        zz = np.empty((xx.shape[0], xx.shape[1]))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                ids = tree.query_radius([[xx[i, j], yy[i, j]]], r=radius_length)
                if len(ids[0]) == 0:
                    zz[i, j] = np.nan
                    continue
                zz[i, j] = np.mean(data[ids[0], 2])
        return zz

    def interp_idw(self):
        pass

    def interp_kriging(self):
        pass

    def get_grid(self, pos=False):
        mins = self.data.min()
        maxes = self.data.max()
        x = np.arange(mins['Lat'], maxes['Lat'], self.spacing)
        y = np.arange(mins['Long'], maxes['Long'], self.spacing)
        mesh = np.meshgrid(x, y)
        return mesh if not pos else np.vstack(list(map(np.ravel, mesh)))

    def plot_surf(self, xx, yy, zz):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz)

    def plot(self):
        # 0 - 2d

        # plt.figure()
        # plt.plot(df['Lat'], df['Long'])
        # plt.show()

        plt.figure()
        p = plt.imshow(self.zz)
        plt.colorbar(p)
        plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(self.grid[0], self.grid[1], zz)
        # plt.show()

# TODO:
    # 7. -> progress bar
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


class Interpolation:
    def __init__(self, spacing=0.1, data_path="./data", filename='wraki utm.txt', window_type=0, window_size=0.1, num_min_points=2):
        self.window_type = window_type  # 0 -> square / 1 -> circle
        self.window = window_size if window_type else (window_size, window_size)
        self.spacing = spacing  # 0.1 -> wraki / 1 -> pozostałe
        self.num_min_points = num_min_points

        df = pd.read_csv(os.path.join(data_path, filename), sep=" ", header=None)
        df = df.dropna(axis=1)
        df.columns = ["Lat", "Long", "depth"]
        self.data = df
        self.grid = self.get_grid()
        self.grid_positions = self.get_grid(pos=True)
        self.grid_size = len(self.grid)
        self.z = self.interp_moving_average()

    def interp_moving_average(self):
        tree = KDTree(self.data[['Lat', 'Long']].to_numpy())
        radius_length = 0.4
        for i in range(self.grid.shape[2]):
            for j in range(self.grid.shape[1]):
                ids = tree.query_radius([[452606.36, 5967819.4]], r=radius_length)
                zz = 0
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
        return np.array(mesh) if not pos else np.vstack(list(map(np.ravel, mesh)))

    def plot(self, dim=0):
        # 0 - 2d

        # plt.figure()
        # plt.plot(df['Lat'], df['Long'])
        # plt.show()

        plt.figure()
        p = plt.imshow(self.z)
        plt.colorbar(p)
        plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(self.grid[0], self.grid[1], zz)
        # plt.show()

# TODO:
    # 7. -> progress bar

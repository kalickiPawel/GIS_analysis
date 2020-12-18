import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import numpy as np
import os
import sys

from src.utils import get_project_root
from pyproj import CRS
from sklearn.neighbors import KDTree

crs = CRS.from_user_input(4326)
root = get_project_root()


class Interpolation:

    def __init__(self, **kwargs):
        self.compress_only = None
        self.input = ("", "")

        self.spacing = None
        # 0.05 -> dla wraki
        # 0.5 -> dla reszty

        self.is_square = None
        self.is_circle = None
        self.window_size = None

        self.num_min_points = None
        self.window_type = None
        self.output = ("", "")
        self.save_format = None

        valid_keys = ["input", "spacing", "window_type", "window_size", 
                      "num_min_points", "output", "save_format", "compress_only"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        self.is_circle = False if self.window_type else True
        self.is_square = True if self.window_type else False

        print("| --------------------------------------------------------------- |")
        print(f"Reading data from: {root}/{self.input[0]}/{self.input[1]}")

        self.df, self.gdf = self.load_data()
        self.grid = self.get_grid()
        self.data = self.df.to_numpy()
        
        if not self.compress_only:
            self.zz = self.interp_moving_average()
            self.save()
            self.plot()

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(*self.input), sep=" ", header=None)
        except OSError:
            print("Could not open/read file:", self.input[1])
            sys.exit()
        df = df.dropna(axis=1)
        df.columns = ["Long", "Lat", "depth"]
        gdf = geopandas.GeoDataFrame(
            df.drop(columns=["Long", "Lat"]),
            geometry=geopandas.points_from_xy(df.Long, df.Lat),
            crs=crs
        )
        return df, gdf

    def get_grid(self):
        mins, maxes = self.df.min(), self.df.max()
        self.spacing = float(self.spacing)
        x = np.arange(mins['Long'], maxes['Long'], self.spacing)
        y = np.arange(mins['Lat'], maxes['Lat'], self.spacing)
        mesh = np.meshgrid(x, y)
        long_size = int(np.ceil((maxes['Long']-mins['Long'])/self.spacing))
        lat_size = int(np.ceil((maxes['Lat']-mins['Lat'])/self.spacing))
        print(f"Size of grid: {(long_size, lat_size)}")
        return mesh

    def plot(self):
        fig = plt.figure()
        fig.suptitle('Interpolation')

        # First subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self.grid[1], self.grid[0], self.zz)

        fig.add_subplot(1, 2, 2)
        p = plt.imshow(self.zz, cmap='jet')
        plt.colorbar(p)
        plt.show()

    def interp_moving_average(self):
        self.window_size = float(self.window_size)
        self.num_min_points = int(self.num_min_points)
        xx, yy = self.grid
        zz = np.empty((xx.shape[0], xx.shape[1]))
        if self.is_circle:
            tree = KDTree(self.data[:, :2])
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    ids = tree.query_radius([[xx[i, j], yy[i, j]]], r=self.window_size)
                    zz[i, j] = np.nan if len(ids[0]) == 0 else np.mean(self.data[ids[0], 2])
                print_progress_bar(round((i / xx.shape[0]) * 100))
            print()
            return zz

    def interp_idw(self):
        # TODO: implementing Inverse distance weighted.
        # TODO: with search as circle.
        # TODO: with search as square.
        pass

    def interp_kriging(self):
        # TODO: implementing Kriging
        # TODO: with search as circle.
        # TODO: with search as square.
        pass

    def save(self):
        if self.save_format == 'csv':
            if not os.path.exists(self.output[0]):
                os.makedirs(self.output[0])
            df = pd.DataFrame({'X': self.grid[0].flatten(), 'Y': self.grid[1].flatten(), 'Z': self.zz.flatten()})
            df.to_csv(os.path.join(*self.output), header=False, sep=' ', na_rep='NaN', index=True)

        if self.save_format == 'xyz':
            save_to_xyz_grid_ascii(self.grid[0], self.grid[1], self.zz, output=self.output)
        return 0


def print_progress_bar(iteration, total=100, prefix='Here', suffix='Now', decimals=0, length=50, fill='█', zfill='-'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    pbar = fill * filled_length + zfill * (length - filled_length)
    print('\r%s' % ('{0} |{1}| {2}% {3}'.format(prefix, pbar, percent, suffix)), end='')


def save_to_xyz_grid_ascii(x, y, z, output):
    # TODO: Save to standard ASCII Gridded XYZ
    pass

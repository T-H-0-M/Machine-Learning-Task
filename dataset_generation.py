
import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from numpy import pi


class Data_Gen():
    def __init__(self):
        self.dataset = None

    def get_two_spiral(self):
        data = pandas.read_csv('Datasets/spiralsdataset.csv',
                               header=None, names=['x1', 'x2', 'y'])
        x = data[['x1', 'x2']].to_numpy()
        y = data[['y']].to_numpy().flatten()
        return [x, y]

    def get_multi_spiral(self):
        data = pandas.read_csv('Datasets/multispiral.csv',
                               header=None, names=['x1', 'x2', 'y'])
        x = data[['x1', 'x2']].to_numpy()
        # x_norm = preprocessing.MinMaxScaler().fit_transform(x)
        y = data[['y']].to_numpy().flatten()
        return [x, y]

    def get_bupa(self):
        data = pandas.read_csv('Datasets/bupa.csv',
                               header=None, names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])

        x = data[['x1', 'x2', 'x3', 'x4', 'x5']].to_numpy()
        x_norm = preprocessing.MinMaxScaler().fit_transform(x)
        y = data[['y']].to_numpy().flatten()
        y = y.astype(np.int64)
        return [x_norm, y]

    # def generate_multispiral(self, num_spirals, num_points):

    #     theta = np.sqrt(np.random.rand(num_points)) * \
    #         2*pi  # np.linspace(0,2*pi,100)
    #     res_list = []

    #     for i in range(num_spirals):
    #         r = 2*theta + pi
    #         data = np.array([np.cos(theta)*r, np.sin(theta)*r]).T
    #         x = (data + np.random.randn(num_points, 2))
    #         res_list.append(np.append(x, np.full((num_points, 1), i), axis=1))

    #     res = np.append(res_list[0], res_list[1])

    #     for i in range(num_spirals - 2):
    #         res = np.append(res, res_list[i+2], axis=0)

    #     np.random.shuffle(res)
    #     np.savetxt("Datasets/multispiral.csv", res, delimiter=",",
    #                header="x,y,label", comments="", fmt='%.5f')

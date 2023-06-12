# Packaged with - main.py, neural_network.py, nn_training.py, svm.py
# Author - Thomas Bandy 
import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from numpy import pi


class Data_Gen():
    """This class returns data from three datasets stored in a csv file"""

    def __init__(self):
        self.dataset = None

    def get_two_spiral(self):
        """Returns the original dataset"""
        data = pandas.read_csv('Datasets/spiralsdataset.csv',
                               header=None, names=['x1', 'x2', 'y'])
        x = data[['x1', 'x2']].to_numpy()
        y = data[['y']].to_numpy().flatten()
        return [x, y]

    def get_multi_spiral(self):
        """Returns the self generated multispiral dataset"""
        data = pandas.read_csv('Datasets/multispiral.csv',
                               header=None, names=['x1', 'x2', 'y'])
        x = data[['x1', 'x2']].to_numpy()
        y = data[['y']].to_numpy().flatten()
        return [x, y.astype(np.int64)]

    def get_bupa(self):
        """Returns the bupa liver conditions dataset, with all features normalised"""
        data = pandas.read_csv('Datasets/bupa.csv',
                            header=None, names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
        x = data[['x1', 'x2', 'x3', 'x4', 'x5']].to_numpy()
        x_norm = preprocessing.MinMaxScaler().fit_transform(x)
        y = data[['y']].to_numpy().flatten()
        y = y.astype(np.int64)
        return [x_norm, y]


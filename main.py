# Packaged with -
# Author - Thomas Bandy (c3374048)
# Description:

import pandas

from nn_training import Train
from neural_network import ANN
from svm import SVM

#-----------------------------------------Importing Data-----------------------------------------#

data = pandas.read_csv('Datasets/spiralsdataset.csv',
                       header=None, names=['x1', 'x2', 'y'])
x = data[['x1', 'x2']].to_numpy()
y = data[['y']].to_numpy().flatten()

#-----------------------------------------Two Sprial FFNN-----------------------------------------#

model = ANN(input_nodes=2, layers=5, nodes_per_layer=22)
test = Train(x, y, model, 0.005)

test.solve(500, 5)


#-----------------------------------------Two Sprial SVM-----------------------------------------#
svm = SVM(x, y, "rbf", 8, 1)
svm.split_data(test_size=0.2, random_state=17)
svm.solve()
svm.generate_graph()
svm. export_results()

#-----------------------------------------Multi Sprial FFNN-----------------------------------------#


#-----------------------------------------Multi Sprial SVM-----------------------------------------#

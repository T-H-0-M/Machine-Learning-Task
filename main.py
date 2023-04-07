# Packaged with -
# Author - Thomas Bandy (c3374048)
# Description:

import time

from dataset_generation import Data_Gen
from nn_training import Train
from neural_network import ANN
from svm import SVM

start_time = time.time()

#-----------------------------------------Importing Data-----------------------------------------#
test = Data_Gen()
# x, y = test.get_multi_spiral()
# x, y = test.get_two_spiral()
x, y = test.get_bupa()
# print(type(y[1]))
# print(type(x[1][1]))



#-----------------------------------------Two Sprial FFNN-----------------------------------------#


model = ANN(input_nodes=5, layers=5, nodes_per_layer=160, output_nodes=2)
print(model)
test = Train(x, y, model, learn_rate=0.001)
test.solve(num_epochs=250, num_folds=10)
print(f"{time.time()-start_time} seconds")
# test.export_results()
# test.generate_learning_graph()


#-----------------------------------------Two Sprial SVM-----------------------------------------#
# svm = SVM(x, y, kernel="rbf", gamma=8, c=1)
# svm.split_data(test_size=0.2, random_state=17)
# svm.solve()
# # svm.generate_graph()
# svm. export_results()

#-----------------------------------------Multi Sprial FFNN-----------------------------------------#


#-----------------------------------------Multi Sprial SVM-----------------------------------------#


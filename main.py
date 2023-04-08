# Packaged with - nn_training.py, neural_network.py, dataset_generation.py, svm.py
# Author - Thomas Bandy (c3374048)

import time

from dataset_generation import Data_Gen
from nn_training import Train
from neural_network import ANN
from svm import SVM

start_time = time.time()

# Comment in the model and dataset you wish to use.
#-----------------------------------------Importing Data-----------------------------------------#
test = Data_Gen()
x, y = test.get_multi_spiral()
# x, y = test.get_two_spiral()
# x, y = test.get_bupa()




#-----------------------------------------Two Sprial FFNN-----------------------------------------#

# model = ANN(input_nodes=2, layers=4, nodes_per_layer=20, output_nodes=2)
# print(model)
# test = Train(x, y, model, learn_rate=0.001)
# test.solve(num_epochs=100, num_folds=4)
# print(f"{time.time()-start_time} seconds")
# test.export_results()
# test.generate_learning_graph()



#-----------------------------------------Two Sprial SVM-----------------------------------------#
svm = SVM(x, y, kernel="rbf", gamma=4, c=0.5)
svm.split_data(test_size=0.25, random_state=17)
svm.solve()
print(f"{time.time()-start_time} seconds")
svm. export_results()
svm.generate_graph()

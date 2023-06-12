# Packaged with - main.py, dataset_generation.py, nn_training.py, svm.py
# Author - Thomas Bandy

from torch.nn import Module, Linear, Softmax, ReLU, ParameterList, Sigmoid, CELU


class ANN(Module):
    """Inherits from the torch.nn.Module package to allow for the creation of a multi layered neural network"""

    def __init__(self, input_nodes, layers, nodes_per_layer, output_nodes):
        """Constructor takes in four params for the input nodes, number of layers, nodes per hidden layer and output nodes"""
        super().__init__()
        self.network_layers = ParameterList()
        self.layers = layers

        for i in range(self.layers):
            if i == 0:
                self.network_layers.append(
                    Linear(input_nodes, nodes_per_layer))
            elif i == (self.layers - 1):
                self.network_layers.append(
                    Linear(nodes_per_layer, output_nodes))
            else:
                self.network_layers.append(
                    Linear(nodes_per_layer, nodes_per_layer))
        self.network_layers.append(Softmax(dim=1))

    def forward(self, x):
        """Forward feeding functions. Iterates through the layers defined when an ANN is initialised. Called by Module"""
        for i in range(self.layers):
            if i == (self.layers - 2):
                x = self.network_layers[i](x)
            elif i == (self.layers - 1):
                x = self.network_layers[i](x)
            else:
                x = ReLU()(self.network_layers[i](x))
        return x

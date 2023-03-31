# Packaged with - main.py
# Author - Thomas Bandy (c3374048)
# Description:

import matplotlib.pyplot as plt
import numpy
import torch

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold


class Train():
    def __init__(self, x, y, model, learn_rate):
        self.x = x
        self.y = y
        self.model = model
        self.loss_fn = CrossEntropyLoss()
        self.optimiser = Adam(model.parameters(), lr=learn_rate)

    def plot_dataset(self):
        """Plot a dataset with two features and binary classes"""
        axes = [max(self.x[0]), min(self.x[0]), min(self.x[1]), max(self.x[1])]
        plt.plot(self.x[0][:, 0][self.x[1] == 0],
                 self.x[0][:, 1][self.x[1] == 0], "bs")
        plt.plot(self.x[0][:, 0][self.x[1] == 1],
                 self.x[0][:, 1][self.x[1] == 1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
        plt.show()

    def solve(self, num_epochs, num_folds):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        kfold = KFold(
            n_splits=num_folds, random_state=17, shuffle=True)

        for i, (train_index, test_index) in enumerate(kfold.split(self.x)):
            print(f"Fold {i}")
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Convert to tensors
            x_train = torch.from_numpy(x_train).type(torch.float32)
            x_test = torch.from_numpy(x_test).type(torch.float32)
            y_train = torch.from_numpy(y_train)
            y_test = torch.from_numpy(y_test)

            # Training the NN
            for epoch in range(num_epochs):
                # Zero gradients
                self.optimiser.zero_grad()
                # Forward pass
                y_pred = self.model(x_train)
                # Calculate loss
                loss = self.loss_fn(y_pred, y_train)
                # Backward pass
                loss.backward()
                # Update weights
                self.optimiser.step()
                # Calculate metrics
                train_acc = torch.sum(torch.argmax(
                    y_pred, dim=1) == y_train) / y_train.shape[0]
                with torch.no_grad():
                    y_pred_test = self.model(x_test)
                    test_loss = self.loss_fn(y_pred_test, y_test)
                    test_acc = torch.sum(torch.argmax(
                        y_pred_test, dim=1) == y_test) / y_test.shape[0]
                # Log metrics
                train_losses.append(loss.item())
                train_accuracies.append(train_acc)
                test_losses.append(test_loss.item())
                test_accuracies.append(test_acc)
                if epoch % 10 == 0:
                    # Print to console
                    print("Epoch {}:\tTrain loss={:.4f}  \tTrain acc={:.2f}  \tTest loss={:.4f}  \tTest acc={:.2f}".format(
                        epoch, loss.item(), train_acc*100, test_loss.item(), test_acc*100))

# Packaged with - main.py
# Author - Thomas Bandy (c3374048)
# Description:

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report


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
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        kfold = KFold(
            n_splits=num_folds, random_state=17, shuffle=True)

        for i, (train_index, test_index) in enumerate(kfold.split(self.x)):
            print(f"Fold {i}")
            self.x_train, self.x_test = self.x[train_index], self.x[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]

            # Convert to tensors
            self.x_train = torch.from_numpy(self.x_train).type(torch.float32)
            self.x_test = torch.from_numpy(self.x_test).type(torch.float32)
            self.y_train = torch.from_numpy(self.y_train)
            self.y_test = torch.from_numpy(self.y_test)

            # Training the NN
            for epoch in range(num_epochs):
                # Zero gradients
                self.optimiser.zero_grad()
                # Forward pass
                y_pred = self.model(self.x_train)
                # Calculate loss
                loss = self.loss_fn(y_pred, self.y_train)
                # Backward pass
                loss.backward()
                # Update weights
                self.optimiser.step()
                # Calculate metrics
                train_acc = torch.sum(torch.argmax(
                    y_pred, dim=1) == self.y_train) / self.y_train.shape[0]
                with torch.no_grad():
                    y_pred_test = self.model(self.x_test)
                    test_loss = self.loss_fn(y_pred_test, self.y_test)
                    test_acc = torch.sum(torch.argmax(
                        y_pred_test, dim=1) == self.y_test) / self.y_test.shape[0]
                # Log metrics
                self.train_losses.append(loss.item())
                self.train_accuracies.append(train_acc)
                self.test_losses.append(test_loss.item())
                self.test_accuracies.append(test_acc)
                if epoch % 10 == 0:
                    # Print to console
                    print("Epoch {}:\tTrain loss={:.4f}  \tTrain acc={:.2f}  \tTest loss={:.4f}  \tTest acc={:.2f}".format(
                        epoch, loss.item(), train_acc*100, test_loss.item(), test_acc*100))

    def export_results(self):
        with torch.no_grad():
            y_pred_probs_test = self.model(self.x_test).numpy()
        y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1)
        with open('temp.txt', 'w') as f:
            f.write(
                f"Classification Report - \n {classification_report(self.y_test, y_pred_classes_test)} \n Confusion Matrix - \n {confusion_matrix(self.y_test, y_pred_classes_test)}")

    def generate_learning_graph(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
        ax1.plot(self.train_losses, color='b', label='train')
        ax1.plot(self.test_losses, color='g', label='test')
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.plot(self.train_accuracies, color='b', label='train')
        ax2.plot(self.test_accuracies, color='g', label='test')
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.show()

    # def generate_graph(self):
    #     activation_range = numpy.arange(-7, 7, 0.1)         # TODO fix
    #     test_coordinates = [(self.x, self.y)
    #                         for self.y in activation_range for self.x in activation_range]

    #     x_, y_ = numpy.meshgrid(activation_range, activation_range)
    #     plt.scatter(
    #         x_, y_, c=['g' if x_ > 0 else 'b' for x_ in test_classifications])
    #     plt.show()

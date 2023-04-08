# Packaged with - main.py, neural_network.py, dataset_generation.py, svm.py
# Author - Thomas Bandy (c3374048)


import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.autograd import Variable


class Train():
    def __init__(self, x, y, model, learn_rate):
        self.x = x
        self.y = y
        self.model = model
        self.loss_fn = CrossEntropyLoss()
        self.optimiser = Adam(model.parameters(), lr=learn_rate)

    def plot_dataset(self):
        """Plot a dataset with two features and binary classes. Adapted from UON COMP3330 Labs Week 3 - 04_svm_two_spiral_dataset"""
        axes = [20, -20, 20, -20]
        plt.plot(self.x[:, 0][self.y == 0],
                 self.x[:, 1][self.y == 0], "bs")
        plt.plot(self.x[:, 0][self.y == 1],
                 self.x[:, 1][self.y == 1], "g^")
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

        # KFold validation loop. Enumerates the indexs from each fold and returns them
        for i, (train_index, test_index) in enumerate(kfold.split(self.x)):
            print(f"Fold {i}")
            self.x_train, self.x_test = self.x[train_index], self.x[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]

            # Convert to tensors
            self.x_train = torch.from_numpy(self.x_train).type(torch.float32)
            self.x_test = torch.from_numpy(self.x_test).type(torch.float32)
            self.y_train = torch.from_numpy(self.y_train)
            self.y_test = torch.from_numpy(self.y_test)

            # Training the NN adapted from UON COMP3330 Labs Week 4 - Pytorch_banknote_dataset
            for epoch in range(num_epochs):
                self.optimiser.zero_grad()
                y_pred = self.model(self.x_train)
                loss = self.loss_fn(y_pred, self.y_train)
                loss.backward()
                self.optimiser.step()

                # Calculation of metrics
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
                    print("Epoch {}:\tTrain loss={:.4f}  \tTrain acc={:.2f}  \tTest loss={:.4f}  \tTest acc={:.2f}".format(
                        epoch, loss.item(), train_acc*100, test_loss.item(), test_acc*100))

        # Calculate average metrics across folds
        avg_train_loss = np.mean(self.train_losses)
        avg_test_loss = np.mean(self.test_losses)
        avg_train_acc = np.mean(self.train_accuracies)
        avg_test_acc = np.mean(self.test_accuracies)
        print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(
            avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc))

    def export_results(self):
        """Export results of last fold to a temp text file"""
        with torch.no_grad():
            y_pred_probs_test = self.model(self.x_test).numpy()
        y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1)
        with open('temp.txt', 'w') as f:
            f.write(
                f"Classification Report - \n {classification_report(self.y_test, y_pred_classes_test)} \n Confusion Matrix - \n {confusion_matrix(self.y_test, y_pred_classes_test)}")

    def generate_learning_graph(self):
        """Generates and displays a learning curve graph. Adapted from UON COMP3330 Labs Week 4 - Pytorch_banknote_dataset"""
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

    # Attempt to generate generalisation graph
    # def generate_graph(self):
    #     color_map = plt.get_cmap(color_map)
    #     # Define region of interest by data limits
    #     xmin, xmax = self.x[:, 0].min() - 1, self.x[:, 0].max() + 1
    #     ymin, ymax = self.y[:, 1].min() - 1, self.y[:, 1].max() + 1
    #     steps = 1000
    #     x_span = np.linspace(xmin, xmax, steps)
    #     y_span = np.linspace(ymin, ymax, steps)
    #     xx, yy = np.meshgrid(x_span, y_span)

    #     # Make predictions across region of interest
    #     self.model.eval()
    #     labels_predicted = self.model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))

    #     # Plot decision boundary in region of interest
    #     labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    #     z = np.array(labels_predicted).reshape(xx.shape)

    #     fig, ax = plt.subplots()
    #     ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)

    #     # Get predicted labels on training data and plot
    #     train_labels_predicted = model(dataset)
    #     ax.scatter(self.x[:, 0], self.x[:, 1], c=labels.reshape(labels.size()[0]), cmap=color_map, lw=0)
    #     plt.show()

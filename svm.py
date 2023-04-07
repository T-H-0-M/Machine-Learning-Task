# Packaged with - main.py
# Author - Thomas Bandy (c3374048)
# Description:

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy


class SVM():
    def __init__(self, x, y, kernel, gamma, c):
        self.svm = SVC(C=c, kernel=kernel, gamma=gamma)
        self.x = x
        self.y = y

    def split_data(self, test_size, random_state):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state)

    def solve(self):
        self.svm.fit(self.x_train, self.y_train)
        self.y_predicted = self.svm.predict(self.x_test)

    def generate_graph(self):
        activation_range = numpy.arange(-7, 7, 0.1)         # TODO add to NN
        test_coordinates = [(self.x, self.y)
                            for self.y in activation_range for self.x in activation_range]
        test_classifications = self.svm.predict(test_coordinates)
        x_, y_ = numpy.meshgrid(activation_range, activation_range)
        plt.scatter(
            x_, y_, c=['g' if x_ > 0 else 'b' for x_ in test_classifications])
        plt.show()

    def export_results(self):
        with open('temp.txt', 'w') as f:
            f.write(
                f"Classification Report - \n {classification_report(self.y_test, self.y_predicted)} \n Confusion Matrix - \n {confusion_matrix(self.y_test, self.y_predicted)} \n Accuracy - {accuracy_score(self.y_test, self.y_predicted)}")

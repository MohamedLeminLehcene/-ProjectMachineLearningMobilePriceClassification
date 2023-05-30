#Importer Librairies
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def find_nearest_neighbors(self, x_test_point):
        distances = []
        for row in range(len(self.x_train)):
            current_train_point = self.x_train[row]
            current_distance = self.euclidean_distance(current_train_point, x_test_point)
            distances.append((current_distance, row))

        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [self.y_train[index] for (_, index) in distances[:self.k]]
        return nearest_neighbors

    def majority_vote(self, nearest_neighbors):
        counter_vote = Counter(nearest_neighbors)
        y_pred = counter_vote.most_common(1)[0][0]
        return y_pred

    def predict(self, x_test):
        y_pred = []
        for x_test_point in x_test:
            nearest_neighbors = self.find_nearest_neighbors(x_test_point)
            y_pred_point = self.majority_vote(nearest_neighbors)
            y_pred.append(y_pred_point)
        return y_pred



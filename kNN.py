# knn.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist


class kNN(BaseEstimator, ClassifierMixin):

  def __init__(self, n_neighbors:int = 3):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self.X_train = np.copy(X)
    self.y_train = np.copy(y)
    return self

  def predict(self, X):
    #calculate the distances
    distances = cdist(X, self.X_train, metric='euclidean')
    #find the the k nearest neighbors
    nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
    #gather the nearest neighbors labels
    nearest_labels = self.y_train[nearest_indices]
    #using the sign of the sum, predict the most common label
    predictions = np.sign(nearest_labels.sum(axis=1))
    return predictions
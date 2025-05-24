from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

class RandomForestRegressor:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=2024)


class SVMRegressor:
    def __init__(self):
        self.svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)


class Trainer:
    def __init__(self, model):
        self.val_mse_rf_scores = []
        self.val_pcc_rf_scores = []
        self.model = model
    def train(self):

    def 
    def test(self):



import torch
import torch.nn as nn


class GNNModel()
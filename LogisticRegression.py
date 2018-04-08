#### Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
import pandas as pd  

class Regression(object):
    def __init__(self, X, y):
        #生成模型
        #X,y = mini_batch
        self.clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        self.clf.fit(X, y)

    def evaluate(self, X, y):
        #简单评估
        return cross_validation.cross_val_score(self.clf, X, y, cv=5)

    def evaluate2(self, X, y):
        return self.clf.score(X, y) 

    def predict(self, X):
        #预测
        return self.clf.predict(X)

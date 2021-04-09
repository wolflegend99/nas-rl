import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import constants as C

class Dataset():
    def __init__(self, path='churn_modelling.csv'):
        self.path = path
        self.dataset = pd.read_csv(C.DIR_PATH + self.path)

    def preprocess(self):
        # this is a preprocessing module for "Churn modelling" dataset
        X = self.dataset.iloc[:, 3:-1].values
        y = self.dataset.iloc[:, -1].values
        # Encoding categorical data
        # Label Encoding the "Gender" column
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        # One Hot Encoding the "Geography" column
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        y = np.array(y)
        return X, y

    def split(self, X, y, fraction=C.TEST_SPLIT_FRACTION):
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = fraction, random_state = 0)
        return X_train, X_test, y_train, y_test

    def scale(self, X_train, X_test):
        #Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test
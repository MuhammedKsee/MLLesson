import numpy as np
import pandas as pd

# numerator Sxy
# denominator Sxx
# intercept Q1
# slope Q2

class simpleLinReg:
    def __init__(self):
        self.numerator = None
        self.denominator = None
        self.intercept = None
        self.slope = None
        self.TSS = None
        self.RSS = None

    def fit(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        self.denominator = np.sum((x - np.mean(x)) ** 2, axis=0)
        self.numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
        self.slope = self.numerator / self.denominator
        self.intercept = np.mean(y) - self.slope * np.mean(x)

    def predict(self, x):
        x = np.array(x).flatten()
        return self.intercept + self.slope * x

    def r2score(self, y, y_pred):
        y = np.array(y).flatten()
        y_pred = np.array(y_pred).flatten()

        self.RSS = np.sum((y - y_pred) ** 2)
        self.TSS = np.sum((y - np.mean(y)) ** 2)
        return 1 - (self.RSS / self.TSS)

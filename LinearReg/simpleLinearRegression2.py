import numpy as np
import matplotlib.pyplot as plt


class simpleLinearReg:
    def __init__(self):
        self.tetha1 = None
        self.tetha2 = None
        self.slope = None
        self.n = None
        self.x = None
        self.y = None
        self.r2 = None
        self.numerator = None
        self.denominator = None
        self.RSS = None
        self.TSS = None
        self.pred = None
    
    def fit(self,x,y):
        self.denominator =np.sum((x-np.mean(x))**2)
        self.numerator = np.sum((x-np.mean(x))*(y-np.mean(y)))
        self.slope = self.numerator/self.denominator
        self.intercept = np.mean(y)-self.slope*np.mean(x)

        # Y_pred = self.tetha1-self.tetha2*x

    def r2_score(self,y,y_pred):
        self.RSS = np.sum((y-y_pred)**2)
        self.TSS = np.sum((y-np.mean(y))**2)
        r2 = 1-(self.RSS/self.TSS)
        return r2
    

    def predict(self,x):
        pred = self.intercept+(x*self.slope)
        return pred
    


# Karşılaştırma

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv("dataset.csv")
model = LinearRegression()
x = dataset[["SquareMeters","Age"]]
y = dataset["Price"]
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

r1 = r2_score(y_test,y_pred)

print(r1)


model2 = simpleLinearReg()

model2.fit(x,y)
y_pred2 = model2.predict(x_test)
r2 = model2.r2_score(y_test,y_pred2)
print(r2," ",r1)
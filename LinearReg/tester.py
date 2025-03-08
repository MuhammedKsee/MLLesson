import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from simpleLinear_reg import simpleLinReg

data = pd.read_csv("dataset.csv")
x = data[["SquareMeters"]].values
y = data["Price"].values


x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)

readyModel = LinearRegression()
myModel = simpleLinReg()

readyModel.fit(x_train,y_train)
myModel.fit(x_train,y_train)

y_pred = readyModel.predict(x_test)
y_pred2 = myModel.predict(x_test)

r2_ready = r2_score(y_test,y_pred)
r2_my = myModel.r2score(y_test,y_pred2)

print(r2_my," and ",r2_ready)

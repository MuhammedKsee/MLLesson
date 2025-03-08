import numpy as np


x = {[12,23],[213,43]}
y = {[13,24],[123,24]}

class simple_kmeans:
    def Maths(self,x,y,a,b,P):
        self.euclidian = np.abs((x[a][1]-x[b][1])**2) + np.abs((x[a][2]-x[b][2])**(1/2))
        self.manhattan = np.abs(x[a][1]-x[b][1]) + np.abs(x[a][2],x[b][2])
        self.minkowski = (np.sum((np.abs(x-y))**P))**(1/P)
        
        
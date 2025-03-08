import numpy as np
class simpleDecisionTree:
    def __init__(self):
        self.s1 = None
        self.s2 = None
        self.entropy = None
        self.gain = None

    def Entropy(self):
        self.entropy = -(self.s1/(self.s1+self.s2)*np.log2(self.s1/(self.s1+self.s2)))-(self.s2/(self.s1+self.s2)*np.log2(self.s2/(self.s1+self.s2)))
        return self
    
    def Gain(self):
        return None
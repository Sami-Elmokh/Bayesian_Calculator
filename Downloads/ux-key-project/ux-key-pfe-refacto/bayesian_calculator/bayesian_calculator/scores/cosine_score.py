import numpy as np

class CosineScore:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def score_inverse(self, s):
        t = 2 * self.t1 * np.arccos(2 * s - 1) / np.pi + self.t0
        return t

    def score(self, t):
        #print("temps :", t, end = ' : ')
        if t <= self.t0:
         
            return 1
    
        elif t > self.t0 + 2 * self.t1:
       
            return 0
        else:
           
            return 0.5 * (1 + np.cos((t - self.t0) * np.pi / (2 * self.t1)))
        

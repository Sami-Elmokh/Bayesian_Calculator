from scipy.stats import gamma as sgamma
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import warnings
import numpy as np
import pickle









class Likelihood:
    
    def __init__(self,seuil) -> None:
        
        self.type = "Likelihood"
        self.seuil = seuil


    
    def process(self, model, data):
        
        print("Likelihood du model est : ", model.likelihood)
        
        return model.likelihood
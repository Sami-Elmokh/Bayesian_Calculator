from scipy.stats import gamma as sgamma
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import warnings
import numpy as np
import pickle









class CI:
    
    
    def __init__(self, confidence_alpha, plot, output_folder):
    
        '''
        Initialize the CI object.
        
        Parameters:
        confidence_alpha (float): The confidence level.
        plot (bool): Whether to plot the results.
        output_folder (str): The folder to save output.
        '''
        
        self.type = "CI"
        
        self.confidence_alpha = confidence_alpha

        self.plot = plot

        self.output_folder = output_folder 
    


        
        return


    
    def process(self, distribution, Score, data):
        '''
        Process the data to compute confidence intervals.
        
        Parameters:
        distribution: The distribution object.
        Score: The scoring object.
        data: The input data.
        
        Returns:
        tuple: The mean and confidence interval.
        '''

        medians = []
        confidence_alpha = self.confidence_alpha
        n = 5000
        params_list = distribution.sample(n)

        for alpha, beta in params_list:
            simulated_data = np.random.gamma(alpha, beta, 100)
            # Calculate the score of each data point
            scores = [Score.score(data_point) for data_point in simulated_data]
            # Store the median of these scores
            medians.append(np.median(scores))

        medians = np.array(medians)
        median_estimate = np.median(medians)
   
        scores = medians
        delta = int((1-confidence_alpha)*n/2)
        s_moy = sum(scores) / n
        s = sorted(scores)
        i_moy = next((i for i in range(len(s) - 1) if s[i] <= s_moy < s[i + 1]), len(s) - 1 if s_moy >= s[-1] else -1)
   
        conf_int = (s[max(0, i_moy-delta+1)], s[min(n-1, i_moy+delta)])
        print(" CI compute !!")
        print("L'interval de confiance est :", conf_int)

        return (s_moy, conf_int)


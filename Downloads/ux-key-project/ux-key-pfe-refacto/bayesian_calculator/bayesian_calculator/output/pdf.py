from scipy.stats import gamma as sgamma
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
import warnings
import numpy as np
import pickle

class PDF: 

    def __init__(self, output_name) -> None:
        self.type = 'PDF' 
        
        self.output_name = output_name
    
    

    def process(self, distribution):
        '''
        Draw the pdf of a set of gamma distributions
        - params: (n,2)-numpy array of (alpha,beta) parameters
        '''
        x = np.linspace(0, 20, 1000)
        params = distribution.sample(30)
        
        with open(f'{self.output_name}.pkl', 'wb') as file:
            # Use pickle to write the list to a file
            pickle.dump(params, file)

        print("pdf generated -- check output folder !!")


def plot_gamma_distributions(param_list, filename):
    x = np.linspace(0, 10, 1000)
    plt.figure(figsize=(10, 6))

    for alpha, beta in param_list:
        y = scipy.stats.gamma.pdf(x, a=alpha, scale=1/beta)
        plt.plot(x, y, label=f'α={alpha}, β={beta}')

    plt.title('Gamma Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)

    plt.savefig(f"{filename}.jpeg", format='jpeg')  # Save as JPEG
    plt.close()
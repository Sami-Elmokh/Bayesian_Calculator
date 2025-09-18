from scipy.stats import gamma as sgamma
from distributions.distribution import distribution
import numpy as np
import matplotlib.pyplot as plt
from math import gamma
import scipy
import random 


def proposal(alpha, beta, stepsize):
    # params is the list of parameters of the distribution( alpha beta in our example)
    params = [alpha, beta]
    return np.array(params) + np.random.normal(size=len(params)) * stepsize

def log_joint_distribution(alpha, beta, log_p, q, r, s):
    # Example: a simple Gaussian distribution
 
    nominateur = (log_p*(alpha-1)+(-beta*q))
    denominateur = ( (np.log(gamma(alpha))*r)+np.log(beta)*(-alpha*s))
    return nominateur-denominateur

class gamma_distribution(distribution):
    '''
The gamma_distribution class represents a gamma distribution and provides methods for parameter updating, likelihood computation, and sampling using the Metropolis-Hastings algorithm.

Attributes:
    - beta: The scale parameter of the gamma distribution.
    - alpha: The shape parameter of the gamma distribution.
    - log_p: The log of the product of data values, used for likelihood computation.
    - q: The sum of data values, used for likelihood computation.
    - r: The number of data points, used for likelihood computation.
    - s: The sum of the logarithm of data values, used for likelihood computation.
    - likelihood: The likelihood of the distribution.

Methods:
    - update_params(data): Updates the parameters based on new data.
    - cdf(x): Computes the cumulative distribution function at x.
    - pdf(x): Computes the probability density function at x.
    - find_median(params): Finds the median of the gamma distribution.
    - draw(n, filename, plot_start, plot_end, nb_points): Draws the probability density function of a set of gamma distributions.
    - metropolis_hastings(iterations, stepsize, alpha_current, beta_current, log_p, q, r, s): Generates samples of (alpha, beta) using the Metropolis-Hastings algorithm.
    - sample(n): Generates n samples of (alpha, beta) using the Metropolis-Hastings algorithm.
    - get_likelihood(data): Computes the likelihood of the distribution based on provided data.
'''


    def __init__(self, beta, alpha, log_p, q, r, s):
        self.beta = beta
        self.alpha = alpha
        self.log_p = log_p 
        self.q = q 
        self.r = r 
        self.s = s
        self.likelihood = None

        self.params_current = alpha, beta

    def update_params(self, data):
   
        log_somme = np.log(data).sum()
        somme = data.sum()
        length = len(data)
        self.log_p = self.log_p + log_somme
        self.q = self.q+somme 
        self.r = self.r + length
        self.s = self.s + length 

        samples = self.sample(30)
        log_probs = []

        for alpha, beta in samples:
    
            log_prob = np.log(scipy.stats.gamma.pdf(data, a=alpha, scale=1/beta)).sum()
            log_probs.append(log_prob)
        
        
        s_max = np.max(log_probs)
        log_probs_adjusted = np.exp(log_probs - s_max)
        

        log_likelihood = s_max + np.log(log_probs_adjusted.sum()) - np.log(len(samples))
        

        self.likelihood = log_likelihood/len(data)
        
    

    
    

    
    
    def cdf(self, x):
        return (sgamma.cdf(x, self.alpha, 1/self.beta))

    def pdf(self, x):
        return (sgamma.pdf(x, self.alpha, self.scale))
    
    def find_median(self, params):
        # params has 2 parameters: (alpha, beta)
        return(sgamma.ppf(0.5, params[0], 1/params[1]))

    def log_joint_distribution(params, prior_params): 
        #params: alpha, beta
        #prior params: log_p, q, r, s
        nominateur = (prior_params[0]*(params[0]-1)+(-params[1]*prior_params[1]))
        denominateur = ( (np.log(gamma(params[0]))*prior_params[2])+np.log(params[1])*(-params[0]*prior_params[3]))
        return nominateur-denominateur
    
    def draw(self, n, filename, plot_start=0, plot_end=10, nb_points=1000):
        '''
        Draw the pdf of a set of gamma distributions
        - params: (n,2)-numpy array of (alpha,beta) parameters
        '''
        param_list = self.sample(n)

        x = np.linspace(plot_start, plot_end,  nb_points)
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

    
    def metropolis_hastings(self, iterations, stepsize,alpha_current,beta_current,log_p, q, r, s):
        
        samples = []


        for i in range(iterations):
            alpha_proposal, beta_proposal = proposal(alpha_current, beta_current, stepsize)
            if (alpha_proposal)<0 or beta_proposal<0:
                continue
            try:
                p_current = log_joint_distribution(alpha_current, beta_current,log_p, q, r, s)
                p_proposal = log_joint_distribution(alpha_proposal, beta_proposal,log_p, q, r, s)
            except:
                continue
            
            try:
                acceptance_probability = min(1, np.exp(p_proposal - p_current))
            except:
                continue
            if np.random.rand() < acceptance_probability:
                alpha_current, beta_current = alpha_proposal, beta_proposal

            samples.append([alpha_current, beta_current])
        return np.array(samples)

    def sample(self,n):
        '''Generate n samples of (alpha, beta) using Metropolis Hastings algorithm'''
        samples =  self.metropolis_hastings(10000, 0.005 , self.alpha, self.beta ,self.log_p, self.q, self.r, self.s)
        #plt.hist(samples[:,0],bins=200)
        #plt.show()
        index_list = random.sample(range(2000,len(samples)),n )
        random_elements = [samples[i] for i in index_list]
        new_alpha = np.mean([elt[0] for elt in random_elements])
        new_beta = np.mean([elt[1] for elt in random_elements])
        self.alpha  = new_alpha 
        self.beta = new_beta
        
        return random_elements
    
    def get_likelihood(self, data):

        samples = self.sample(30)
        log_probs = []

        for alpha, beta in samples:
    
            log_prob = np.log(scipy.stats.gamma.pdf(data, a=alpha, scale=1/beta)).sum()
            log_probs.append(log_prob)
        
        
        s_max = np.max(log_probs)
        log_probs_adjusted = np.exp(log_probs - s_max)
        

        log_likelihood = s_max + np.log(log_probs_adjusted.sum()) - np.log(len(samples))
        print("likelihood computed !!")
        print("Likelihood = ", log_likelihood/len(data))
        
        return log_likelihood/len(data)
    

# Example usage:
if __name__ == "__main__":
    gamma_dist = gamma_distribution(alpha=2, beta=2)
    data = np.array([1.2, 0.5, 1.5, 2.3])
    gamma_dist.update_params(data)

    x_values = np.linspace(0, 5, 100)
    pdf_values = gamma_dist.pdf(x_values)

    

import numpy as np 
import random
import matplotlib.pyplot as plt
import scores.cosine_score as score_class


class distribution:
    """
    A base class representing a distribution.

    This class defines common methods and provides an interface for subclasses to implement specific distribution functions.

    Methods:
        __init__: Initializes a distribution object.
        update_params: Updates parameters based on data (to be implemented by subclasses).
        cdf: Cumulative distribution function (to be implemented by subclasses).
        pdf: Probability density function (to be implemented by subclasses).
        log_joint_distribution: Logarithm of the joint distribution function (to be implemented by subclasses).
        proposal: Proposal function for Metropolis-Hastings algorithm (to be implemented by subclasses).
        find_median: Finds the median of parameters (to be implemented by subclasses).
        get_likelihood: Gets the likelihood of data (to be implemented by subclasses).
        metropolis_hastings: Metropolis-Hastings algorithm for sampling parameters.
        sample: Generates samples using the Metropolis-Hastings algorithm.
        confidence_interval: Computes confidence interval.
        score_draw: Draws score function plots.
    """
    def __init__(self):
        pass
    
    def update_params(self, data):
        raise NotImplementedError("Subclasses must implement update_params method")
    
    def cdf():
        raise NotImplementedError("Subclasses must implement cdf method")
    
    def pdf(x):
        raise NotImplementedError("Subclasses must implement pdf method")
    
    def log_joint_distribution(self, params_current,prior_params):
        raise NotImplementedError("Subclasses must implement log_joint_distribution method")

    def proposal(self):
        raise NotImplementedError("Subclasses must implement log_joint_distribution method")
    
    def find_median(params):
        raise NotImplementedError("Subclasses must implement log_joint_distribution method")
    
    def get_likelihood(data):
        raise NotImplementedError("Subclasses must implement update_params method")

    
    def metropolis_hastings(self, nb_iterations, step_size, params_current, prior_params):

        """
        Performs the Metropolis-Hastings algorithm for sampling parameters.

        Args:
            nb_iterations (int): The number of iterations.
            step_size (float): The step size for the proposal function.
            params_current (array-like): The current parameters.
            prior_params (array-like): The prior parameters.

        Returns:
            np.array: An array of sampled parameters.
        """

        samples = []
        size = len(params_current)

        for i in range(nb_iterations):
            print(i)
            params_proposal = np.array(params_current) + self.proposal_function(size) * step_size

         
            p_current = self.log_joint_distribution(self.params_current,prior_params)
        
            p_proposal = self.log_joint_distribution(params_current,prior_params)

            try:
                acceptance_probability = min(1, np.exp(p_proposal - p_current))
            except:
                continue
            if np.random.rand() < acceptance_probability:
                params_current = params_proposal

            samples.append(params_current)

        return np.array(samples)
    
    def sample(self,n_iterations, step_size, params_current, prior_params,burn_in = 100):
        """
            Generate samples of (alpha, beta) using the Metropolis-Hastings algorithm.

            Args:
                n_iterations (int): The number of iterations.
                step_size (float): The step size for the proposal function.
                params_current (array-like): The current parameters.
                prior_params (array-like): The prior parameters.
                burn_in (int): The burn-in period.

            Returns:
                list: A list of randomly selected elements from the generated samples.
    """
        
        samples =  self.metropolis_hastings(n_iterations, step_size, params_current, prior_params)
        index_list = random.sample(range(burn_in,len(samples)),n_iterations )
        random_elements = [samples[i] for i in index_list]
        
        return random_elements

    def confidence_interval(self, n, conf_alpha, score):
        delta = int((1-conf_alpha)*n/2)
        params_list = self.sample(n)  # generate a list of n (alpha,beta) samples
        scores=[]
        t_med_list=[]
        # find s_i for each theta_i
        for params in params_list:
            t_med = self.find_median(params)
            t_med_list.append(t_med)
            scores.append(score.score(t_med))
        s_moy = sum(scores) / n
        s = sorted(scores)
        i_moy = next((i for i in range(len(s) - 1) if s[i] <= s_moy < s[i + 1]), len(s) - 1 if s_moy >= s[-1] else -1)
   
        
        conf_int = (s[max(0, i_moy-delta+1)], s[min(n-1, i_moy+delta)])
        
        return (s_moy, conf_int, t_med_list, scores)
    
    def score_draw(self, n,score, filename):
        # generate a list of n samples
        param_list = self.sample(n)
        for param in param_list:
            S = np.linspace(0, 1, 1000)
            T = [score.score_inverse(s) for s in S[1:-1]]

            y = self.cdf(T)
            y_ordered = [1-i for i in y]

            p1 = 1
            p0 = 0
            # Add conversion to list to ensure concatenation
            y_extremes = []
            y_extremes.append(p0)
            y_extremes.extend(y_ordered)
            y_extremes.append(p1)

            plt.plot(S, y_extremes, label=f'α={param[0]}, β={param[1]}')  # eliminate label pour generalisation?

        plt.title('CDF of the score function')
        plt.xlabel('Score')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.savefig(f"{filename}.jpeg", format='jpeg')
        plt.close()

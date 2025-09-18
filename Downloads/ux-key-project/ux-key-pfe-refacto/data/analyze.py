import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import gamma
from scipy.stats import gamma as sgamma

import yaml

import matplotlib.animation as animation
from PIL import Image
import random 
import scipy.stats
from scipy import stats

def get_metrics(filename):
    '''
    Read preprocessed CVS dataset 
    '''
    generic = lambda x: np.array([ float(elt) for elt in re.findall(r'(\d*\.?\d+)',x) ])
    converters = { metric_name: generic for metric_name in [ 'intuitiveness', 'fluidity', 'speed' ] }
    df = pd.read_csv(filename, converters=converters)
    return df

# Declaration of the datasets
filename = {
    'short':"data/endsomethingthou.csv",    
    'long':"data/fiftyrearconsider.csv",
    'new1':"data/beautifulbentline.csv",
    'new2':"data/ascityborder.csv",
    'new3':"data/cuttingstrangersell.csv",
    'new4':"data/flowerpleasantmad.csv",
}   

def show_metric(metric, dataset):
    '''
    Plot distributions of a given metric for all websites of a given dataset
    '''
    df = get_metrics(filename[dataset])
    n_sites = len(df)

    fig, axes = plt.subplots(n_sites, num=f'{metric} of {dataset} dataset')
    for i in range(n_sites):
        sns.kdeplot(df[metric].iloc[i], bw_method=0.05, ax=axes[i])
        
#show_metric('intuitiveness', 'short')
#show_metric('intuitiveness', 'long')
#plt.show()


def log_joint_distribution(alpha, beta, log_p, q, r, s):
    # Example: a simple Gaussian distribution
 
    nominateur = (log_p*(alpha-1)+(-beta*q))
    denominateur = ( (np.log(gamma(alpha))*r)+np.log(beta)*(-alpha*s))
    return nominateur-denominateur


# Proposal function (Gaussian random walk)
def normal_proposal(params, stepsize):
    # params is the list of parameters of the distribution( alpha beta in our example)
    return np.array(params) + np.random.normal(size=len(params)) * stepsize

# Metropolis-Hastings sampling
def metropolis_hastings(iterations, stepsize,alpha_current,beta_current,log_p, q, r, s):
  
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


def create_mp4_from_images(image_filenames, output_filename, frame_duration=500):
        
        frames = [Image.open(image_filename) for image_filename in image_filenames]

        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        img = plt.imshow(frames[0])

        def update(frame):
            img.set_data(frame)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=frame_duration)

        ani.save(output_filename, writer='ffmpeg')


class GammaDistribution:
    # use the conjugate prior to the gamma likelihood (in the wiki) 
    # initialize with a flat prior : p,q,r,s
    # update p,q,r,s with iteration expression in wiki
    # sampling (alpha,beta) as a couple from the joint distribution using hastings
    # generate pdfs of gamma
    # video

    def __init__(self, beta, alpha, log_p, q, r, s):
        self.beta = beta
        self.alpha = alpha
        self.log_p = log_p 
        self.q = q 
        self.r = r 
        self.s = s 

    def update_params(self, data):
        # data is a pandas series 
        log_somme = np.log(data).sum()
        somme = data.sum()
        length = len(data)
        self.log_p = self.log_p + log_somme
        self.q = self.q+somme 
        self.r = self.r + length
        self.s = self.s + length 



    def sample(self,n):
        '''Generate n samples of (alpha, beta) using Metropolis Hastings algorithm'''
        samples =  metropolis_hastings(10000, 0.005 , self.alpha, self.beta ,self.log_p, self.q, self.r, self.s)
        #plt.hist(samples[:,0],bins=200)
        #plt.show()
        index_list = random.sample(range(2000,len(samples)),n )
        random_elements = [samples[i] for i in index_list]

        return random_elements

    def draw(self,n,filename):
        '''
        Draw the pdf of a set of gamma distributions
        - params: (n,2)-numpy array of (alpha,beta) parameters
        '''
        x = np.linspace(0, 20, 1000)
        params = self.sample(n)
        plot_gamma_distributions(params, filename)

    def score_draw(self, n, filename):
        param_list = self.sample(n)  # generate a list of n (alpha,beta) samples
        for alpha, beta in param_list:
            S = np.linspace(0, 1, 1000)
            t1 = 1
            t0 = 1
            T = [score_inverse(s, t1, t0) for s in S[1:-1]]

            y = scipy.stats.gamma.cdf(T, a=alpha, scale=1 / beta)
            y_ordered = [1-i for i in y]

            #p1 = 1 - scipy.stats.gamma.cdf(t0 + 2 * t1, a=alpha, scale=1 / beta) + y_ordered[-1]
            p1 = 1
            #p0 = scipy.stats.gamma.cdf(t0, a=alpha, scale=1 / beta)
            p0 = 0
            #p1 = 1-y_ordered[-1]

            y_extremes = []  # Added conversion to list to ensure concatenation
            y_extremes.append(p0)
            y_extremes.extend(y_ordered)
            y_extremes.append(p1)

            plt.plot(S, y_extremes, label=f'α={alpha}, β={beta}')

        plt.title('CDF of the score function')
        plt.xlabel('Score')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.savefig(f"{filename}.jpeg", format='jpeg')
        plt.close()

    def confidence_interval(self, n, x): # x is the delta of the confidence interval
        delta = int((1-x)*n/2)
        params_list = self.sample(n)  # generate a list of n (alpha,beta) samples
        scores=[]
        t_med_list=[]
        # find s_i for each theta_i
        for alpha, beta in params_list:
            t_med = sgamma.ppf(0.5, alpha, 1/beta)
            t_med_list.append(t_med)
            scores.append(score(t_med))
        s_moy = sum(scores) / n
        s = sorted(scores)
        i_moy = next((i for i in range(len(s) - 1) if s[i] <= s_moy < s[i + 1]), len(s) - 1 if s_moy >= s[-1] else -1)
        #print("s is", s ,"s_moy is  ", s_moy)
        print("i_moy is ", i_moy, "delta is", delta ,"range is ", i_moy-delta+1, i_moy+delta)
        conf_int = (s[max(0, i_moy-delta+1)], s[min(n-1, i_moy+delta)])
        return (s_moy, conf_int, t_med_list, scores)


def cosine_score_inverse(s, t1=1, t0=1):
    t = 2 * t1 * np.arccos(2 * s - 1) / np.pi + t0
    return (t)


def cosine_score(t, t1=1, t0=1):
    s = (np.cos((np.pi * (t - t0)) / (2 * t1)) + 1) / 2
    return (s)


def main_loop(n, data, data_step, params_samples_nb=1000, confid_alpha=0.1, path = "output\\gamma_images\\"):
    n_values=[]
    lower_bounds=[]
    upper_bounds=[]
    means=[]

    for i in range(n):
        print(i)
        df = data[100*data_step:100*(data_step+1)]
        first_metric_page_1.update_params(df)
     
        filename_output = path + "test_" + i
        first_metric_page_1.score_draw(1,filename_output)

        x = first_metric_page_1.confidence_interval(params_samples_nb, confid_alpha)
        n_values.append(10*(i+1))
        lower_bounds.append(x[1][0])
        upper_bounds.append(x[1][1])
        means.append(x[0])
    return (means, n_values, upper_bounds, lower_bounds, x)

import warnings 

if __name__ == '__main__':

    warnings.filterwarnings('ignore') 
    log_p = 1
    q = 10
    r = 10
    s = 10
    first_metric_page_1 = GammaDistribution(1,1,log_p,q,r,s)
    confid=[]
    n_values=[]
    lower_bounds=[]
    upper_bounds=[]
    means=[]

    data = get_metrics(filename["short"])['intuitiveness'][0]
    random.shuffle(data)
    for i in range(1000):
        print(i)
        df= data[100*i:100*(i+1)]

        first_metric_page_1.update_params(df)
    
        filename_output = f"output\\gamma_images\\test_{i}"
        first_metric_page_1.score_draw(1,filename_output)
        x  = first_metric_page_1.confidence_interval(1000,0.1)
        n_values.append(10*(i+1))
        lower_bounds.append(x[1][0])
        upper_bounds.append(x[1][1])
        means.append(x[0])
    t_med_list, scores_list = x[2], x[3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ECDF for t_med_list
    sns.ecdfplot(data=t_med_list, ax=axes[0], label='tmed')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Cumulative Probability')
    axes[0].set_title('Empirical CDF - tmed')
    axes[0].legend()
    axes[0].grid(True)

    # Plot ECDF for scores_list
    sns.ecdfplot(data=scores_list, ax=axes[1], label='scores')
    axes[1].set_xlabel('Values')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Empirical CDF - scores')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('empirical_cdfs.png')
    plt.show()
    # # Plot ECDF for t_med_list
    # sns.ecdfplot(data=t_med_list, label='tmed')

    # # Plot ECDF for scores_list
    # sns.ecdfplot(data=scores_list, label='scores')

    # plt.xlabel('Values')
    # plt.ylabel('Cumulative Probability')
    # plt.title('Empirical CDF')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Plotting confidence interval
    print(len(n_values),n_values)
    print(len(means),means)
    print(len(lower_bounds),lower_bounds)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=n_values, y=means, markers=None, label='Mean')
    plt.fill_between(n_values, lower_bounds, upper_bounds, color='lightgrey', alpha=0.5)
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean')
    plt.title('Mean and Confidence Interval vs Number of Samples')
    plt.grid(True)
    plt.savefig('intervalle_confiance_plot2.png')
    plt.show()
 
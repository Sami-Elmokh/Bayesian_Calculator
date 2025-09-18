import numpy as np
import pandas as pd
from distributions.gamma import gamma_distribution as GammaDistribution
from output import Likelihood as ll
from output import ic 
from output import pdf
from  scores.cosine_score import CosineScore as Cosine
import matplotlib.pyplot as plt
import seaborn as sns
import re
import yaml
import json
import seaborn as sns



# Model and Score Factories
model_factories = {
    'gamma': lambda params: GammaDistribution(**params),
    # Add other models as needed
}

score_factories = {
    'cosine': lambda params: Cosine(**params),
    # Add other scores as needed
}

output_factories = {
    'Likelihood': ll.Likelihood,
    'CI': ic.CI,
    'PDF':pdf.PDF,
}


class RequestProcessor:
    """
    
    A class for processing data through pipelines defined in a configuration file.

    This class initializes pipelines based on the configuration, processes data through specific pipelines,
    and provides methods for processing data through all pipelines.

    Attributes:
        config (dict): Configuration loaded from a YAML file.
        pipelines (dict): A dictionary of pipelines, where each pipeline consists of a model, score, and output processor.
    
    
    """
    def __init__(self, yaml_config_filename):
        with open(yaml_config_filename, 'r') as file:
            self.config = yaml.safe_load(file)
        
        
        self.pipelines = self.initialize_pipelines()

    def initialize_pipelines(self):
        pipelines = {}
        for pipeline_config in self.config['pipelines']:
            metric = pipeline_config['metric']
            model = model_factories[pipeline_config['model']](pipeline_config.get('model_params', {}))
            score = score_factories[pipeline_config['score']](pipeline_config.get('score_params', {}))
            output_processor_class = output_factories[pipeline_config['output']]
            lower_bound = pipeline_config['preprocessing']['lower_bound']
            upper_bound = pipeline_config['preprocessing']['upper_bound']
            output_params = pipeline_config.get('output_params', {})
            output_processor = output_processor_class(**output_params)
            pipeline = {
                'metric': metric,
                'model': model,
                'score': score,
                'output_processor': output_processor,
                'output_params': pipeline_config.get('output_params', []),
                'preprocessing_bounds': (lower_bound, upper_bound),
            }
            pipelines[pipeline_config['name']] = pipeline
        return pipelines

    def process(self, pipeline_name, data):
        '''
        Process the data through a specific pipeline.

        Parameters:
        pipeline_name (str): The name of the pipeline.
        data: The input data.

        Returns:
        tuple: The output processor, model, and score.
        '''
        pipeline = self.pipelines[pipeline_name]
        model = pipeline['model']
        score = pipeline['score']
        output_processor = pipeline['output_processor']
        output_params = pipeline['output_params']
        
      
        # Update the model with the data
        model.update_params(data)
        print(model.sample(10))
     
        # Assuming the output processor's process method requires model, score, and params
        # output = output_processor.process(model, score)
        #print("outptut :", output)
        return output_processor, model, score

    def process_all(self):

        '''
        Process the data through all pipelines.

        Returns:
        list: A list containing output processors, models, scores, metrics, and bounds for each pipeline.
        '''


        out = []
        for pip in self.pipelines:
            pipeline = self.pipelines[pip]
            metric = pipeline['metric']
            model = pipeline['model']
            score = pipeline['score']
            output_processor = pipeline['output_processor']
            output_params = pipeline['output_params']
            bounds = pipeline['preprocessing_bounds']
            out.append([output_processor, model, score, metric, bounds])
        
        return out


def read_data(path,metric,bounds):

    '''
    Read and preprocess the data.

    Parameters:
    path (str): The path to the data file.
    metric (str): The metric to be used.
    bounds (tuple): The lower and upper bounds for preprocessing.

    Returns:
    numpy.ndarray: The preprocessed data.
    
    '''

    generic = lambda x: np.array([ float(elt) for elt in re.findall(r'(\d*\.?\d+)',x) ])
    converters = { metric_name: generic for metric_name in [ 'intuitiveness', 'fluidity', 'speed' ] }
    df = pd.read_csv(path, converters=converters)
    df = df[metric][0]
    upper_bound = bounds[1]
    lower_bound = bounds[0]
    df = df[df<upper_bound]
    df = df[df>lower_bound]
    data = df
    return data
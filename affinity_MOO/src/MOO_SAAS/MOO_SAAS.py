import pandas as pd
from ax import *

import torch
import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

import utils


# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

def f_IFN(x: dict) -> float:
    print(x)
    param_names = [f"x{i+1}" for i in range(12)]
    x_sorted = np.array([x[p_name] for p_name in param_names])
    activity = 0
    f = open('../Data/Data_20230111.csv', 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    num_feature = RawData.shape[1]-2
    for i in range(RawData.shape[0]):
        
        if ((RawData[i, :num_feature] == x_sorted).all()):
            # RawData[i, num_feature+3] = 1
            activity = RawData[i, num_feature+1]
            #print(activity)
    return float(activity)
    
def f_GFP(x: dict) -> float:
    param_names = [f"x{i+1}" for i in range(12)]
    x_sorted = np.array([x[p_name] for p_name in param_names])
    activity = 0 
    f = open('../Data/Data_20230111.csv', 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    num_feature = RawData.shape[1]-2
    for i in range(RawData.shape[0]):
        
        if ((RawData[i, :num_feature] == x_sorted).all()):
            # RawData[i, num_feature+2] = 1
            activity = RawData[i, num_feature]
            #print(activity)
    return float(activity)

def build_experiment(MOO_search_space,optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=MOO_search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment
## Initialize with selected points

def initialize_experiment(experiment,RawData):
    b = []
    print('Fetching data...')
    for i in range(0,RawData.shape[0]):
        a = [Arm(parameters = {f'x{x+1}': RawData[i,x] for x in range(RawData.shape[1]-2)})]
        b=b+a
    gr = GeneratorRun(b)
    experiment.new_batch_trial(generator_run=gr).run()
    return experiment.fetch_data()

def define_space(num_features:int,weighted_sum:float):
    '''
        define the design space given number of features and weighted sum of the features
    '''
    MOO_search_space = SearchSpace(
        parameters = [
            RangeParameter(
                name=f"x{i+1}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=100.0
            )
            for i in range(num_features)
        ]
    )
    # Set Sum constraints 
    sum_constraint_upper = SumConstraint(
        parameters = [MOO_search_space.parameters[f'x{i+1}'] for i in range(num_features)],
        # parameters = list(GPx_search_space.parameters.values()),
        # parameters = [GPx_search_space.parameters[f'x{i}'] for i in range(1, 8)]),
        is_upper_bound = True,
        bound = weighted_sum*1.01,
    )
    sum_constraint_lower = SumConstraint(
        parameters = [MOO_search_space.parameters[f'x{i+1}'] for i in range(num_features)],
        is_upper_bound = False,
        bound = weighted_sum*0.99,
    )
    MOO_search_space.add_parameter_constraints([sum_constraint_upper])
    MOO_search_space.add_parameter_constraints([sum_constraint_lower])
    return MOO_search_space

def MOO_SAAS(data_path='../Data/Data_20230111.csv'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f = open(data_path, 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    num_data = RawData.shape[0]
    num_features = RawData.shape[1]-2
    N_BATCH = 6
    search_space =define_space(num_features,100)
    param_names = [f"x{i+1}" for i in range(num_features)]
    metric_IFN = GenericNoisyFunctionMetric("IFN", f=f_IFN, noise_sd=0, lower_is_better=False)
    metric_GFP = GenericNoisyFunctionMetric("GFP", f=f_GFP, noise_sd=0, lower_is_better=True)
    mo = MultiObjective(
        objectives=[Objective(metric=metric_IFN), Objective(metric=metric_GFP)],
    )

    # set the reference point as (4,4)
    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in [(metric_IFN,3),(metric_GFP,3)]
    ]

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
    )
    experiment = build_experiment(MOO_search_space=search_space,optimization_config=optimization_config)
    data = initialize_experiment(experiment,RawData)
    hv_list = []
    model = None
    model = Models.FULLYBAYESIANMOO(
            experiment=experiment, 
            data=data,
            # use fewer num_samples and warmup_steps to speed up this tutorial
            num_samples=512,
            warmup_steps=512,
            torch_device='cuda',
            verbose=False,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC  
        )

    c = model.gen(96)
    output = utils.prediction_result_output(c)
    output.to_csv('../Data/Proposed/20230415.csv',index=False,sep=',')
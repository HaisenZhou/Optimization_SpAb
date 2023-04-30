import pandas as pd
from ax import *
from datetime import date

import torch
import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

from MOO_SAAS import utils


# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

def f_target(x: dict) -> float:
    '''
    Target is the activity to be maximized.
    First row in the data file.
    '''
    param_names = [f"x{i+1}" for i in range(len(x))]
    x_sorted = np.array([x[p_name] for p_name in param_names])
    activity = 0
    RawData,num_feature = utils.get_data()
    for i in range(RawData.shape[0]):
        if ((RawData[i, :num_feature] == x_sorted).all()):
            activity = RawData[i, num_feature]
    return float(activity)
    
def f_control(x: dict) -> float:
    '''
    Control is the activity to be minimized.
    Second row in the data file.
    '''
    param_names = [f"x{i+1}" for i in range(len(x))]
    x_sorted = np.array([x[p_name] for p_name in param_names])
    activity = 0 
    RawData,num_feature = utils.get_data()
    for i in range(RawData.shape[0]):
        if ((RawData[i, :num_feature] == x_sorted).all()):
            activity = RawData[i, num_feature+1]
    return float(activity)

def f_blank(x: dict) -> float:
    '''
    Blank is the activity to be minimized.
    Thrid row in the data file.
    '''
    param_names = [f"x{i+1}" for i in range(len(x))]
    x_sorted = np.array([x[p_name] for p_name in param_names])
    activity = 0 
    RawData,num_feature = utils.get_data()
    for i in range(RawData.shape[0]):
        if ((RawData[i, :num_feature] == x_sorted).all()):
            activity = RawData[i, num_feature+2]
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
        a = [Arm(parameters = {f'x{x+1}': RawData[i,x] for x in range(RawData.shape[1]-3)})]
        b=b+a
    gr = GeneratorRun(b)
    experiment.new_batch_trial(generator_run=gr).run()
    print('Done!')
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

def MOO_SAAS():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RawData,num_features = utils.get_data()
    print(num_features)
    N_BATCH = 6
    search_space =define_space(num_features,100)
    param_names = [f"x{i+1}" for i in range(num_features)]
    metric_target = GenericNoisyFunctionMetric("target", f=f_target, noise_sd=0.1, lower_is_better=False)
    metric_control = GenericNoisyFunctionMetric("control", f=f_control, noise_sd=0.1, lower_is_better=True)
    metric_blank = GenericNoisyFunctionMetric("blank", f=f_blank, noise_sd=0.1, lower_is_better=True)
    mo = MultiObjective(
        objectives=[Objective(metric=metric_target), Objective(metric=metric_control), Objective(metric=metric_blank)],
    )

    # set the reference point as (3.2,3.2,3.2)
    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in [(metric_target,3.2),(metric_control,3.2),(metric_blank,3.2)]
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

    c = model.gen(1)
    output = utils.prediction_result_output(c)
    today = date.today()
    today=format(today,"%Y%m%d")
    output.to_csv(f'../Data/Proposed/{today}.csv',index=False,sep=',')
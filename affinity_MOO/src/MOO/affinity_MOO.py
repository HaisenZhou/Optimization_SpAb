import numpy as np
from ax import *
from ax.core.metric import Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner
import torch
# Factory methods for creating multi-objective optimization model.
from ax.modelbridge.factory import get_MOO_EHVI

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
import MOO.utils as utils


# Plotting imports and initialization

from ax import ParameterType, RangeParameter, SearchSpace, SumConstraint, Objective
from ax.core.generator_run import GeneratorRun

import pandas as pd


# Read data from CSV


class IFN_data(NoisyFunctionMetric):

    def f(self, x: np.ndarray) -> float:
        activity = 0
        f = open('../Data/Data_20230111.csv', 'r',encoding='utf-8-sig')
        RawData = np.genfromtxt(f, delimiter=",")
        num_feature = RawData.shape[1]-2
        for i in range(RawData.shape[0]):
            #label the fetched data
            if ((RawData[i, :num_feature] == x).all()):
                # RawData[i, num_feature+3] = 1
                activity = RawData[i, num_feature+1]
                print('IFN=',activity)
        return float(activity)
    
class GFP_data(NoisyFunctionMetric):

    def f(self, x: np.ndarray) -> float:
        activity = 0 
        f = open('../Data/Data_20230111.csv', 'r',encoding='utf-8-sig')
        RawData = np.genfromtxt(f, delimiter=",")
        num_feature = RawData.shape[1]-2
        for i in range(RawData.shape[0]):
            #label the fetched data
            if ((RawData[i, :num_feature] == x).all()):
                # RawData[i, num_feature+2] = 1
                activity = RawData[i, num_feature]
                print('GPF=',activity)
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
        bound = weighted_sum*1.02,
    )
    sum_constraint_lower = SumConstraint(
        parameters = [MOO_search_space.parameters[f'x{i+1}'] for i in range(num_features)],
        is_upper_bound = False,
        bound = weighted_sum*0.98,
    )
    MOO_search_space.add_parameter_constraints([sum_constraint_upper])
    MOO_search_space.add_parameter_constraints([sum_constraint_lower])
    return MOO_search_space

def MOO(data_path='../Data/Data_20230111.csv'):

    f = open(data_path, 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    num_data = RawData.shape[0]
    num_features = RawData.shape[1]-2
    N_BATCH = 6
    search_space =define_space(num_features,100)
    list_name = [f"x{i+1}" for i in range(num_features)]
    metric_IFN = IFN_data("IFN", param_names=list_name, noise_sd=0, lower_is_better=False)
    metric_GFP = GFP_data("GFP", param_names=list_name, noise_sd=0, lower_is_better=True)
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
    ehvi_experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    ehvi_data = initialize_experiment(ehvi_experiment,RawData)
    ehvi_hv_list = []
    ehvi_model = None
    ehvi_model = get_MOO_EHVI(
            experiment=ehvi_experiment, 
            data=ehvi_data,
            device= torch.device("cpu")
        )
    hv = observed_hypervolume(modelbridge=ehvi_model)
    c = ehvi_model.gen(96)
    output = utils.prediction_result_output(c)
    output.to_csv('test',index=False,sep=',')
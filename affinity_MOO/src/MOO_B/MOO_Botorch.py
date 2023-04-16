import pandas as pd
from ax import *

import os
import torch
import numpy as np

# import modules for model initialization
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

import utils


# Read the experiment data

def get_data(data_path:str = '../Data/Data_20230111.csv',num_objective=2,n=6):
    f = open(data_path, 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")
    # generate training data
    train_x = torch.from_numpy(RawData[:,:(num_objective*-1)])
    train_obj = torch.from_numpy(RawData[:,(num_objective*-1):])
    train_obj[:,-1] = train_obj[:,-1] *-1
    return train_x, train_obj

def initialize_model(train_x, train_obj,upper_bound=100):
    # define models for objective and constraint
    print(train_obj)
    bounds = torch.stack([torch.zeros(train_x.shape[1]),torch.ones(train_x.shape[1])*upper_bound])
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, 0.1 ** 2)
        models.append(
            FixedNoiseGP(
                train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# defind helper functions
def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    upper_bound=100
    bounds = torch.vstack([torch.zeros(train_x.shape[1]),torch.ones(train_x.shape[1])*upper_bound])
    ref_point = [3,3]
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point
        X_baseline=train_x,
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    constraint = (torch.tensor(range(12), dtype=torch.int64),torch.ones(12),100) # sum of all Xs equal to 100
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=2,
        equality_constraints=[constraint],
        num_restarts=20,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    return candidates



def MOO_Botorch(data_path='../Data/Data_20230111.csv'):

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    f = open(data_path, 'r',encoding='utf-8-sig')
    RawData = np.genfromtxt(f, delimiter=",")

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    N_BATCH = 20
    MC_SAMPLES = 128 
    ref_point = torch.tensor([2,-2])
    verbose = True

    hvs_qnehvi = []

    # call helper functions to generate initial training data and initialize model
    train_x_qnehvi, train_obj_qnehvi = get_data()
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

    # compute hypervolume
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_qnehvi)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch


    t0 = time.monotonic()

    # fit the models
    fit_gpytorch_mll(mll_qnehvi)

    # define the qNEI acquisition modules using a QMC sampler

    qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # optimize acquisition functions and get new observations

    new_x_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
        )
    print(new_x_qnehvi)

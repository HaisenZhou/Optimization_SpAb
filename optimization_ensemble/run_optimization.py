import numpy as np
from pathlib import Path

from optimization.single_objective_BO import GP_qNEI
from optimization.GA_constraint import constraint_genetic_algorithm
from optimization.random_int import random_int


def fitness_function(y: np.ndarray, target_mean: float, control_mean: float) -> np.ndarray:
    
     # second last column devided by final column as the fitness
    fitness = (y[:, -2] - target_mean) / target_mean - (y[:, -1] - control_mean) / control_mean
    return fitness

def run_optimization (
        all_data_dir: Path,
        last_iteration_data_dir: Path,
        random_data_dir = Path,
        num_BO_candidates: int = 32,
        num_GA_candidates: int = 32,
        num_random_candidates: int = 32,
        output_dir: Path = Path('../Data/Data_TNF/Proposed/Proposed_datax.csv'),
        num_objectives: int = 2,
        ):
      
     # Read data from random_data_dir and calculated average of the 9th and 10th column, random_data_dir is the path to the file containing all the randomly sampled data
    random_data = np.genfromtxt(random_data_dir, delimiter=',', encoding='utf-8-sig')
    target_mean = np.mean(random_data[:, 8])  # average of the 9th column
    control_mean = np.mean(random_data[:, 9])  # average of the 10th column

    # read the data from the all_data_dir, remember to remove the header, all_data_dir is the path to the file containing all the data prevously collected
    raw_data = np.genfromtxt(all_data_dir, delimiter=',',encoding='utf-8-sig')
    x = raw_data[:, :num_objectives*-1]
    x = x / x.sum(axis=1, keepdims=True)
    y = raw_data[:, -num_objectives:]
    fitness = fitness_function(y, target_mean, control_mean)
    
    last_iteration_data = np.genfromtxt(last_iteration_data_dir, delimiter=',',encoding='utf-8-sig')
    last_iteration_x = last_iteration_data[:, :num_objectives*-1]
    last_iteration_x = last_iteration_x / last_iteration_x.sum(axis=1, keepdims=True)
    last_iteration_y = last_iteration_data[:, -num_objectives:]
    last_iteration_fitness = fitness_function(last_iteration_y, target_mean, control_mean)
    
    print('initializing optimization')
    GA = constraint_genetic_algorithm(last_iteration_x, last_iteration_fitness, num_GA_candidates)
    BO = GP_qNEI(x, fitness, num_BO_candidates)
    rand = random_int(num_random_candidates)
    
    print('proposing new candidates with GA')
    GA_candidates = GA.propose_new_candidates()
    print('proposing new candidates with BO')
    BO_candidates = BO.propose_new_candidates()
    print('proposing new candidates with random')
    rand_candidates = rand.propose_new_candidates()
    # combine the candidates
    all_candidates = np.concatenate((GA_candidates, BO_candidates, rand_candidates), axis=0)
     # multiply by 100 and round to integer
    all_candidates = np.rint(all_candidates * 100).astype(int)

    print('all_candidates:', all_candidates.shape)
    np.savetxt(output_dir, all_candidates, delimiter=',',fmt='%.3f')

run_optimization(
        all_data_dir = Path('../Data/Data_TNF/All_data_240509.csv'),
        last_iteration_data_dir= Path('../Data/Data_TNF/N7P102.csv'),
        random_data_dir = Path('../Data/Data_TNF/Random_data_240509.csv'),
        num_BO_candidates = 1,
        num_GA_candidates = 1,
        num_random_candidates = 1,  
        num_objectives = 2)


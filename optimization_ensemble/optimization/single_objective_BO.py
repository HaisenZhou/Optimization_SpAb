from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.utils.transforms import normalize, standardize

from botorch.sampling.normal import SobolQMCNormalSampler
import sklearn
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

def model_evaluation(model, X_test, y_test):
    # evaluate the model
    model.eval()
    y_hat = model(X_test).mean.cpu().detach().numpy()
    r_2 = sklearn.metrics.r2_score(y_hat, y_test.cpu().numpy())
    mse = sklearn.metrics.mean_squared_error(y_hat, y_test.cpu().numpy())
    print("Train R2 score:", r_2)
    print("Train MSE score:", mse)
    return {'r_2': r_2, 'mse': mse}

class GP_qNEI:
    
    def __init__(
            self, 
            x: np.ndarray,
            y: np.ndarray,
            num_candidates: int
            ):
        
        x = torch.tensor(x, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double)
        self.num_candidates = num_candidates
        self.x = x.to('cuda')
        self.y = standardize(y).unsqueeze(-1).to('cuda')
        
    def initialize_model(
            self,
            train_x, 
            train_y, 
            state_dict=None
            ):

        # define models for objective and constraint
        train_Yvar = torch.full_like(train_y, 0.12)
        model = SingleTaskGP(train_x, train_y, train_Yvar)
        # combine into a multi-output GP model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model
    
    def optimize_acqf_and_get_observation(
            self,
            acq_func,
            bounds,
            num_restarts = 20,
            raw_samples = 512
            )->np.ndarray:
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        
        # set the sum of all standardized Xs equal to 1
        convex_constraint = (
            torch.tensor(range(self.x.shape[1]), device='cuda', dtype=torch.int),
            torch.ones(self.x.shape[1],device='cuda', dtype=torch.double),
            1
            ) 
    
        inequality_constraints = []
        # X5 < 0.4 
        inequality_constraints.append((
            torch.tensor([4], dtype=torch.long, device='cuda'),  
            torch.tensor([-1.0], dtype=torch.double, device='cuda'),  
            torch.tensor([-0.4], dtype=torch.double, device='cuda')  # X5 - 0.4 <= 0
        ))

        #  X1 + X2 + X3 + X4 + X7 - X6 - X8 > 0
        inequality_constraints.append((
            torch.tensor([0, 1, 2, 3, 6, 5, 7], dtype=torch.long, device='cuda'),  
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0], dtype=torch.double, device='cuda'),  
            torch.tensor([0.0], dtype=torch.double, device='cuda')  
        ))

                
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            equality_constraints=[convex_constraint],
            inequality_constraints=inequality_constraints,
            q=self.num_candidates,
            num_restarts=num_restarts,
            raw_samples=raw_samples,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            timeout_sec=3600*4,
            sequential=False
        )
        # print new values
        new_x = candidates.detach()
        new_x = new_x.cpu().numpy()
        return new_x


    def propose_new_candidates(self):
        
        # set random seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
        
        mll, model = self.initialize_model(self.x, self.y)
        print('Training the GP model')
        fit_gpytorch_mll(mll)
        model_eval = model_evaluation(model, X_test, y_test)

        
        bounds = torch.stack([torch.zeros(X_train.shape[1]), torch.ones(X_train.shape[1])]).to(X_train)
        
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
        qNEI = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=self.x,
                    sampler=qmc_sampler
                )
        qEI = qExpectedImprovement(
                    model=model,
                    best_f=self.y.max(),
                    sampler=qmc_sampler
                )
        EI = LogExpectedImprovement(
                    model=model,
                    best_f=self.y.max()
                )
        print('Proposing new candidates...')
        #new_x = self.optimize_discrete_EI(EI)
        new_x = self.optimize_acqf_and_get_observation(qNEI,bounds)
        # transform the new_x to numpy array
        new_x = new_x.round(2)
        return new_x
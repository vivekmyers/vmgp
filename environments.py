import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.notebook as tqdm
from config import device
sns.set_style('darkgrid')


def default_transform(x):
    '''Function to squash and transform last dimension of x.'''
    return torch.sum(x, dim=-1)

clamp_val = 100

def csc_tranform(x):
    return torch.sum(1 / torch.sin(x), dim=-1).clamp(min=-clamp_val, max=clamp_val)

def tan_transform(x):
    return torch.sum(torch.tan(x), dim=-1).clamp(min=-clamp_val, max=clamp_val)

def atan_transform(x):
    return torch.sum(torch.atan(1 / x), dim=-1)

def sin_inv_transform(x):
    return torch.sum(torch.sin(1 / x), dim=-1)

def sum_square_transform(x):
    return torch.sum(x.pow(2), dim=-1)

def square_sum_transform(x):
    return torch.sum(x, dim=-1).pow(2)

def sum_sin_transform(x):
    return torch.sum(torch.sin(x), dim=-1)

def prod_transform(x):
    return torch.prod(x, dim=-1)

def sign_transform(x):
    return torch.prod(torch.sign(x), dim=-1)

def sum_floor_transform(x):
    return 5 * torch.sum(torch.floor(x), dim=-1)

def noisy_sum_floor_transform(x):
    mean = 10 * torch.sum(torch.floor(x), dim=-1)
    noise = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).to(x.device)
    return noise + mean 

class FunctionTaskGenerator(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, lengthscale=0.5, transform=default_transform, train_size=None):
        '''Define distribution F over regression tasks. A task f~F is a function 
        f(X) = transform(Z) where Z~GP(X) is sampled from a multitask GP using 
        an RBF kernel with latent_dim tasks and transform is an arbitrary map.
        '''
        super().__init__()
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=latent_dim
        )
        rbf = gpytorch.kernels.RBFKernel()
        rbf.raw_lengthscale.data[...] = lengthscale
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            rbf, num_tasks=latent_dim, rank=1
        )
        self.transform = transform
        self.input_dim = input_dim
        self.dummy = nn.Parameter(torch.empty([]))
    
    def forward(self, batch, K=5, validation=False):
        '''Samples batch of regression tasks with K examples per task. 
        Returns:
        X: datapoints sampled from N(0, I) for batch of tasks 
            with shape [batch, K, input_dim]
        Y: labels f(X) for f~F for batch of tasks with shape [batch, K]
        '''
        shape = [batch, K, self.input_dim]
        with torch.no_grad():
            X = torch.normal(torch.zeros(shape), torch.ones(shape)).to(self.dummy.device)
            Z = gpytorch.distributions.MultitaskMultivariateNormal(
                self.mean_module(X), self.covar_module(X)
            ).sample()
            Y = self.transform(Z)
            assert Y.dim() == 2
        return X, Y


class SinusoidTaskGenerator(nn.Module):
    def __init__(self, a_min=0.1, a_max=5.0, p_min=0.0, p_max=2*np.pi, f_min=0.5, f_max=2.0, 
                 tr_min=-5.0, tr_max=5.0, te_min=-5.0, te_max=10.0, noise=0.01, 
                 transform=torch.sin, out_of_range_val=False):
        '''Define distribution parameters for sinusoidal regression task.'''
        super().__init__()
        self.amp_range = [a_min, a_max]             # DKT: [0.1, 5.0], BMAML: [0.1, 5.0]
        self.phase_range = [p_min, p_max]           # DKT: [0.1, pi], BMAML: [0.0, 2*pi]
        self.freq_range = [f_min, f_max]            # DKT: [1.0, 1.0] (unused), BMAML: [0.5, 2.0]
        self.train_samp_range = [tr_min, tr_max]    # DKT: [-5.0, 5.0], BMAML: [-5.0, 5.0]
        self.test_samp_range = [te_min, te_max]     # DKT: Used for out-of-range testing, if needed
        self.noise_var = noise                      # DKT: 0.0 (unused), BMAML: 0.01.
        self.input_dim = 1
        self.transform = transform
        self.out_of_range_val = out_of_range_val
        self.dummy = nn.Parameter(torch.empty([]))
    
    def forward(self, batch, K, validation=False):
        '''Samples batch of sinusoidal regression tasks with K examples per task.
        Returns:
        X: datapoints of shape (batch, support_size+query_size), sampled uniformly from 
            self.train_samp_range/self.test_samp_range
        Y: Labels for the datapoints in X. Each element Y[i][j] = A[i]*np.sin(B[i]*X[i][j] + C[i]) + eps[i] for all (i, j).
            A[i] is the amplitude for the i-th task, sampled uniformly from self.amplitude_range
            B[i] is the frequency for the i-th task, sampled uniformly from self.freq_range
            C[i] is the phase for the i-th task, sampled uniformly from self.phase_range
            eps[i] is for the i-th task, sampled from Normal(0, (0.01*A[i])^2)
        '''
        amp = torch.empty(batch).uniform_(*self.amp_range)
        phase = torch.empty(batch).uniform_(*self.phase_range)
        freq = torch.empty(batch).uniform_(*self.freq_range)
        shape = [batch, K]
        noise = torch.normal(torch.zeros(shape), self.noise_var * amp[:, None].expand(shape))
        with torch.no_grad():
            samp_range = self.test_samp_range if (self.out_of_range_val and validation) else self.train_samp_range
            X = torch.empty(shape).uniform_(*samp_range)
            Y = amp[:, None] * self.transform(freq[:, None] * X + phase[:, None]) + noise    
        return X.unsqueeze(-1).to(self.dummy.device), Y.to(self.dummy.device)


class StepFunctionTaskGenerator(nn.Module):
    def __init__(self, s_min=-2.5, s_max=2.5, samp_min=-5.0, samp_max=5.0, noise=0.03):
        '''Define distribution parameters for step function task.'''
        super().__init__()
        self.switch_range = [s_min, s_max]
        self.samp_range = [samp_min, samp_max]
        self.noise = noise
        self.input_dim = 1
        self.dummy = nn.Parameter(torch.empty([]))

    def forward(self, batch, K, validation=False):
        '''Samples batch of step function tasks.'''
        switches = torch.empty((batch, 3)).uniform_(*self.switch_range)
        print("Switches:", switches)
        shape = [batch, K]
        noise = torch.normal(torch.zeros(shape), self.noise)
        with torch.no_grad():
            X = torch.empty(shape).uniform_(*self.samp_range)
            num_greater = torch.sum(X[:, :, None] > switches[:, None, :], dim=2)
            Y = 2 * (num_greater % 2) - 1 + noise
        return X.unsqueeze(-1).to(self.dummy.device), Y.to(self.dummy.device)


class ConstantTaskGenerator(nn.Module):
    def __init__(self, input_dim=2):
        '''Define distribution F over trivial constant regression tasks.'''
        super().__init__()
        self.input_dim = input_dim
        self.dummy = nn.Parameter(torch.empty([]))
  
    def forward(self, batch, K=5, validation=False):
      '''Samples batch of regression tasks with K examples per task. 
      Returns:
      X: datapoints sampled from N(0, I) for batch of tasks 
          with shape [batch, K, input_dim]
      Y: labels f(X) for f~F for batch of tasks with shape [batch, K]
      '''
      shape = [batch, K, self.input_dim]
      with torch.no_grad():
          X = torch.normal(torch.zeros(shape), torch.ones(shape)).to(self.dummy.device)
          Y = torch.rand([batch, 1]).to(self.dummy.device).expand(-1, K)
      return X, Y

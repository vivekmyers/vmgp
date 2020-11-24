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


class FunctionPriorModel(gpytorch.models.ExactGP):
    '''Exact GP model.'''
    def __init__(self, likelihood):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def clear(self):
        self.train_inputs = None
        self.train_targets = None

class LinearRegression(gpytorch.models.ExactGP):
    '''Bayesian linear regression model.'''
    def __init__(self, likelihood, input_dim):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_dim)
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def clear(self):
        self.train_inputs = None
        self.train_targets = None

class MLP(nn.Module):
    '''Simple multilayer perceptron with relu activations and batchnorm.'''
    def __init__(self, input_units, hidden_units, output_units, hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        units = input_units
        for i in range(hidden_layers):
            self.layers.extend([
                nn.Linear(units, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(),
            ])
            units = hidden_units
        self.layers.extend([
            nn.Linear(units, output_units),
            nn.BatchNorm1d(output_units),
        ])
        
    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(*batch_shape, -1)
        
class LatentPriorModel(gpytorch.models.ExactGP):
    '''Exact multitask GP model. Optionally uses provided deep kernel model.'''
    def __init__(self, likelihood, output_dim, deep_kernel=None):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=output_dim
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=output_dim, rank=1
        )
        self.deep_kernel = deep_kernel

    def forward(self, x):
        if self.deep_kernel is not None:
            x = self.deep_kernel(x)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)
    
    def clear(self):
        self.train_inputs = None
        self.train_targets = None

class VariationalModel(gpytorch.models.ExactGP):
    '''Multitask variational posterior model that uses deep kernel
    and mean functions.
    '''
    def __init__(self, likelihood, input_dim, latent_dim, hidden_units, hidden_layers):
        super().__init__(None, None, likelihood)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=latent_dim, rank=1
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mlp = MLP(input_dim + 1, hidden_units, input_dim + latent_dim, hidden_layers)

    def forward(self, x, y):
        inputs = torch.cat([x] + [y.unsqueeze(-1)], dim=-1)
        outputs = self.mlp(inputs)
        mean = outputs[..., :self.latent_dim]
        embedding = outputs[..., self.latent_dim:]
        assert embedding.size(-1) == self.input_dim
        covar = self.covar_module(embedding)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)
    
    def clear(self):
        self.train_inputs = None
        self.train_targets = None


class VariationalMetaGP(nn.Module):
    '''Variational GP meta-learner with deep non-Gaussian likelihood.
    out_var: downweighting of the KL-divergence term in the loss; represents
      the variance of the Gaussians in the mixture representing the model's
      posterior predictions
    deep_kernel_dim: (optional int): if set, use a deep kernel for the 
      latent prior p(z|x) represented as a learned projection from input_dim to
      deep_kernel_dim composed with an RBF kernel.
    '''
    def __init__(
        self,
        input_dim,
        hidden_units,
        latent_dim,
        hidden_layers,
        out_var,
        deep_kernel_dim=None,
    ):
        super().__init__()
        if deep_kernel_dim is None:
            kernel_dim = input_dim
            self.deep_kernel = None
        else:
            kernel_dim = deep_kernel_dim
            self.deep_kernel = MLP(input_dim, hidden_units, kernel_dim, hidden_layers)
        self.variational_posterior = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=latent_dim)
        self.variational_model = VariationalModel(
            self.variational_posterior, input_dim, latent_dim, hidden_units, hidden_layers
        )
        self.latent_prior = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=latent_dim)
        self.latent_prior_model = LatentPriorModel(self.latent_prior, latent_dim, self.deep_kernel)
        self.likelihood_transform = MLP(latent_dim, hidden_units, 1, hidden_layers)
        self.latent_dim = latent_dim
        self.out_var = out_var
        
    def loss(self, X, Y):
        '''Compute loss over batch of tasks with datapoints X and labels Y.
        Returns the ELBO loss across all tasks.
        '''
        self.train()
        assert Y.dim() == X.dim() - 1
        self.latent_prior_model.eval()
        self.variational_model.eval()
        p_Z_X = self.latent_prior(self.latent_prior_model(X))
        q_Z_Y = self.variational_posterior(self.variational_model(X, Y))
        Z_samp = q_Z_Y.rsample()
        Y_pred = self.likelihood_transform(Z_samp).squeeze(-1)
        log_p_Y_Z_samp = -torch.pow(Y_pred - Y, 2).sum()
        dkl = torch.distributions.kl.kl_divergence(q_Z_Y, p_Z_X).sum()
        ELBO = log_p_Y_Z_samp - self.out_var * dkl
        return -ELBO

    def forward(self, X_train, Y_train, X_test, samples=100):
        '''Evaluate model predictions on a single meta-test task.
        Takes D_train = (X_train, Y_train) and returns the predicted Y_test 
        corresponding to X_test (in the form of samples from the prediction
        distribution along dimension 0 of the returned tensor).
        '''
        assert X_train.dim() == 2
        self.eval()
        Z_samp = self.variational_posterior(
            self.variational_model(X_train, Y_train)
        ).sample(sample_shape=torch.Size([samples]))
        Y_pred_all = []

        for Z_task_samp in Z_samp:
            self.latent_prior_model.set_train_data(X_train, Z_task_samp, strict=False)
            Z_test = self.latent_prior(self.latent_prior_model(X_test)).sample()
            Y_pred = self.likelihood_transform(Z_test).squeeze(-1)
            Y_pred_all.append(Y_pred)
            self.latent_prior_model.clear()
        
        return torch.stack(Y_pred_all).detach()

class MetaGP(nn.Module):
    '''Baseline non-variational GP meta-learner.
    deep_kernel_dim: (optional int): if set, use a deep kernel for the 
      latent prior p(z|x) represented as a learned projection from input_dim to
      deep_kernel_dim composed with an RBF kernel.
    '''
    def __init__(
        self,
        input_dim,
        deep_kernel_dim=None,
        hidden_units=None,
        hidden_layers=None,
    ):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = FunctionPriorModel(self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        self.deep_kernel = (
            MLP(input_dim, hidden_units, deep_kernel_dim, hidden_layers)
            if deep_kernel_dim is not None
            else None
        )
        
    def loss(self, X, Y):
        '''Compute meta loss over batch of tasks with datapoints X and labels Y.
        Returns negative marginal log likelihood across all tasks.
        '''
        self.train()
        self.gp.eval()
        if self.deep_kernel is not None:
            X = self.deep_kernel(X)
        pred_loss = 0
        p_Y_X = self.gp(X)
        loss = -self.mll(p_Y_X, Y)
        return loss.sum()

    def forward(self, X_train, Y_train, X_test, samples=100):
        '''Evaluate model predictions on a single meta-test task.
        Takes D_train = (X_train, Y_train) and returns the predicted Y_test 
        corresponding to X_test.
        '''
        assert X_train.dim() == 2
        self.eval()
        if self.deep_kernel is not None:
            X_train = self.deep_kernel(X_train)
            X_test = self.deep_kernel(X_test)
        self.gp.set_train_data(X_train, Y_train, strict=False)
        Y_pred = self.likelihood(self.gp(X_test)).sample(sample_shape=torch.Size([samples]))
        self.gp.clear()
        return Y_pred.detach()

class FunctionalMLP(nn.Module):
    '''Simple multilayer perceptron, allowing functional passing of parameters.'''
    def __init__(self, input_units, hidden_units, output_units, hidden_layers):
        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        units = input_units
        for i in range(hidden_layers):
            self.weights.append(nn.Parameter(
                torch.empty([units, hidden_units])
            ))
            self.biases.append(nn.Parameter(
                torch.zeros([hidden_units])
            ))
            units = hidden_units
        self.weights.append(nn.Parameter(
            torch.empty([units, output_units])
        ))
        self.biases.append(nn.Parameter(
            torch.zeros([output_units])
        ))
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        
    def forward(self, x, params=None):
        batch_shape = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        
        if params is None:
            params = list(self.parameters())

        weights = params[:len(params) // 2]
        biases = params[len(params) // 2:]
        
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = x @ weight + bias
            if i < len(weights) - 1:
                x = F.relu(x)

        return x.reshape(*batch_shape, -1)

class EMAML(nn.Module):
    '''An ensemble of MAMLs. Can view predictions of ensembles as samples from predictive posterior.'''
    def __init__(self, support_size, query_size, input_dim, hidden_units, hidden_layers, num_mamls=20, inner_lr=0.1):
        super().__init__()
        self.population = num_mamls
        self.input_dim = input_dim
        self.support_size = support_size
        self.query_size = query_size
        self.herd = nn.ModuleList([
            MAML(input_dim=self.input_dim, hidden_units=hidden_units, 
                 hidden_layers=hidden_layers, inner_lr=inner_lr) 
            for k in range(self.population)
        ])
    
    def loss(self, X, Y):
        X_support = X[:, :self.support_size ,:]
        X_query = X[:, self.support_size:, :]
        Y_support = Y[:, :self.support_size]
        Y_query = Y[:, self.support_size:]
        maml_losses = [m.loss(X_support, Y_support, X_query, Y_query) for m in self.herd]
        return sum(maml_losses)
    
    def forward(self, X_train, Y_train, X_test, samples=None):
        preds = [m.forward(X_train, Y_train, X_test) for m in self.herd]
        Y_pred = torch.stack(preds)
        return Y_pred.detach()

class MAML(nn.Module):
    '''Implementation of MAML'''
    def __init__(self, input_dim, hidden_units, hidden_layers, inner_lr):
        super().__init__()
        self.input_dim = input_dim
        self.net = FunctionalMLP(input_dim, hidden_units, 1, hidden_layers)
        self.inner_lr = inner_lr
        self.loss_func = nn.MSELoss()

    def inner_loop(self, init_weights, X_support, Y_support, X_query):
        '''Do inner update step on mini-batch of tasks with input data X and labels Y.'''
        out = self.net(X_support, init_weights).squeeze(-1)
        loss = self.loss_func(out, Y_support)
        grads = torch.autograd.grad(loss, init_weights, create_graph=True)
        temp_weights = [x - self.inner_lr * g for x, g in zip(init_weights, grads)]
        return self.net(X_query, temp_weights).squeeze(-1)

    def loss(self, X_support, Y_support, X_query, Y_query):
        init_weights = [x for x in self.net.parameters()]
        Y_pred = self.inner_loop(init_weights, X_support, Y_support, X_query)
        meta_loss = self.loss_func(Y_pred, Y_query)
        return meta_loss.mean()

    def forward(self, X_train, Y_train, X_test):
        init_weights = [x for x in self.net.parameters()]
        Y_pred = self.inner_loop(init_weights, X_train, Y_train, X_test)
        return Y_pred

class Alpaca(MetaGP):
    '''Implementation of Alpaca algorithm, viewed as Bayesian linear regression.'''
    def __init__(
        self,
        input_dim,
        deep_kernel_dim=None,
        hidden_units=None,
        hidden_layers=None,
    ):
        super().__init__(input_dim, deep_kernel_dim, hidden_units, hidden_layers)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = LinearRegression(self.likelihood, deep_kernel_dim or 1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        self.deep_kernel = (
            MLP(input_dim, hidden_units, deep_kernel_dim, hidden_layers)
            if deep_kernel_dim is not None
            else None
        )

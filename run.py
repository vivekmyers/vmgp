import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from config import *
from device import device
sns.set_style('darkgrid')


input_dim = data_generator.input_dim

if train_size is not None:
    training_tasks = []
    for _ in range(train_size):
        task_x, task_y = data_generator(batch=batch, K=K + query_size)
        training_tasks.append((task_x, task_y))

def get_training_tasks():
    if train_size is None:
        while 1:
            yield data_generator(batch=batch, K=K + query_size)
    else:
        while 1:
            random.shuffle(training_tasks)
            for task_x, task_y in training_tasks:
                yield task_x, task_y

training_generator = iter(get_training_tasks())

train_losses = []
validation_nll = []
validation_mse = []
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

def print_stats():
    if train_losses: 
        print(f'Iteration {itr} loss:', np.mean(train_losses[-val_interval:]))
    if validation_nll: 
        print(f'Iteration {itr} nll:', validation_nll[-1])
        print(f'Iteration {itr} mse:', validation_mse[-1])
        print()

def nll_metric(pred_y, test_y, out_var=0.1):
    err = torch.pow(pred_y - test_y.unsqueeze(0), 2) / out_var
    return -torch.logsumexp(-err, dim=0).mean() + np.log(pred_y.size(0))

def mse_metric(pred_y, test_y):
    return torch.pow(pred_y.mean(dim=0) - test_y, 2).mean()

def validate_model(val_trials, query_size=1, out_of_range=False, return_se=False):
    nlls = []
    mses = []
    for _ in range(val_trials):
        task_x, task_y = data_generator(batch=1, K=K + query_size, validation=True)
        train_x, train_y = task_x[0, :K], task_y[0, :K]
        test_x, test_y = task_x[0, -query_size:], task_y[0, -query_size:]
        pred_y = model(train_x, train_y, test_x, samples=val_samples)
        nlls.append(nll_metric(pred_y, test_y).item())
        mses.append(mse_metric(pred_y, test_y).item())
    if return_se:
        return np.mean(nlls), np.mean(mses), np.std(nlls) / np.sqrt(val_trials), np.std(mses) / np.sqrt(val_trials)
    return np.mean(nlls), np.mean(mses)

# Example validation task
task_x, task_y = data_generator(batch=1, K=K + query_size + 50000)
train_x, train_y = task_x[0, :K], task_y[0, :K]
test_x, test_y = task_x[0, K:K+query_size], task_y[0, K:K+query_size]
plot_x, plot_y = task_x[0, K+query_size:], task_y[0, K+query_size:]

print('train_x:', train_x.tolist())
print('train_y:', train_y.tolist())
print('test_x:', test_x.tolist())
print('test_y:', test_y.tolist())
print()

pred_y = model(train_x, train_y, test_x, samples=250)
print('mu_test_y:', pred_y.mean(dim=0).tolist())
print('sigma_test_y:', pred_y.std(dim=0).tolist())
print()

nll = nll_metric(pred_y, test_y)
mse = mse_metric(pred_y, test_y)
print('nll:', nll.item())
print('mse:', mse.item())

test_x_ = torch.arange(
    task_x.min() - 1., task_x.max() + 1., 1e-1, device=device
)[:, None].expand(-1, train_x.size(1))
pred_y_ = model(train_x, train_y, test_x_, samples=50)
pred_mu = pred_y_.mean(dim=0)
pred_sigma = pred_y_.std(dim=0)
plt.figure()
plt.hist(pred_y[..., 0].cpu().numpy(), bins=50)
plt.show()
plt.figure(dpi=150)
plt.scatter(train_x[:, 0].cpu().numpy(), train_y.cpu().numpy(), s=10, color='blue', label='training', zorder=3)
plt.scatter(plot_x.cpu().numpy(), plot_y.cpu().numpy(), s=2, color='limegreen', label='actual', zorder=1)
plt.plot(test_x_[:, 0].cpu().numpy(), pred_mu.cpu().numpy(), color='orange', label='prediction', zorder=2)
plt.fill_between(
    test_x_[:, 0].cpu().numpy(),
    (pred_mu - pred_sigma).cpu().numpy(), 
    (pred_mu + pred_sigma).cpu().numpy(),
    alpha=0.2,
    color='orange',
)
plt.scatter(test_x[:, 0].tolist(), test_y.tolist(), s=10, color='red', label='testing', zorder=4)
plt.ylabel('$y$')
plt.xlabel('$x$')
plt.legend()
plt.show()


for _ in tqdm.trange(10000):
    task_x, task_y = next(training_generator)
    if val_interval and itr % val_interval == 0:
        nll, mse = validate_model(val_trials, query_size=query_size)
        validation_nll.append(nll)
        validation_mse.append(mse)
        print_stats()
    loss = model.loss(task_x, task_y)
    train_losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    itr += 1


def smooth(data, kernel, maxnorm=np.inf):
    return nn.functional.conv1d(
        torch.tensor(data)[None, None, :].float().clamp(min=-maxnorm, max=maxnorm),
        torch.ones(kernel)[None, None, :] / kernel,
    ).flatten().numpy()

results = {}
def cache_results(name):
    results[name] = (train_losses, validation_nll, validation_mse)

plt.figure(dpi=100)
plt.plot(smooth(train_losses, 1), linewidth=0.3)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.show()

plt.figure(dpi=100)
plt.plot(smooth(validation_nll, 1), linewidth=0.8)
plt.title('Validation NLL')
plt.xlabel('Iteration')
plt.ylabel('NLL')
plt.show()

plt.figure(dpi=100)
plt.plot(smooth(validation_mse, 1), linewidth=0.8)
plt.title('Validation MSE')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

plt.figure(dpi=100)
start_iter = 5

for k, (_, nlls, mses) in results.items():
    plt.plot(np.arange(start_iter, len(nlls)), nlls[start_iter:], label=k)

plt.ylabel('NLL')
plt.xlabel('Iteration')
plt.legend()
plt.show()

plt.figure(dpi=100)
for k, (_, nlls, mses) in results.items():
    plt.plot(np.arange(start_iter, len(mses)), mses[start_iter:], label=k)
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.legend()
plt.show()

nll_mean, mse_mean, nll_se, mse_se = validate_model(val_trials=2000, query_size=query_size, out_of_range=False, return_se=True)
print(dict(nll_mean=nll_mean, nll_se=nll_se, mse_mean=mse_mean, mse_se=mse_se))

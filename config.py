import torch
from models import *
from environments import *

device = torch.device('cuda:0')

train_size = None
K = 5
query_size = 5

data_generator = FunctionTaskGenerator(input_dim=1, transform=atan_transform).to(device)

batch = 50
val_interval = 1000
val_trials = 50
val_samples = 20
itr = 0
learning_rate = 1e-3

model_name = 'Alpaca'
model = {
    'VMGP': lambda: VariationalMetaGP(
        input_dim=input_dim,
        hidden_units=40,
        latent_dim=10,
        hidden_layers=2,
        out_var=1e-2,
        deep_kernel_dim=10,
    ),
    'MGP': lambda: MetaGP(
        input_dim=input_dim,
        deep_kernel_dim=10,
        hidden_units=40,
        hidden_layers=2,
    ),
    'EMAML': lambda: EMAML(
        input_dim=input_dim,
        hidden_units=40,
        hidden_layers=2,
        support_size=K,
        query_size=query_size
    ),
    'Alpaca': lambda: Alpaca(
        input_dim=input_dim,
        deep_kernel_dim=10,
        hidden_units=40,
        hidden_layers=2,
    )
}[model_name]().to(device)

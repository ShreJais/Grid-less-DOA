import os, time
import numpy as np
import pandas as pd
from scipy.io import loadmat

# pytorch libraries.
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
import utils

import warnings
warnings.filterwarnings('ignore')

def training(model, dataset_cfg, model_cfg, training_cfg, metrics_cfg, train_loader, val_loader, 
    test_loader, device):
    checkpoint_path=training_cfg['checkpoint_path']
    max_epochs, optimizer_type=training_cfg['max_epochs'], training_cfg['optimizer_type']
    learning_rate, loss_type=training_cfg['learning_rate'], training_cfg['loss_type']
    weight_decay, loss_weights=training_cfg['weight_decay'], training_cfg['loss_weights']
    is_norm=training_cfg['is_norm']
    rmse_threshold=metrics_cfg['rmse_threshold']
    actfn_name=model_cfg['act_fn1']
    n_snap, max_val, max_angle=dataset_cfg['n_snap'], dataset_cfg['max_val'], dataset_cfg['max_angle']

    model_name=f'complex_model_{actfn_name}_{optimizer_type}_loss={loss_type}'
    model_path=os.path.join(checkpoint_path, model_name)
    fig_path=os.path.join(model_path, 'images')
    print(f'MODEL-NAME: {model_name}')

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    train_hist, val_hist1, val_hist2, avg_val_hist= utils.train_model(net=model, model_path=model_path, 
        model_name=model_name, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        device=device, max_epochs=max_epochs, lr=learning_rate, loss_type=loss_type, optimizer_type=optimizer_type, 
        weight_decay=weight_decay, rmse_threshold=rmse_threshold, loss_weights=loss_weights, n_snap=n_snap, 
        max_val=max_val, max_angle=max_angle, is_norm=is_norm)
    
    # Save the data for plotting.
    train_hist.to_csv(os.path.join(model_path, 'train_hist.csv'))
    val_hist1.to_csv(os.path.join(model_path, 'val_hist1.csv'))
    val_hist2.to_csv(os.path.join(model_path, 'val_hist2.csv'))
    avg_val_hist.to_csv(os.path.join(model_path, 'avgval_hist.csv'))

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # Make changes on this yml file for different experiment.
    config_file = './config.yml'
    dataset_cfg, model_cfg, training_cfg, metrics_cfg = utils.get_configurations(config_file=config_file)
    train_df, val_df, test_df=utils.get_annotation_file(dataset_cfg=dataset_cfg, shuffle=True)

    train_loader=utils.get_dataloader(csv_file=train_df, dataset_cfg=dataset_cfg, 
        dataset_name='Training', is_traindataset=True)
    val_loader=utils.get_dataloader(csv_file=val_df, dataset_cfg=dataset_cfg, 
        dataset_name='Validation', is_traindataset=False)
    test_loader=utils.get_dataloader(csv_file=test_df, dataset_cfg=dataset_cfg, 
        dataset_name='Test', is_traindataset=False)
    
    actfn_by_name = {
        'relu': nn.ReLU, 'tanh': nn.Tanh, 'swish': nn.SiLU, 'sigmoid': nn.Sigmoid, 'identity': nn.Identity, 
        'leaky_relu': nn.LeakyReLU
        }
    
    base_model=utils.get_model(model_cfg=model_cfg, actfn_by_name=actfn_by_name)
    summary(model=base_model.to(device), input_size=(512, 1, 8, 30, 2), depth=8,
            col_names=['input_size', 'output_size', 'num_params'])
    total_params=sum(p.numel() for p in base_model.parameters())
    print(f'Total parameters: {total_params}')

    training(
        model=base_model, dataset_cfg=dataset_cfg, model_cfg=model_cfg, training_cfg=training_cfg, 
        metrics_cfg=metrics_cfg, train_loader=train_loader, val_loader=val_loader, 
        test_loader=test_loader, device=device
        )


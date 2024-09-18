# Pytorch imports.
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import torch.optim as optim

# Standard libraries.
import os, random, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import model

# Function for setting seed.
def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	# GPU operation have seperate seed.
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

def get_annotation_file(N, seed=42, val_size=0.1, test_size=0.1, shuffle=True):
    file_num=np.arange(0, N, 1)
    test_fnum, train_fnum = train_test_split(file_num, test_size=(1-test_size-val_size), 
        random_state=seed, shuffle=shuffle)
    test_fnum, val_fnum = train_test_split(test_fnum, test_size=0.5, random_state=seed, 
        shuffle=shuffle)
    train_df=pd.DataFrame(train_fnum, columns=['idx'])
    val_df=pd.DataFrame(val_fnum, columns=['idx'])
    test_df=pd.DataFrame(test_fnum, columns=['idx'])
    return train_df, val_df, test_df

def normalize_data(x, axis=(2,3)):
    minima = np.min(x, axis=axis, keepdims=True)
    maxima = np.max(x, axis=axis, keepdims=True)
    x = (x - minima) / (maxima - minima)
    return x

def generate_each_pixel_track(doa_max=80, alpha_max=5, img_size=(11, 161), n_snapshot=100):
    doas = torch.arange(0, img_size[1])
    alphas = torch.arange(0, img_size[0])
    doa_grid, alpha_grid = torch.meshgrid(doas-doa_max, alphas-alpha_max)
    doa_grid, alpha_grid = doa_grid.t().reshape(-1, 1), alpha_grid.t().reshape(-1, 1)
    # print(doa_grid.shape, alpha_grid.shape)
    each_pixel_track = doa_grid + alpha_grid * torch.arange(n_snapshot) / (n_snapshot - 1)
    return each_pixel_track

def rmse_target_map(each_pixel_track, doa, alpha, n_source, n_snapshot=100, sigma=1):
    if n_source==1:
        doa, alpha=torch.as_tensor(doa[0].reshape(-1, 1)), torch.as_tensor(alpha[0].reshape(-1, 1))
    else:
        doa, alpha = torch.as_tensor(doa).reshape(-1, 1), torch.as_tensor(alpha).reshape(-1, 1)
    gt_track = doa + alpha * torch.arange(n_snapshot) / (n_snapshot - 1)
    track_diff = gt_track.unsqueeze(1) - each_pixel_track
    track_diff_rmse = torch.sqrt((track_diff**2).sum(-1) / n_snapshot)
    track_diff_rmse = track_diff_rmse.view(-1, 11, 161)
    t_map = (np.exp(-track_diff_rmse.numpy()**2 / sigma**2)).sum(0)
    t_map = (t_map - np.min(t_map)) / (np.max(t_map) - np.min(t_map))
    return t_map 

class CustomDataset(data.Dataset):
    def __init__(self, annotation_file, dfile_path, labels_path):
        super().__init__()
        self.data_path=dfile_path
        self.labels=pd.read_csv(labels_path)
        self.xy_idx=annotation_file
        self.each_pixel_track = generate_each_pixel_track()
        self.n_source=2
    
    def __len__(self):
        return len(self.xy_idx)
    
    def __getitem__(self, index):
        idx=self.xy_idx.iloc[index]['idx']
        img=torch.as_tensor(normalize_data(np.load(os.path.join(self.data_path, f'img_{idx}.npy')), axis=(0,1))).float()
        # n_source=self.labels.iloc[idx]['n_sources']
        theta_gt=torch.as_tensor(self.labels.iloc[idx][['src1_theta', 'src2_theta']].to_numpy())
        alpha_gt=torch.as_tensor(self.labels.iloc[idx][['src1_alpha', 'src2_alpha']].to_numpy())
        tmap=torch.as_tensor(rmse_target_map(each_pixel_track=self.each_pixel_track, 
            n_source=self.n_source, doa=theta_gt, alpha=alpha_gt)).float()
        return img, theta_gt, alpha_gt, tmap
    
def get_dataloader(annotation_file, dfile_path, labels_path, dataset_name='train', batch_size=512, 
    num_workers=10, is_trainset=True):
    dataset=CustomDataset(annotation_file=annotation_file, dfile_path=dfile_path, 
    labels_path=labels_path)
    dataloader=data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_trainset, 
    num_workers=num_workers, drop_last=False)
    print(f'Number of batches for {dataset_name} per epoch: {len(dataloader)}')
    return dataloader

# Functions for saving and loading the model.
# The hyperparameters are stored in a configuration file.

def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details.
    return os.path.join(model_path, model_name + '.config')

def _get_model_file(model_path, model_name):
    # Name of the file for storing n/w parameters.
    return os.path.join(model_path, model_name+'.tar')

def load_model(model_path, model_name, device, net):
    """
    Loads a saved model from disk.
    Inputs:
        model_path: Path of the checkpoint directory.
        model_name: Name of the model(str).
        net: The state dict is loaded into this model.
    """
    config_file = _get_config_file(model_path=model_path, model_name=model_name)
    model_file = _get_model_file(model_path=model_path, model_name=model_name)
    assert os.path.isfile(config_file), (f'Could not find the config file: {config_file}.', 
        'Are you sure this is the correct path and you have your model config stored here?')
    assert os.path.isfile(model_file), (f'Could not find the config file: {model_file}.', 
        'Are you sure this is the correct path and you have your model config stored here?')
    with open(config_file, 'r') as f:
        config_dict =json.load(f)
    net.load_state_dict(torch.load(f=model_file, map_location=device))
    return net

def save_model(model, model_path, model_name):
	"""
	Given a model, we save the state_dict and hyperparameters.
	Inputs:
		model: Network Object to save parameters from
		model_path: Path of the checkpoint directory.
		model_name: Name of the model(str)
	"""
	config_dict = model.config 
	os.makedirs(model_path, exist_ok=True)
	config_file, model_file = ( 
		_get_config_file(model_path=model_path, model_name=model_name),
		_get_model_file(model_path=model_path, model_name=model_name))
	with open(config_file, 'w') as f:
		json.dump(config_dict, f)
	torch.save(model.state_dict(), model_file)

def loss_fn(preds, labels, reg_parameter=None):
    if reg_parameter is None:
        loss = f.mse_loss(input=preds, target=labels)
    else:
        loss = f.mse_loss(input=preds, target=labels) + reg_parameter * torch.abs(preds).mean()
    return loss

def train_model(
    net:nn.Module, model_path, model_name, fig_path, train_loader, val_loader, test_loader, device, 
    act_fn_by_name, max_epochs=100, lr=1e-3, reg_parameter = 5e-3, weight_decay=1e-5, overwrite=True):
    
    file_exists = os.path.isfile(_get_model_file(model_path=model_path, model_name=model_name))
    print(f'Is file exists?: {file_exists}')

    if file_exists and not overwrite:
        print('Model file already exists. Skipping training...')
    else:
        print('Training...')
        if reg_parameter is None:
            print('Training with only L2 Loss')
        else:
            print('Training with L2 + L1 loss.')
        print(f'Max_epochs: {max_epochs}')
        print(f'Starting Learning rate: {lr}, Regularization Parameter: {reg_parameter}, weight_decay: {weight_decay}')
        
        # Defining optimizer.
        optimizer=optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses_each_epoch, val_losses_each_epoch = [], []
        best_val_epoch = -1
        net=net.to(device)
        print(next(net.parameters()).is_cuda)
        for epoch in range(max_epochs):
            # Training.
            net.train()
            count=0.
            train_epoch_loss=0
            # t=tqdm(iterable=train_loader, leave=False)
            if (epoch+1) % 40 == 0:
                lr=lr*0.1
                optimizer=optim.Adam(params=net.parameters(), lr=lr)
                print(f'Learning rate lr:{lr} changed after epoch: {epoch+1}')
            
            for imgs, _, _, tmaps in train_loader:
            # for imgs, _, _, tmaps, _ in t:
                imgs, tmaps = imgs.to(device), tmaps.to(device)
                optimizer.zero_grad()
                preds =net(imgs.unsqueeze(1)).squeeze(dim=1)
                loss=loss_fn(preds, tmaps, reg_parameter=reg_parameter)
                loss.backward()
                optimizer.step()

                # t.set_description(f'Epoch {epoch+1} -training loss: {loss.item()}')
                train_epoch_loss += loss.item()
            
            train_losses_each_epoch.append(train_epoch_loss/len(train_loader))
            val_loss=test_model(net=net, data_loader=val_loader, device=device, reg_parameter=reg_parameter)
            val_losses_each_epoch.append(val_loss.item())

            print(f'[Epoch {epoch+1:2d}] train_loss: {train_losses_each_epoch[epoch]:4.5f}',
                    f'- val_loss: {val_losses_each_epoch[epoch]: 4.5f}')
            
            if len(val_losses_each_epoch) == 1 or val_loss < val_losses_each_epoch[best_val_epoch]:
                print('\tNew best performance, saving model...')
                best_model_path=os.path.join(model_path, f'best_model')
                best_model_name=model_name+f'_epoch={epoch+1}'
                save_model(model=net, model_path=best_model_path, model_name=best_model_name)
                best_val_epoch=epoch

            print('Saving model for each epochs...')
            diff_model_path=os.path.join(model_path, 'diff')
            diff_model_name=model_name+f'_epoch={epoch+1}'
            save_model(model=net, model_path=diff_model_path, model_name=diff_model_name)

    net1 = load_model(
        model_path=best_model_path, model_name=model_name+f'_epoch={best_val_epoch+1}', 
        act_fn_by_name=act_fn_by_name, net=net, device=device)
    
    test_loss = test_model(net=net1, data_loader=test_loader, device=device, reg_parameter=reg_parameter)
    print(f'test_loss: {test_loss: 4.5f}')
    return train_losses_each_epoch, val_losses_each_epoch

def test_model(net, data_loader, device, reg_parameter):
    net.eval()
    val_loss = 0
    # t=tqdm(iterable=data_loader, leave=False)
    # for imgs, labels in t:
    for imgs, _, _, tmaps in data_loader:
        imgs, tmaps = imgs.to(device), tmaps.to(device)
        with torch.no_grad():
            preds=net(imgs.unsqueeze(1)).squeeze(dim=1)
            val_loss += loss_fn(preds, tmaps, reg_parameter=reg_parameter)
    val_loss = val_loss / len(data_loader)
    return val_loss 
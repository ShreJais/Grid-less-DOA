# pytorch libraries.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# standard libraries.
import os, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


from torchinfo import summary 
import logging
import time
import utils, model

import warnings
warnings.filterwarnings("ignore")

def training(path):
    for act_fn_name in act_fn_by_name:
        print(f'Training base network with {act_fn_name} activation.')
        act_fn=act_fn_by_name[act_fn_name]

        # Path to checkpoint folder.
        CHECKPOINT_PATH=os.path.join(path, f'unet_{act_fn_name}_adam')
        FIG_PATH=CHECKPOINT_PATH+'/images/'

        # create checkpoint path if it doesn't exist.
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(FIG_PATH, exist_ok=True)
        
        base_model=model.unet2(
            act_fn=act_fn, c_in=1, c_out=[6, 12, 24, 48], d_conv_stride=1, u_conv_stride=1,
            conv_filter_size=3, convt_filter_size=[4, 5, 5], convt_stride=2,
            d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, 
            use_bn=True, pool_size=2, pool_stride=2, double_conv=True, is_concat=True)
        summary(model=base_model.to(device), input_size=(512, 1, 11, 161), depth=4)

        # training.

        # if l2 --> reg_parameter=None, weight_decay=0
        # if l2_wd --> reg_parameter=None, weight_decay=1e-5
        # if l2_l12 --> reg_parameter=5e-4, weight_decay=0
        # if l2_l11 --> reg_parameter=5e-4, weight_decay= 1e-5

        train_losses, val_losses = utils.train_model(
            net=base_model.to(device), model_path= CHECKPOINT_PATH, model_name=f'unet_{act_fn_name}_adam',
            fig_path=FIG_PATH, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device, act_fn_by_name=act_fn_by_name, max_epochs=50, reg_parameter=None, 
            weight_decay=0)
        
        np.save(file=CHECKPOINT_PATH+'/train_loss.npy', arr=train_losses)
        np.save(file=CHECKPOINT_PATH+'/val_loss.npy', arr=val_losses)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    print('Loading Data and Labels....')
    DATA_PATH='../dataset_folder/power_spec.npy'
    LABEL_PATH='../dataset_folder/traindata_01.csv'

    dataset=np.load(DATA_PATH) 
    new_data_path='/scratch/sj/dataset'
    os.makedirs(new_data_path, exist_ok=True)

    for i in range(len(dataset)):
        np.save(os.path.join(new_data_path, f'img_{i}.npy'), dataset[i])
    del dataset
    
    labels=pd.read_csv(LABEL_PATH)
    train_df, val_df, test_df=utils.get_annotation_file(N=len(labels))

    doas = labels[['src1_theta', 'src2_theta']].to_numpy()
    alphas = labels[['src1_alpha', 'src2_alpha']].to_numpy()
    print(doas.shape, alphas.shape)

    target_type = 'gaussian_rmse'
    print(f'Target map Type: {target_type}')

    batch_size = 512

    train_loader=utils.get_dataloader(annotation_file=train_df, dfile_path=new_data_path, 
        labels_path=LABEL_PATH, batch_size=batch_size)
    val_loader=utils.get_dataloader(annotation_file=val_df, dfile_path=new_data_path, 
        labels_path=LABEL_PATH, dataset_name='validation', batch_size=batch_size, is_trainset=False)
    test_loader=utils.get_dataloader(annotation_file=test_df, dfile_path=new_data_path, 
        labels_path=LABEL_PATH, dataset_name='test', batch_size=batch_size, is_trainset=False)

    print(f'len(train_loader): {len(train_loader)}')
    print(f'len(val_loader): {len(val_loader)}')
    print(f'len(test_loader): {len(test_loader)}')

    act_fn_by_name = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'swish': nn.SiLU}

    # exp10 - sigma = 1, 2-skip connection, rmse, l2, l2_l11--5e-4, l2_l12--5e-3
    # exp11 - sigma = 0.6, 2 skip connection.
    # exp12 - rmse target map, sigma=1., l2_l11: 5e-3, l2_l12: 5e-4
    # exp14 - max_epoch=50, rmse target map, sigma=1, l2, l2_l11-- 5e-4, l2_l12-- 5e-3
    # exp 15 - max_epoch: 50, rmse target map, sigma=1, 
    #           input_image: diff amplitude, l2, l2_l11-- 5e-4, l2_l12-- 5e-3

    # exp 16 - max_epoch: 50, rmse target map, sigma=1,
    #           dataset split = (80, 10, 10). l2, l2_l12

    # exp18 - newdataset.

    CHECKPOINT_PATH=f'./saved_models/exp18/gaussian_rmse/l2'
    print(CHECKPOINT_PATH)
    training(path=CHECKPOINT_PATH)

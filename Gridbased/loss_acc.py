import os, time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from photutils.detection import find_peaks
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data

from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
import utils, model

import warnings
warnings.filterwarnings("ignore")

def normalize_data(x, axis=(2,3)):
    minima = np.min(x, axis=axis, keepdims=True)
    maxima = np.max(x, axis=axis, keepdims=True)
    x = (x - minima)/(maxima - minima)
    return x

def track_rmse1(preds, tmaps, n_peaks, rmse_threshold=1.2, n_sources=2, n_snapshot=30):
    acc1, acc2, avg_rmse = 0., 0., 0.
    avg_rmse=0
    preds_track_rmse=torch.zeros((preds.shape[0], n_sources))
    for i in range(preds.shape[0]):
        label_peaks = find_peaks(data=tmaps[i], threshold=0, box_size=3, npeaks=n_peaks)
        alpha_gt_loc, theta_gt_loc = torch.as_tensor(label_peaks['y_peak'])-5., torch.as_tensor(label_peaks['x_peak'])-80.
        # alpha_gt_loc, theta_gt_loc = alphas[i], thetas[i]
        pred_peaks=find_peaks(data=preds[i], threshold=0., box_size=3, npeaks=n_peaks)
        # # if len(pred_peaks) != 0:
        if pred_peaks != None:
            alpha_pred_loc, theta_pred_loc = torch.as_tensor(pred_peaks['y_peak'])-5., torch.as_tensor(pred_peaks['x_peak'])-80.
            true_track=theta_gt_loc.reshape(-1, 1) + (alpha_gt_loc.reshape(-1, 1)* torch.arange(n_snapshot))/ (n_snapshot-1)
            pred_track=theta_pred_loc.reshape(-1, 1) + (alpha_pred_loc.reshape(-1, 1)*torch.arange(n_snapshot)) / (n_snapshot-1)
            track_diff = true_track.unsqueeze(dim=1) - pred_track
            track_rmse_values = torch.sqrt((track_diff**2).sum(-1) / n_snapshot)
            row_id, col_id = linear_sum_assignment(cost_matrix=track_rmse_values)
            # avg_rmse += (track_rmse_values[row_id, col_id].sum() / len(theta_gt_loc)).item()
            preds_track_rmse[i]=track_rmse_values[row_id, col_id]
        else:
            preds_track_rmse[i]=torch.tensor([10, 10])
    return preds_track_rmse

def track_rmse3(preds, theta_gts, alpha_gts, n_peaks, rmse_threshold, n_snapshot=30):
    preds_track_rmse=torch.zeros((preds.shape[0], n_peaks))
    for i in range(preds.shape[0]):
        pred_peaks=find_peaks(data=preds[i], threshold=0., box_size=3, npeaks=n_peaks)
        alpha_true_loc, theta_true_loc=torch.as_tensor(alpha_gts[i]), torch.as_tensor(theta_gts[i])
        if pred_peaks!=None:
            alpha_pred_loc, theta_pred_loc = pred_peaks['y_peak'], pred_peaks['x_peak']
            alpha_pred_loc, theta_pred_loc = torch.as_tensor(alpha_pred_loc)-5, torch.as_tensor(theta_pred_loc)-80
            # rmse_values = torch.zeros((n_peaks[i], n_peaks[i]))
            true_track = theta_true_loc.reshape(-1, 1) + ((alpha_true_loc.reshape(-1, 1) * torch.arange(n_snapshot)) / (n_snapshot-1))
            pred_track = theta_pred_loc.reshape(-1, 1) + ((alpha_pred_loc.reshape(-1, 1) * torch.arange(n_snapshot)) / (n_snapshot-1))
            track_diff = true_track.unsqueeze(dim=1) - pred_track 
            track_rmse_values = torch.sqrt((track_diff**2).sum(-1) / n_snapshot)
            row_id, col_id = linear_sum_assignment(track_rmse_values)
            # rmse_sum += track_rmse_values[row_id, col_id].sum()
            # p = track_rmse_values[row_id, col_id] < rmse_threshold
            preds_track_rmse[i]=track_rmse_values[row_id, col_id]
        else:
            preds_track_rmse[i]=torch.tensor([10, 10])
    return preds_track_rmse

def split_data2(dataset, slabels, test_size=0.8, seed=42, shuffle=True):
    (test_data, train_data, test_slabels, train_slabels) = train_test_split(dataset, slabels, test_size=test_size, 
            random_state=seed, shuffle=shuffle)
    (test_data, val_data, test_slabels, val_slabels) = train_test_split(test_data, test_slabels, test_size=0.5, 
            random_state=seed, shuffle=shuffle)
    
    print(train_data.shape, val_data.shape, test_data.shape)
    print(train_slabels.shape, val_slabels.shape, test_slabels.shape)
    return (train_data, train_slabels, val_data, val_slabels, test_data, test_slabels)

def get_2_source_data(labels):
    thetas = labels[['src1_theta', 'src2_theta']].to_numpy()
    alphas = labels[['src1_alpha', 'src2_alpha']].to_numpy()
    print(f'thetas.shape: {thetas.shape}, alphas.shape: {alphas.shape}')
    sparse_maps = np.zeros((len(thetas), 11, 161))
    for i in range(len(thetas)):
        for j in range(thetas.shape[1]):
            sparse_maps[i, int(alphas[i, j]+5), int(thetas[i,j]+80)]=1.
    sparse_maps=torch.as_tensor(sparse_maps).float()
    print(f'sparse_maps.shape: {sparse_maps.shape}')
    return sparse_maps, torch.as_tensor(thetas), torch.as_tensor(alphas)

def acc_vs_epochs(net, model_path, model_name, rmse_threshold, data_loader, device, act_fn_by_name, max_epochs=50):
    diff_model_path=os.path.join(model_path, 'diff')
    track_acc1=np.zeros((max_epochs, len(rmse_threshold)))
    track_acc2=np.zeros((max_epochs, len(rmse_threshold)))
    for i in tqdm(range(max_epochs)):
        epoch_name=f'{model_name}_epoch={i+1}'
        net1=utils.load_model(model_path=diff_model_path, model_name=epoch_name, 
                act_fn_by_name=act_fn_by_name, device=device, net=net)
        acc1, acc2=get_model_track_acc(net=net, dataloader=data_loader, device=device, 
            rmse_threshold=rmse_threshold)
        for j, th in enumerate(rmse_threshold):
            track_acc1[i, j]=acc1[th] / len(data_loader)
            track_acc2[i, j]=acc2[th] / len(data_loader)
    track_acc1=np.array(track_acc1)
    track_acc2=np.array(track_acc2)
    np.save(os.path.join(model_path, 'track_acc1.npy'), track_acc1)
    np.save(os.path.join(model_path, 'track_acc2.npy'), track_acc2)

def get_model_track_acc(net, dataloader, device, rmse_threshold, n_source=2):
    net=net.to(device)
    net.eval()
    acc1, acc2={}, {}
    for rt in rmse_threshold:
        acc1[rt], acc2[rt]=0., 0.
    for imgs, tmaps, theta_gts, alpha_gts in tqdm(dataloader):
        imgs=imgs.to(device)
        preds=net(imgs.unsqueeze(1)).squeeze(1)
        preds=preds.detach().cpu().numpy()
        preds=normalize_data(x=preds, axis=(1, 2))
        preds_track_rmse=track_rmse3(preds=preds, theta_gts=theta_gts, alpha_gts=alpha_gts, 
            n_peaks=n_source, rmse_threshold=rmse_threshold)
        for i, th in enumerate(rmse_threshold):
            acc1[th]+=(((preds_track_rmse[:, 0] < th).sum() / len(preds_track_rmse))*100).item()
            acc2[th]+=(((preds_track_rmse[:, 1] < th).sum() / len(preds_track_rmse))*100).item()
    return acc1, acc2

def get_best_model(net, model_path, model_name, device):
    track_acc1=np.load(os.path.join(model_path, 'track_acc1.npy'))
    track_acc2=np.load(os.path.join(model_path, 'track_acc2.npy'))
    avg_acc=(track_acc1[:, 0]+track_acc2[:, 0])/2
    best_epoch=np.argmax(avg_acc)+1
    best_model_path = os.path.join(model_path, 'diff')
    best_model_name=model_name+f'_epoch={best_epoch}'
    print(f'Best Epoch: {best_epoch}')
    net = utils.load_model(model_path=best_model_path, model_name=best_model_name, net=net, device=device, act_fn_by_name=act_fn_by_name)
    return net
    
def track_rmse2(preds, gt_thetas, gt_alphas, n_peaks=2, n_sources=2, n_snapshot=30):
    preds_track_rmse=torch.zeros((preds.shape[0], n_sources))
    for i in range(preds.shape[0]):
        pred_peaks=find_peaks(data=preds[i], threshold=0., box_size=3, npeaks=n_peaks)
        alpha_gt_loc, theta_gt_loc = gt_alphas[i], gt_thetas[i]
        if pred_peaks != None:
            if (len(pred_peaks) == n_sources) or (len(pred_peaks) > n_sources):
                alpha_pred_loc, theta_pred_loc = torch.as_tensor(pred_peaks['y_peak'])-5., torch.as_tensor(pred_peaks['x_peak'])-80.
                pred_track=theta_pred_loc.reshape(-1, 1) + (alpha_pred_loc.reshape(-1, 1)*torch.arange(n_snapshot)) / (n_snapshot-1)
                true_track=theta_gt_loc.reshape(-1, 1) + (alpha_gt_loc.reshape(-1, 1)* torch.arange(n_snapshot))/ (n_snapshot-1)
                track_diff = true_track.unsqueeze(dim=1) - pred_track
                track_rmse_values = torch.sqrt((track_diff**2).sum(-1) / n_snapshot)
                row_id, col_id = linear_sum_assignment(cost_matrix=track_rmse_values)
                preds_track_rmse[i]=track_rmse_values[row_id, col_id]
            else:
                diff=n_sources-len(pred_peaks)
                alpha_pred_loc, theta_pred_loc = torch.as_tensor(pred_peaks['y_peak'])-5., torch.as_tensor(pred_peaks['x_peak'])-80.
                alpha_pred_loc=torch.cat((alpha_pred_loc, 10*torch.ones(diff)))
                theta_pred_loc=torch.cat((theta_pred_loc, 160*torch.ones(diff)))
                pred_track=theta_pred_loc.reshape(-1, 1) + (alpha_pred_loc.reshape(-1, 1)*torch.arange(n_snapshot)) / (n_snapshot-1)
                true_track=theta_gt_loc.reshape(-1, 1) + (alpha_gt_loc.reshape(-1, 1)* torch.arange(n_snapshot))/ (n_snapshot-1)
                track_diff = true_track.unsqueeze(dim=1) - pred_track
                track_rmse_values = torch.sqrt((track_diff**2).sum(-1) / n_snapshot)
                row_id, col_id = linear_sum_assignment(cost_matrix=track_rmse_values)
                preds_track_rmse[i]=track_rmse_values[row_id, col_id]
        else:
            preds_track_rmse[i]=torch.ones(n_sources)*100
    return preds_track_rmse

def acc_newtestdata_2source(net, dataloader, device, rmse_threshold, n_peaks=2):
    net=net.to(device)
    net.eval()
    acc1, acc2, avg_acc={}, {}, {}
    count, det_rmse, avg_det_rmse={}, {}, {}
    
    for rt in rmse_threshold:
        acc1[rt], acc2[rt]=0., 0.
        det_rmse[rt], count[rt], avg_det_rmse[rt]=0., 0., 0.
    
    for imgs, thetas, alphas in tqdm(dataloader):
        imgs=imgs.to(device)
        preds=net(imgs.unsqueeze(1)).squeeze(1)
        preds=preds.detach().cpu().numpy()
        preds=normalize_data(x=preds, axis=(1, 2))
        preds_track_rmse=track_rmse2(preds=preds, gt_thetas=thetas, gt_alphas=alphas, n_peaks=n_peaks)

        for i, th in enumerate(rmse_threshold):
            acc1[th]+=(((preds_track_rmse[:, 0] < th).sum() / len(preds_track_rmse))*100).item()
            acc2[th]+=(((preds_track_rmse[:, 1] < th).sum() / len(preds_track_rmse))*100).item()
            mask=(preds_track_rmse < th).int()
            count[th] += mask.sum()
            det_rmse[th] += (preds_track_rmse*mask).sum()
    
    for th in rmse_threshold:
        acc1[th]=acc1[th] / len(dataloader)
        acc2[th]=acc2[th] / len(dataloader)
        avg_acc[th]=(acc1[th] + acc2[th]) / 2
        avg_det_rmse[th] = det_rmse[th] / count[th]
        print(f'th: {th}, acc1: {acc1[th]}, acc2: {acc2[th]}, avg_acc: {avg_acc[th]}',
                f'avg_det_rmse: {avg_det_rmse[th]}')
        
    # return acc1, acc2, avg_acc

def acc_tlcbf_2source(dataloader, device, rmse_threshold, n_peaks=2):
    acc1, acc2, avg_acc={}, {}, {}
    count, det_rmse, avg_det_rmse={}, {}, {}
    
    for rt in rmse_threshold:
        acc1[rt], acc2[rt]=0., 0.
        det_rmse[rt], count[rt], avg_det_rmse[rt]=0., 0., 0.
    
    for imgs, thetas, alphas in tqdm(dataloader):
        # imgs=imgs.to(device)
        # preds=net(imgs.unsqueeze(1)).squeeze(1)
        # preds=preds.detach().cpu().numpy()
        # preds=normalize_data(x=preds, axis=(1, 2))
        preds_track_rmse=track_rmse2(preds=imgs, gt_thetas=thetas, gt_alphas=alphas, n_peaks=n_peaks)

        for i, th in enumerate(rmse_threshold):
            acc1[th]+=(((preds_track_rmse[:, 0] < th).sum() / len(preds_track_rmse))*100).item()
            acc2[th]+=(((preds_track_rmse[:, 1] < th).sum() / len(preds_track_rmse))*100).item()
            mask=(preds_track_rmse < th).int()
            count[th] += mask.sum()
            det_rmse[th] += (preds_track_rmse*mask).sum()
    
    for th in rmse_threshold:
        acc1[th]=acc1[th] / len(dataloader)
        acc2[th]=acc2[th] / len(dataloader)
        avg_acc[th]=(acc1[th] + acc2[th]) / 2
        avg_det_rmse[th] = det_rmse[th] / count[th]
        print(f'th: {th}, acc1: {acc1[th]}, acc2: {acc2[th]}, avg_acc: {avg_acc[th]}',
                f'avg_det_rmse: {avg_det_rmse[th]}')

def acc_newtestdata_Nsource(net, dataset, device, rmse_threshold, batch_size, 
    n_sources=4, n_peaks=9):
    net=net.to(device)
    net.eval()
    
    data_loader = data.DataLoader(dataset=data.TensorDataset(*dataset), batch_size=batch_size, shuffle=False, 
        drop_last=False, pin_memory=True, num_workers=10)
    num=dataset[0].shape[0] // batch_size
    acc_vs_snr_th=np.zeros((num, n_sources, len(rmse_threshold)))
    det_rmse_vs_snr_th=np.zeros((num, len(rmse_threshold)))
    count_vs_snr_th=np.zeros((num, len(rmse_threshold)))
    
    for i, (imgs, thetas, alphas, snrs) in tqdm(enumerate(data_loader)):
        imgs=imgs.to(device)
        preds=net(imgs.unsqueeze(1)).squeeze(1)
        preds=preds.detach().cpu().numpy()
        preds=normalize_data(x=preds, axis=(1, 2))
        preds_track_rmse=track_rmse2(preds=preds, gt_thetas=thetas, gt_alphas=alphas, n_peaks=n_peaks, n_sources=n_sources)
        # preds_track_rmse=preds_track_rmse.numpy() 
        for j, th in enumerate(rmse_threshold):
            acc_vs_snr_th[i, :, j]=((preds_track_rmse < th).sum(0) / batch_size)*100
            mask=(preds_track_rmse < th).int()
            det_rmse_vs_snr_th[i, j]=(preds_track_rmse * mask).sum()
            count_vs_snr_th[i, j]=mask.sum()
    return acc_vs_snr_th, det_rmse_vs_snr_th, count_vs_snr_th

def acc_tlcbf(dataset, rmse_threshold, batch_size, n_sources=2, n_peaks=2):
    data_loader = data.DataLoader(dataset=data.TensorDataset(*dataset), batch_size=batch_size, shuffle=False, 
        drop_last=False, pin_memory=True, num_workers=10)
    num=dataset[0].shape[0] // batch_size
    acc_vs_snr_th=np.zeros((num, n_sources, len(rmse_threshold)))
    det_rmse_vs_snr_th=np.zeros((num, len(rmse_threshold)))
    count_vs_snr_th=np.zeros((num, len(rmse_threshold)))

    for i, (imgs, thetas, alphas, snrs) in tqdm(enumerate(data_loader)):
        preds_track_rmse=track_rmse2(preds=imgs, gt_thetas=thetas, gt_alphas=alphas, 
            n_peaks=n_peaks, n_sources=n_sources)
        # preds_track_rmse=preds_track_rmse.numpy()
        for j, th in enumerate(rmse_threshold):
            acc_vs_snr_th[i, :, j]=((preds_track_rmse < th).sum(0) / batch_size)*100
            mask=(preds_track_rmse < th).int()
            det_rmse_vs_snr_th[i, j]=(preds_track_rmse * mask).sum()
            count_vs_snr_th[i, j]=mask.sum()
    return acc_vs_snr_th, det_rmse_vs_snr_th, count_vs_snr_th

# def acc_tlcbf

# moving case 3 sources.
def acc_newtestdata_3source(net, dataset, device, rmse_threshold, batch_size, n_sources=3, n_peaks=7):
    net=net.to(device)
    net.eval()
    acc_vs_snr_th=np.zeros((4, 20, n_sources, len(rmse_threshold)))
    for i in range(4):
        new_dataset=(dataset[0][i*4000:(i+1)*4000], dataset[1][i*4000:(i+1)*4000], 
            dataset[3][i*4000:(i+1)*4000], dataset[3][i*4000:(i+1)*4000])
        data_loader = data.DataLoader(dataset=data.TensorDataset(*new_dataset), batch_size=batch_size, shuffle=False, 
            drop_last=False, pin_memory=True, num_workers=10)

        for j, (imgs, thetas, alphas, snrs) in tqdm(enumerate(data_loader)):
            imgs=imgs.to(device)
            preds=net(imgs.unsqueeze(1)).squeeze(1)
            preds=preds.detach().cpu().numpy()
            preds=normalize_data(x=preds, axis=(1, 2))
            preds_track_rmse=track_rmse2(preds=preds, gt_thetas=thetas, gt_alphas=alphas, n_peaks=n_peaks, n_sources=n_sources)
            preds_track_rmse=preds_track_rmse.numpy()
            for k, th in enumerate(rmse_threshold):
                acc_vs_snr_th[i, j, :, k]=((preds_track_rmse < th).sum(0) / batch_size)*100
    # breakpoint()
    return acc_vs_snr_th

def acc_tlcbf_3sources(dataset, rmse_threshold, batch_size, n_sources=4, n_peaks=7):
    acc_vs_snr_th=np.zeros((4, 20, n_sources, len(rmse_threshold)))
    for i in range(4):
        new_dataset=(dataset[0][i*4000:(i+1)*4000], dataset[1][i*4000:(i+1)*4000], 
            dataset[3][i*4000:(i+1)*4000], dataset[3][i*4000:(i+1)*4000])
        data_loader = data.DataLoader(dataset=data.TensorDataset(*new_dataset), batch_size=batch_size, shuffle=False, 
            drop_last=False, pin_memory=True, num_workers=10)
        for j, (imgs, thetas, alphas, snrs) in tqdm(enumerate(data_loader)):
            preds_track_rmse=track_rmse2(preds=imgs, gt_thetas=thetas, gt_alphas=alphas, n_peaks=n_peaks, n_sources=n_sources)
            preds_track_rmse=preds_track_rmse.numpy()
            for k, th in enumerate(rmse_threshold):
                acc_vs_snr_th[i, j, :,k]=((preds_track_rmse < th).sum(0) / batch_size)*100
    # breakpoint()
    return acc_vs_snr_th

class CustomDataset(data.Dataset):
    def __init__(self, annotation_file, dfile_path, labels_path):
        super().__init__()
        self.dataset=np.load(dfile_path)
        self.labels=pd.read_csv(labels_path)
        self.xy_idx=annotation_file
        self.each_pixel_track = generate_each_pixel_track()

    def __len__(self):
        return len(self.xy_idx)
    
    def __getitem__(self, index):
        # breakpoint()
        idx=self.xy_idx.iloc[index]['idx']
        img=torch.as_tensor(normalize_data(self.dataset[idx], axis=(0,1))).float()
        n_source=self.labels.iloc[idx]['n_sources']
        theta_gt=torch.as_tensor(self.labels.iloc[idx][['src1_theta', 'src2_theta']].to_numpy())
        alpha_gt=torch.as_tensor(self.labels.iloc[idx][['src1_alpha', 'src2_alpha']].to_numpy())
        tmap=torch.as_tensor(rmse_target_map(each_pixel_track=self.each_pixel_track, n_source=n_source,
                doa=theta_gt, alpha=alpha_gt)).float()
        return img, theta_gt, alpha_gt, tmap, n_source

def get_dataloader(annotation_file, dfile_path, labels_path, dataset_name='train', batch_size=512, 
    num_workers=10, is_trainset=True):
    dataset=CustomDataset(annotation_file=annotation_file, dfile_path=dfile_path, 
    labels_path=labels_path)
    dataloader=data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_trainset, 
    num_workers=num_workers, drop_last=False)
    print(f'Number of batches for {dataset_name} per epoch: {len(dataloader)}')
    return dataloader

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

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    print('Loading Data and Labels....')
    DATA_PATH='../dataset_folder/power_spec.npy'
    LABEL_PATH='../dataset_folder/traindata_01.csv'

    # dataset=np.load(DATA_PATH)
    # labels=pd.read_csv(LABEL_PATH)
    
    # train_df, val_df, test_df=get_annotation_file(N=len(labels))
    # # new_data_path='/scratch/shreyas.jaiswal/dataset'
    # val_dataset=CustomDataset(annotation_file=val_df, dfile_path=DATA_PATH, labels_path=LABEL_PATH)
    # val_loader=get_dataloader(annotation_file=val_df, dfile_path=DATA_PATH, labels_path=LABEL_PATH, dataset_name='val',
    #     is_trainset=False)

    # doas = labels[['src1_theta', 'src2_theta']].to_numpy()
    # alphas = labels[['src1_alpha', 'src2_alpha']].to_numpy()
    # n_sources = labels['n_sources']
    # (train_data, train_slabels, val_data, 
    #     val_slabels, test_data, test_slabels)=split_data2(dataset=dataset, slabels=labels)
    # batch_size=512

    # 2 source data.
    # Validation data.
    # val_labels=pd.read_csv('../dataset_folder/val_test_05_old/val_labels.csv')
    # val_tlcbf=normalize_data(x=np.load('../dataset_folder/val_test_05_old/val_power_spec.npy'), axis=(1, 2))
    # val_tlcbf=torch.as_tensor(val_tlcbf).float()
    # val_smaps, val_thetas, val_alphas=get_2_source_data(labels=val_labels)
    # val_dataset=(val_tlcbf, val_smaps, val_thetas, val_alphas)

    # val_loader = data.DataLoader(dataset=data.TensorDataset(*val_dataset), batch_size=256, 
    #     shuffle=False, drop_last=False, pin_memory=True, num_workers=10)

    # 2 source data.
    # Test data.
    # test_labels=pd.read_csv('../dataset_folder/val_test_05_old/test_labels.csv')
    # test_tlcbf=torch.as_tensor(np.load('../dataset_folder/val_test_05_old/test_power_spec.npy')).float()
    # test_smaps, test_thetas, test_alphas=get_2_source_data(labels=test_labels)
    # test_dataset=(test_tlcbf, test_smaps, test_thetas, test_alphas)
    # test_loader = data.DataLoader(dataset=data.TensorDataset(*test_dataset), batch_size=256, shuffle=False, 
    #     drop_last=False, pin_memory=True, num_workers=10)
    
    rmse_threshold=np.round(np.arange(1.2, 3.7, 0.4), decimals=1)

    act_fn_by_name = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'swish': nn.SiLU}
    act_fn=act_fn_by_name['relu']
    
    loss_type='l2' # l2_l12, l2
    print(f'loss_type: {loss_type}')
    checkpoint_path=f'./saved_models/exp18/gaussian_rmse/{loss_type}'
    base_model_l2=model.unet2(
        act_fn=act_fn, c_in=1, c_out=[6, 12, 24, 48], d_conv_stride=1, u_conv_stride=1,
        conv_filter_size=3, convt_filter_size=[4, 5, 5], convt_stride=2,
        d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, 
        use_bn=True, pool_size=2, pool_stride=2, double_conv=True, is_concat=True)
    model_name=f'unet_relu_adam'
    model_path=os.path.join(checkpoint_path, model_name)
    base_model_l2=get_best_model(net=base_model_l2, model_path=model_path, 
        model_name=model_name, device=device)

    loss_type='l2_l12' # l2_l12, l2
    print(f'loss_type: {loss_type}')
    checkpoint_path=f'./saved_models/exp18/gaussian_rmse/{loss_type}'
    base_model_l2_l12=model.unet2(
        act_fn=act_fn, c_in=1, c_out=[6, 12, 24, 48], d_conv_stride=1, u_conv_stride=1,
        conv_filter_size=3, convt_filter_size=[4, 5, 5], convt_stride=2,
        d_dilation=2, u_dialtion=1, u_padding=1, d_padding=2, bias=True, 
        use_bn=True, pool_size=2, pool_stride=2, double_conv=True, is_concat=True)
    model_name=f'unet_relu_adam'
    model_path=os.path.join(checkpoint_path, model_name)
    base_model_l2_l12=get_best_model(net=base_model_l2_l12, model_path=model_path, model_name=model_name, device=device)
    
    # breakpoint()
    # acc_vs_epochs(net=base_model_l2_l12, model_path=model_path, model_name=model_name, 
        # rmse_threshold=rmse_threshold, data_loader=val_loader, device=device, 
        # act_fn_by_name=act_fn_by_name, max_epochs=50)

    # track_acc1=np.load(os.path.join(model_path, 'track_acc1.npy'))
    # track_acc2=np.load(os.path.join(model_path, 'track_acc2.npy'))

    # breakpoint()
    # base_model_l2=get_best_model(net=base_model_l2, model_path=model_path, 
    #     model_name=model_name, device=device)
    
    # test_acc1, test_acc2=get_model_track_acc(
    #     net=base_model, dataloader=test_loader, device=device, rmse_threshold=rmse_threshold)
    # breakpoint()

    # data_dir='../dataset_folder/val_test_05_old'
    # batch_size=256
    # # # Large Val data
    # val_data_path=os.path.join(data_dir, 'val_power_spec.npy')
    # val_label_path=os.path.join(data_dir, 'val_labels.csv')

    # val_tlcbf=torch.as_tensor(np.load(val_data_path)).float()
    # val_labels=pd.read_csv(val_label_path)
    # val_thetas = torch.as_tensor(val_labels[['src1_theta', 'src2_theta']].to_numpy()).float()
    # val_alphas = torch.as_tensor(val_labels[['src1_alpha', 'src2_alpha']].to_numpy()).float()
    # val_dataset=(val_tlcbf, val_thetas, val_alphas)
    # val_loader = data.DataLoader(dataset=data.TensorDataset(*val_dataset), 
    #     batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=10)
    # breakpoint()
    # acc1, acc2, avg_acc=acc_newtestdata_2source(net=base_model_l2, dataloader=val_loader, 
    #     device=device, rmse_threshold=rmse_threshold, n_peaks=5)
    
    # 2 sources.
    # # Large Test data.
    # test_data_path=os.path.join(data_dir, 'test_power_spec.npy')
    # test_label_path=os.path.join(data_dir, 'test_labels.csv')
    # test_tlcbf=normalize_data(x=np.load(test_data_path), axis=(1,2))
    # test_tlcbf=torch.as_tensor(test_tlcbf).float()
    # test_labels=pd.read_csv(test_label_path)
    # test_thetas = torch.as_tensor(test_labels[['src1_theta', 'src2_theta']].to_numpy()).float()
    # test_alphas = torch.as_tensor(test_labels[['src1_alpha', 'src2_alpha']].to_numpy()).float()
    # test_dataset=(test_tlcbf, test_thetas, test_alphas)
    # test_loader = data.DataLoader(dataset=data.TensorDataset(*test_dataset), batch_size=batch_size, shuffle=False, 
    #     drop_last=False, pin_memory=True, num_workers=10)
    
    # print('TL-CBF results')
    # acc_tlcbf_2source(dataloader=test_loader, device=device, rmse_threshold=rmse_threshold, n_peaks=2)
    # print('U-Net-l2 results')
    # acc_newtestdata_2source(net=base_model_l2, dataloader=test_loader, device=device, 
    #     rmse_threshold=rmse_threshold, n_peaks=2)
    # print('U-Net-l2_l12 results')
    # acc_newtestdata_2source(net=base_model_l2_l12, dataloader=test_loader, device=device, 
    #     rmse_threshold=rmse_threshold, n_peaks=2)
 
    data_dir='../dataset_folder/test_data'

    # 2 sources
    file_names=['2sources_close_0_20snr', '2sources_1moving_10dbsnr', 
        '3sources_2close_1far_0_20snr', '3sources_1moving_5dbsnr']
    n_sources=[2, 2, 3, 3]
    for file_name, n_source in zip(file_names, n_sources):
        test_data_path=os.path.join(data_dir, f'power_spec_tlcbf_{file_name}.npy')
        test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
        test_tlcbf=normalize_data(x=np.load(test_data_path), axis=(1,2))
        test_tlcbf=torch.as_tensor(np.load(test_data_path)).float()
        test_labels=pd.read_csv(test_label_path)
        test_thetas = torch.as_tensor(test_labels[['src1_theta', 'src2_theta']].to_numpy()).float()
        test_alphas = torch.as_tensor(test_labels[['src1_alpha', 'src2_alpha']].to_numpy()).float()
        test_snrs=torch.as_tensor(test_labels[['snr_db']].to_numpy()).float()

        batch_size=200
        test_dataset=(test_tlcbf, test_thetas, test_alphas, test_snrs)
        folder_name='l2'

        acc_vs_snr_th1, det_rmse_vs_snr_th1, count_vs_snr_th1=acc_newtestdata_Nsource(net=base_model_l2, 
            dataset=test_dataset, device=device, rmse_threshold=rmse_threshold, n_peaks=n_source, 
            batch_size=batch_size, n_sources=n_source)
        np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), count_vs_snr_th1)
        print(f'l2, avg_acc: {acc_vs_snr_th1[:, :, 3].sum() / (20*n_source)}',
                f'avg_detected_rmse: {det_rmse_vs_snr_th1[:, 3].sum() / count_vs_snr_th1[:, 3].sum()}')

        folder_name='l2_l12'
        acc_vs_snr_th1, det_rmse_vs_snr_th1, count_vs_snr_th1=acc_newtestdata_Nsource(net=base_model_l2_l12, 
            dataset=test_dataset, device=device, rmse_threshold=rmse_threshold, n_peaks=n_source, 
            batch_size=batch_size, n_sources=n_source)
        np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), count_vs_snr_th1)
        print(f'l2_l12, avg_acc: {acc_vs_snr_th1[:, :, 3].sum() / (20*n_source)}',
                f'avg_detected_rmse: {det_rmse_vs_snr_th1[:, 3].sum() / count_vs_snr_th1[:, 3].sum()}')
        
        folder_name='tlcbf'
        acc_vs_snr_th1, det_rmse_vs_snr_th1, count_vs_snr_th1=acc_tlcbf(dataset=test_dataset, 
            rmse_threshold=rmse_threshold, batch_size=batch_size, n_sources=n_source, n_peaks=n_source)
        np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_snr_th1)
        np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), count_vs_snr_th1)
        print(f'tlcbf, avg_acc: {acc_vs_snr_th1[:, :, 3].sum() / (20*n_source)}',
                f'avg_detected_rmse: {det_rmse_vs_snr_th1[:, 3].sum() / count_vs_snr_th1[:, 3].sum()}')

import os, time
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# pytorch libraries.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from torchinfo import summary
import model, utils
import warnings
warnings.filterwarnings('ignore')

def get_best_model(training_cfg, model_cfg, net, device):
    checkpoint_path=training_cfg['checkpoint_path']
    optimizer_type=training_cfg['optimizer_type']
    loss_type=training_cfg['loss_type']
    act_fn1=model_cfg['act_fn1']
    model_name=f'complex_model_{act_fn1}_{optimizer_type}_loss={loss_type}'
    model_path=os.path.join(checkpoint_path, model_name)
    best_model_path=os.path.join(model_path, 'best_model_avgacc')
    net=utils.load_model(model_path=best_model_path, 
        model_name=model_name+'_bm_avgacc', device=device, net=net)
    return net

def get_correct_idx(preds_params, gts_params, device, max_val=[80, 5], n_snap=30):
    preds_phi, preds_alpha = max_val[0]*preds_params[:, 0].reshape(-1, 1), max_val[1]*preds_params[:, 1].reshape(-1, 1)
    gts_phi, gts_alpha = gts_params[:, 0], gts_params[:, 1]
    preds_track=preds_phi + preds_alpha*torch.arange(n_snap, device=device) / (n_snap-1)
    gts_track=gts_phi.unsqueeze(2) + gts_alpha.unsqueeze(2)*torch.arange(n_snap, device=device) / (n_snap-1)
    rmse_error=torch.sqrt(((preds_track.unsqueeze(1)-gts_track)**2).mean(-1))
    src_idx =torch.argmin(rmse_error, dim=-1)
    return src_idx

def test_compute_loss_acc(preds_sigs, preds_params, preds_traj, gts_sigs, gts_params, device, 
	loss_type='loss4', n_snap=30, max_val=[80, 5], loss_weights=[1, 1, 1], 
	rmse_threshold=2.4, max_angle=90, n_source=2):
    src_idx=get_correct_idx(preds_params=preds_params, gts_params=gts_params, device=device, 
        max_val=max_val, n_snap=n_snap)
    all_idxes=torch.arange(n_source, device=device)
    mask = (all_idxes.unsqueeze(0) == src_idx.unsqueeze(1))
    t_loss, track_dict, traj_dict, recvsig_dict, sig_dict=utils.compute_loss_and_acc(
		preds_sigs=preds_sigs, preds_params=preds_params, preds_traj=preds_traj, 
		gts_sigs=gts_sigs[mask], gts_params=gts_params.permute(0,2,1)[mask], 
		device=device, loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
		loss_weights=loss_weights, rmse_threshold=rmse_threshold, max_angle=max_angle)
    return t_loss, track_dict, traj_dict, recvsig_dict, sig_dict
    
def get_test_loss_acc(base_model, dataset_cfg, model_cfg, training_cfg, data_loader):
    checkpoint_path=training_cfg['checkpoint_path']
    loss_type=training_cfg['loss_type']
    optimizer_type=training_cfg['optimizer_type']
    act_fn1=model_cfg['act_fn1']
    n_snap, max_val, max_angle=dataset_cfg['n_snap'], dataset_cfg['max_val'], dataset_cfg['max_angle']
    loss_weights=training_cfg['loss_weights']
    model_name=f'complex_model_{act_fn1}_{optimizer_type}_loss={loss_type}'
    model_path=os.path.join(checkpoint_path, model_name)
    print(f'Model-path: {model_path}')

    rmse_threshold=np.round(np.arange(1.2, 3.7, 0.4), decimals=1)
    track_acc1, track_acc2, track_acc_avg={}, {}, {}
    count, det_rmse={}, {}

    test_log_dict={'tloss': {0:0, 1:0}, 'sig_mse': {0:0, 1:0}, 'sig_re': {0:0, 1:0}, 
        'sig_mage': {0:0, 1:0}, 'phase_cossim': {0:0, 1:0}, 'si_sdr': {0:0, 1:0}, 
		'si_snr': {0:0, 1:0}, 'recvsig_mse': {0:0, 1:0}, 'recvsig_re': {0:0, 1:0}, 
		'traj_e1': {0:0, 1:0}, 'traj_e2': {0:0, 1:0}, 'trajdiff_mse': {0:0, 1:0}, 
		'track_acc': {0:0, 1:0}}
    test_hist1=pd.DataFrame(columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
            'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
            'trajdiff_mse', 'track_acc'])
    test_hist2=pd.DataFrame(columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
        'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
        'trajdiff_mse', 'track_acc'])
    avg_test_hist=pd.DataFrame(columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
        'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
        'trajdiff_mse', 'track_acc'])
    
    for rt in rmse_threshold:
        track_acc1[rt], track_acc2[rt]=0, 0
        count[rt], det_rmse[rt]=0, 0
    
    start_time=time.time()
    total_datapoints=0
    for imgs, gts_sigs, gts_params, snr_db in tqdm(data_loader):
        (imgs, gts_sigs, gts_params)=imgs.to(device), gts_sigs.to(device), gts_params.to(device)
        total_datapoints+=imgs.shape[0]
        with torch.no_grad():
            for i in range(2):
                if i==0:
                    preds_sigs, preds_params, preds_traj=base_model(imgs)
                    (t_loss, track_dict, traj_dict, recvsig_dict, sig_dict, 
                        src_idx)=test_compute_loss_acc(preds_sigs=preds_sigs, 
							preds_params=preds_params, preds_traj=preds_traj, 
							gts_sigs=gts_sigs, gts_params=gts_params, device=device, 
                            loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
							loss_weights=loss_weights, max_angle=max_angle)
                    imgs=imgs-recvsig_dict['preds_recvsig'].unsqueeze(1)
                    test_log_dict=utils.update_epoch_log(log_dict=test_log_dict, t_loss=t_loss, 
						sig_dict=sig_dict, traj_dict=traj_dict, recvsig_dict=recvsig_dict, iter=i)
                    track_rmse1=traj_dict['traj_rmse']
                else:
                    preds_sigs, preds_params, preds_traj=base_model(imgs)
                    (t_loss, track_dict, traj_dict, recvsig_dict, sig_dict, 
                        src_idx)=test_compute_loss_acc(preds_sigs=preds_sigs, 
							preds_params=preds_params, preds_traj=preds_traj, 
							gts_sigs=gts_sigs, gts_params=gts_params, device=device, 
                            loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
							loss_weights=loss_weights, max_angle=max_angle)
                    test_log_dict=utils.update_epoch_log(log_dict=test_log_dict, t_loss=t_loss, 
						sig_dict=sig_dict, traj_dict=traj_dict, recvsig_dict=recvsig_dict, iter=i)
                    track_rmse2=traj_dict['traj_rmse']
        for i, th in enumerate(rmse_threshold):
            track_acc1[th]+=(track_rmse1 < th).sum().item()
            track_acc2[th]+=(track_rmse2 < th).sum().item()
            mask1, mask2=(track_rmse1 < th).int(), (track_rmse2 < th).int()
            count[th] += mask1.sum() + mask2.sum()
            det_rmse[th] += (track_rmse1 * mask1).sum() + (track_rmse2 * mask2).sum()
        
    end_time=time.time()
    test_epoch_time=end_time-start_time
    test_hist1=utils.update_val_hist(hist=test_hist1, log_dict=test_log_dict, epoch=0, 
        l=len(data_loader), src_num=0)
    utils.do_print(epoch=0, hist=test_hist1, epoch_tt=test_epoch_time, stage='Test 1st src')
    test_hist2=utils.update_val_hist(hist=test_hist2, log_dict=test_log_dict, epoch=0, 
        l=len(data_loader), src_num=1)
    utils.do_print(epoch=0, hist=test_hist2, epoch_tt=test_epoch_time, stage='Test 2nd src')
    avg_test_hist=utils.update_avg_hist(avg_hist=avg_test_hist, hist1=test_hist1, 
        hist2=test_hist2, epoch=0)
    utils.do_print(epoch=0, hist=avg_test_hist, epoch_tt=test_epoch_time, stage='Test Avg')

    for i, th in enumerate(rmse_threshold):
        track_acc1[th]=(track_acc1[th] / total_datapoints)*100
        track_acc2[th]=(track_acc2[th] / total_datapoints)*100
        track_acc_avg[th]=(track_acc1[th]+track_acc2[th]) / 2
    
    for th in rmse_threshold:
        print(f'rmse_threshold: {th}, track_acc1: {track_acc1[th]:0.2f}, track_acc2: {track_acc2[th]:0.2f}, ',
              f'track_acc_avg: {track_acc_avg[th]}, avg_det_rmse: {det_rmse[th] / count[th]}')

class CustomDataset(data.Dataset):
    def __init__(self, dataset_path, sig_path, label_path, n_sources):
        self.x_img=loadmat(dataset_path)['recv_sig_with_noise']
        self.sig_amp=loadmat(sig_path)['sig_amp']
        self.labels=pd.read_csv(label_path)
        self.last_dim=True
        self.n_sources=n_sources
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img=torch.as_tensor(self.convert_img(x=self.x_img[index])).float()
        sig_nsrc_gt=self.sig_amp[index]
        snr_db=self.labels.iloc[index][['snr_db']].to_numpy()
        sig_gt=torch.as_tensor(self.convert_sig(x=sig_nsrc_gt)).float()
        if self.n_sources==4:
            theta_gt=self.labels.iloc[index][['src1_theta', 'src2_theta', 'src3_theta', 'src4_theta']].to_numpy()
            alpha_gt=self.labels.iloc[index][['src1_alpha', 'src2_alpha', 'src3_theta', 'src4_theta']].to_numpy()
        elif self.n_sources==3:
            theta_gt=self.labels.iloc[index][['src1_theta', 'src2_theta', 'src3_theta']].to_numpy()
            alpha_gt=self.labels.iloc[index][['src1_alpha', 'src2_alpha', 'src3_theta']].to_numpy()
        elif self.n_sources==2:
            theta_gt=self.labels.iloc[index][['src1_theta', 'src2_theta']].to_numpy()
            alpha_gt=self.labels.iloc[index][['src1_alpha', 'src2_alpha']].to_numpy()
        doa_param=torch.as_tensor(np.array([theta_gt, alpha_gt])).float()
        return img, sig_gt, doa_param, snr_db
    
    def convert_img(self, x):
        if self.last_dim:
            x=np.concatenate((x.real[None, :, :, None], x.imag[None, :, :, None]), axis=-1)
        else:
            x=np.concatenate((x.real[None, ...], x.imag[None, ...]), axis=0)
        return x
    
    def convert_sig(self, x):
        if self.last_dim:
            x=np.concatenate((x.real[..., None], x.imag[..., None]), axis=-1)
        else:
            x=np.concatenate((x.real[None, ...], x.imag[None, ...]), axis=0)
        return x

def loss_acc_testdata(net, dataset, device, dataset_cfg, training_cfg, batch_size=200, 
    n_source=2):
    checkpoint_path=training_cfg['checkpoint_path']
    loss_type=training_cfg['loss_type']
    optimizer_type=training_cfg['optimizer_type']
    act_fn1=model_cfg['act_fn1']
    n_snap, max_val, max_angle=dataset_cfg['n_snap'], dataset_cfg['max_val'], dataset_cfg['max_angle']
    loss_weights=training_cfg['loss_weights']
    model_name=f'complex_model_{act_fn1}_{optimizer_type}_loss={loss_type}'
    model_path=os.path.join(checkpoint_path, model_name)
    print(f'Model-path: {model_path}')

    rmse_threshold=np.round(np.arange(1.2, 3.7, 0.4), decimals=1)
    data_loader=data.DataLoader(dataset=dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False, pin_memory=True, num_workers=10)
    
    sig_rel_error=torch.zeros((20, n_sources))
    acc_vs_param=torch.zeros((20, n_sources, len(rmse_threshold)))
    counts_vs_param=np.zeros((20, n_sources, len(rmse_threshold)))
    det_rmse_vs_param=np.zeros((20, n_sources, len(rmse_threshold)))

    t=tqdm(iterable=data_loader, leave=False)
    for i, (imgs, gts_sigs, gts_params, snr_db) in enumerate(t):
        (imgs, gts_sigs, gts_params)=imgs.to(device), gts_sigs.to(device), gts_params.to(device)
        with torch.no_grad():
            for j in range(n_source):
                preds_sigs, preds_params, preds_traj = net(imgs)
                (t_loss, track_dict, traj_dict, recvsig_dict, sig_dict)=test_compute_loss_acc(preds_sigs=preds_sigs, 
                    preds_params=preds_params, preds_traj=preds_traj, gts_sigs=gts_sigs, 
                    gts_params=gts_params, device=device, loss_type=loss_type, n_snap=n_snap, 
                    max_val=max_val, max_angle=max_angle, rmse_threshold=2.4, 
                    n_source=n_source)
                imgs=imgs-recvsig_dict['preds_recvsig'].unsqueeze(1)
                sig_rel_error[i, j] = sig_dict['sig_re']
                for k, th in enumerate(rmse_threshold):
                    acc_vs_param[i, j, k]=((traj_dict['traj_rmse'] < th).sum() / batch_size)*100
                    mask=(traj_dict['traj_rmse'] < th).int()
                    counts_vs_param[i, j, k]=mask.sum()
                    det_rmse_vs_param[i, j, k]=(traj_dict['traj_rmse']*mask).sum()
    return acc_vs_param, sig_rel_error, counts_vs_param, det_rmse_vs_param

if __name__=="__main__":
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    print(f'Using device: {device}')

    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    config_file = './config.yml'
    dataset_cfg, model_cfg, training_cfg, metrics_cfg = utils.get_configurations(
        config_file=config_file)
    train_df, val_df, test_df=utils.get_annotation_file(
        dataset_cfg=dataset_cfg, shuffle=True)
    
    test_loader=utils.get_dataloader(csv_file=test_df, dataset_cfg=dataset_cfg, 
        dataset_name='Test', is_traindataset=False)
    
    actfn_by_name = { 'relu': nn.ReLU, 'tanh': nn.Tanh, 'swish': nn.SiLU, 'sigmoid': nn.Sigmoid, 'identity': nn.Identity, 
        'leaky_relu': nn.LeakyReLU }
    
    base_model=utils.get_model(model_cfg=model_cfg, actfn_by_name=actfn_by_name)
    total_params=sum(p.numel() for p in base_model.parameters())
    print(f'Total parameters: {total_params}')

    base_model=get_best_model(training_cfg=training_cfg, model_cfg=model_cfg, 
        net=base_model, device=device)
    base_model=base_model.to(device)
    base_model.eval()

    get_test_loss_acc(base_model=base_model, dataset_cfg=dataset_cfg, model_cfg=model_cfg, 
        training_cfg=training_cfg, data_loader=test_loader)

    rmse_threshold=np.round(np.arange(1.2, 3.7, 0.4), decimals=1)
    folder_name='l2_largedata'
    os.makedirs(os.path.join('./results', folder_name), exist_ok=True)
    data_dir='../dataset_folder/test_data'

    # Test-Data:1.
    n_sources=2
    file_name='2sources_close_0_20snr'
    
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)

    # for idx, (img, sig_gt, doa_param, snr_db) in enumerate(test_dataset):
    #     print(f'idx: {idx}')
    #     print(f'img.shape: {img.shape}')
    #     print(f'sig_gt.shape: {sig_gt.shape}')
    #     print(f'doa_param: {doa_param}')
    #     print(f'snr_db: {snr_db}')
    #     break
    # breakpoint()

    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)
    
    # Test-Data:2.
    n_sources=2
    file_name='2sources_1moving_10dbsnr'
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)
    
    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)
    
    # Test-Data: 3.
    n_sources=3
    file_name='3sources_2close_1far_0_20snr'
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)
    
    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)

    # Test-Data: 4.
    n_sources=3
    file_name='3sources_1moving_5dbsnr'
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)
    
    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)
    
    # Test-Data: 5.
    n_sources=3
    file_name='3sources_1moving_15dbsnr'
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)
    
    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)
    
    # Test-Data: 6.
    n_sources=3
    file_name='3sources_2close_1far_10_20snr'
    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    test_dataset=CustomDataset(dataset_path=test_data_path, sig_path=test_sig_path, 
        label_path=test_label_path, n_sources=n_sources)
    
    (acc_vs_param, sig_rel_error, counts_vs_param, 
        det_rmse_vs_param) = loss_acc_testdata(net=base_model, 
            dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
            training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    print(f'file_name: {file_name}, avg_acc: {acc_vs_param[:, :, 3].sum() / (n_sources*20)}')
    print(f'avg_detected_rmse: {det_rmse_vs_param[:, :, 3].sum() / counts_vs_param[:, :, 3].sum()}')
    np.save(os.path.join(f'./results/{folder_name}', f'acc_snr_th_{file_name}.npy'), acc_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name}.npy'), sig_rel_error)
    np.save(os.path.join(f'./results/{folder_name}', f'count_vs_snr_th_{file_name}.npy'), counts_vs_param)
    np.save(os.path.join(f'./results/{folder_name}', f'det_rmse_vs_snr_th_{file_name}.npy'), det_rmse_vs_param)
    
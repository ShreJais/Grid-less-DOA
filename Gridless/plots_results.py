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
import model, utils, latex_fig
import warnings
warnings.filterwarnings('ignore')

import testdata_loss_acc

def plot_signal_relative_error():
    folder_name='l2_largedata'
    file_name1, file_name2='2sources_close_0_20snr', '2sources_1moving_10dbsnr'
    file_name3, file_name4='3sources_2close_1far_0_20snr', '3sources_1moving_5dbsnr'
    n_sources=[2, 3]
    sig_rel_path1=os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name1}.npy')
    sig_rel_path2=os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name2}.npy')
    sig_rel_path3=os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name3}.npy')
    sig_rel_path4=os.path.join(f'./results/{folder_name}', f'sig_rel_error_{file_name4}.npy')

    sig_rel1, sig_rel2=np.load(sig_rel_path1), np.load(sig_rel_path2)
    sig_rel3, sig_rel4=np.load(sig_rel_path3), np.load(sig_rel_path4)

    snr_val, changing_theta=np.arange(0, 20, 1), np.arange(-60, 55, 6)
    markers=['*', 's', 'v', 'o']
    line_styles=['-', '--', '-.', ':']
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    ax[0, 0].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[0, 1].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[1, 0].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[1, 1].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)

    for i in range(n_sources[0]):
        ax[0, 0].plot(snr_val.tolist(), sig_rel1[:, i], 
            linestyle=line_styles[i], marker=markers[i], markersize=10, label=f'AE: S-{i+1}')
        ax[0, 1].plot(changing_theta.tolist(), sig_rel2[:, i], 
            linestyle=line_styles[i], marker=markers[i], markersize=10, label=f'AE: S-{i+1}')
    ax[0, 0].plot(snr_val.tolist(), sig_rel1.sum(-1)/n_sources[0], 
        linestyle=line_styles[-1], marker=markers[-1], markersize=10, label='Avg. AE')
    ax[0, 1].plot(changing_theta.tolist(), sig_rel2.sum(-1)/n_sources[0], 
        linestyle=line_styles[-1], marker=markers[-1], markersize=10, label='Avg. AE')
    
    ax[0, 0].set_xlabel('SNR', fontsize=20)
    ax[0, 0].set_ylabel('Amplitude Error [AE] (%)', fontsize=20)
    ax[0, 0].set_ylim([0, 100.5])
    ax[0, 0].set_xticks(np.arange(0, 20.01, 4).tolist())
    ax[0, 0].set_yticks(np.arange(0, 101, 20).tolist())
    ax[0, 0].legend(frameon=True, fontsize=15, loc=1)
    ax[0, 0].grid(color='gray')

    ax[0, 1].set_xlabel(r'$\phi_0$', fontsize=20)
    ax[0, 1].set_ylabel('Amplitude Error [AE] (%)', fontsize=20)
    ax[0, 1].set_ylim([0, 100.5])
    ax[0, 1].set_xticks(np.arange(-60, 60.01, 20).tolist())
    ax[0, 1].set_yticks(np.arange(0, 101, 20).tolist())
    ax[0, 1].legend(frameon=True, fontsize=15, loc=2)
    ax[0, 1].grid(color='gray')
    
    # ax[0, 0].text(0.5,-0.35, "(a)", size=15, ha="center", transform=ax[0, 0].transAxes)
    # ax[0, 1].text(0.5,-0.35, "(b)", size=15, ha="center", transform=ax[0, 1].transAxes)

    for i in range(n_sources[1]):
        ax[1, 0].plot(snr_val.tolist(), sig_rel3[:, i], 
            linestyle=line_styles[i], marker=markers[i], markersize=10, label=f'AE: S-{i+1}')
        ax[1, 1].plot(changing_theta.tolist(), sig_rel4[:, i], 
            linestyle=line_styles[i], marker=markers[i], markersize=10, label=f'AE: S-{i+1}')
    ax[1, 0].plot(snr_val.tolist(), sig_rel3.sum(-1)/n_sources[1], 
        linestyle=line_styles[-1], marker=markers[-1], markersize=10, label='Avg. AE')
    ax[1, 1].plot(changing_theta.tolist(), sig_rel4.sum(-1)/n_sources[1], 
        linestyle=line_styles[-1], marker=markers[-1], markersize=10, label='Avg. AE')
    
    ax[1, 0].set_xlabel('SNR', fontsize=20)
    ax[1, 0].set_ylabel('Amplitude Error [AE] (%)', fontsize=20)
    ax[1, 0].set_ylim([0, 100.5])
    ax[1, 0].set_xticks(np.arange(0, 20.01, 4).tolist())
    ax[1, 0].set_yticks(np.arange(0, 101, 20).tolist())
    ax[1, 0].legend(frameon=True, fontsize=15, loc=1)
    ax[1, 0].grid(color='gray')

    ax[1, 1].set_xlabel(r'$\phi_0$', fontsize=20)
    ax[1, 1].set_ylabel('Amplitude Error [AE] (%) ', fontsize=20)
    ax[1, 1].set_ylim([0, 100.5])
    ax[1, 1].set_xticks(np.arange(-60, 60.01, 20).tolist())
    ax[1, 1].set_yticks(np.arange(0, 101, 20).tolist())
    ax[1, 1].legend(frameon=True, fontsize=15, loc=2)
    ax[1, 1].grid(color='gray')
    
    # ax[1, 0].text(0.5,-0.35, "(c)", size=15, ha="center", transform=ax[1, 0].transAxes)
    # ax[1, 1].text(0.5,-0.35, "(d)", size=15, ha="center", transform=ax[1, 1].transAxes)

    ax[0,0].set_title(f'(a)', fontsize=20); ax[0,1].set_title(f'(b)', fontsize=20)
    ax[1,0].set_title(f'(c)', fontsize=20); ax[1, 1].set_title(f'(d)', fontsize=20)
    # fig.subplots_adjust(hspace=0.5, wspace=0.9)
    latex_fig.savefig(f'subplots_sig_rel_error_{n_sources[0]}_{n_sources[1]}.png', tight_layout=True)
    plt.close()

def plot_accuracy():
    folder_names=['tlcbf', 'l2', 'l2_l12', f'l2_largedata']
    label_names={'tlcbf': 'TL-CBF', 'l2': r'U-Net-$l_{2}$', 'l2_l12': r'U-Net-$l_{21}$',
        f'l2_largedata': 'Proposed'}
    n_sources=[2, 3]
    markers=['*', 's', 'v', 'o']
    line_styles=['-', '--', '-.', ':']

    snr_val, changing_theta=np.arange(0, 20, 1), np.arange(-60, 55, 6)

    file_names1=['2sources_close_0_20snr', '2sources_1moving_10dbsnr'] 
    file_names2=['3sources_2close_1far_0_20snr', '3sources_1moving_5dbsnr']
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    ax[0, 0].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[0, 1].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[1, 0].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    ax[1, 1].tick_params(direction='out', length=4, width=1, colors='k', grid_color='k', 
        labelsize=20)
    for i, fn in enumerate(folder_names):
        for j, f_name in enumerate(file_names1):
            acc=np.load(os.path.join(f'./results/{fn}', f'acc_snr_th_{f_name}.npy'))
            if j==0:
                ax[0, 0].plot(snr_val.tolist(), acc[:, :, 3].sum(1) / n_sources[0], 
                    linestyle=line_styles[i], marker=markers[i], markersize=10, label=label_names[fn])
            else:
                ax[0, 1].plot(changing_theta.tolist(), acc[:, :, 3].sum(1) / n_sources[0], 
                    linestyle=line_styles[i], marker=markers[i], markersize=10, label=label_names[fn])
    ax[0, 0].set_xlabel('SNR', fontsize=20)
    ax[0, 0].set_ylabel('Accuracy (%)', fontsize=20)
    ax[0, 0].set_ylim([0, 100.5])
    ax[0, 0].set_xticks(np.arange(0, 20.01, 4).tolist())
    ax[0, 0].set_yticks(np.arange(0, 101, 20).tolist())
    ax[0, 0].legend(frameon=True, fontsize=15)
    ax[0, 0].grid(color='gray')

    ax[0, 1].set_xlabel(r'$\phi_0$', fontsize=20)
    ax[0, 1].set_ylabel('Accuracy (%)', fontsize=20)
    ax[0, 1].set_ylim([0, 100.5])
    ax[0, 1].set_xticks(np.arange(-60, 60.01, 20).tolist())
    ax[0, 1].set_yticks(np.arange(0, 101, 20).tolist())
    ax[0, 1].legend(frameon=True, fontsize=15)
    ax[0, 1].grid(color='gray')

    for i, fn in enumerate(folder_names):
        for j, f_name in enumerate(file_names2):
            acc=np.load(os.path.join(f'./results/{fn}', f'acc_snr_th_{f_name}.npy'))
            if j==0:
                ax[1, 0].plot(snr_val.tolist(), acc[:, :, 3].sum(1) / n_sources[1], 
                    linestyle=line_styles[i], marker=markers[i], markersize=10, label=label_names[fn])
            else:
                ax[1, 1].plot(changing_theta.tolist(), acc[:, :, 3].sum(1) / n_sources[1], 
                    linestyle=line_styles[i], marker=markers[i], markersize=10, label=label_names[fn])
    
    ax[1, 0].set_xlabel('SNR', fontsize=20)
    ax[1, 0].set_ylabel('Accuracy (%)', fontsize=20)
    ax[1, 0].set_ylim([0, 100.5])
    ax[1, 0].set_xticks(np.arange(0, 20.01, 4).tolist())
    ax[1, 0].set_yticks(np.arange(0, 101, 20).tolist())
    ax[1, 0].legend(frameon=True, fontsize=15)
    ax[1, 0].grid(color='gray')

    ax[1, 1].set_xlabel(r'$\phi_0$', fontsize=20)
    ax[1, 1].set_ylabel('Accuracy (%)', fontsize=20)
    ax[1, 1].set_ylim([0, 100.5])
    ax[1, 1].set_xticks(np.arange(-60, 60.01, 20).tolist())
    ax[1, 1].set_yticks(np.arange(0, 101, 20).tolist())
    ax[1, 1].legend(frameon=True, fontsize=15)
    ax[1, 1].grid(color='gray')
    ax[0,0].set_title(f'(a)', fontsize=20); ax[0,1].set_title(f'(b)', fontsize=20)
    ax[1,0].set_title(f'(c)', fontsize=20); ax[1, 1].set_title(f'(d)', fontsize=20)
    # fig.subplots_adjust(hspace=0.5, wspace=0.9)
    latex_fig.savefig(f'subplots_acc_snr_phi0_{n_sources[0]}_{n_sources[1]}.png', 
        tight_layout=True, pad_inches=0.2)
    plt.close()

def getimg_nsources(net, dataset, device, dataset_cfg, training_cfg, batch_size=200,n_source=2):
    n_snap, max_val, max_angle=dataset_cfg['n_snap'], dataset_cfg['max_val'], dataset_cfg['max_angle']
    loss_weights=training_cfg['loss_weights']
    loss_type=training_cfg['loss_type']
    data_loader=data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, 
        drop_last=False, pin_memory=True, num_workers=10)
    new_img=np.zeros((4000, n_sources, 8, 30), dtype=np.complex128)
    gt_img=np.zeros((4000, n_sources, 8, 30), dtype=np.complex128)
    pred_thetas=np.zeros((4000, n_sources))
    pred_alphas=np.zeros((4000, n_sources))

    prev_count=0
    curr_count=0
    for i, (imgs, gts_sigs, gts_params, snr_db) in enumerate(tqdm(data_loader)):
        (imgs, gts_sigs, gts_params)=imgs.to(device), gts_sigs.to(device), gts_params.to(device)
        orig_imgs=imgs
        curr_count+=imgs.shape[0]
        with torch.no_grad():
            for j in range(n_sources):
                preds_sigs, preds_params, preds_traj=net(imgs)
                (t_loss, track_dict, traj_dict, 
                    recvsig_dict, sig_dict)=testdata_loss_acc.test_compute_loss_acc(
                        preds_sigs=preds_sigs, preds_params=preds_params, 
                        preds_traj=preds_traj, gts_sigs=gts_sigs, gts_params=gts_params, 
                        device=device, loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
                        loss_weights=loss_weights, rmse_threshold=2.4, 
                        max_angle=max_angle, n_source=n_source)
                
                imgs=imgs-recvsig_dict['preds_recvsig'].unsqueeze(1)
                new_img[prev_count:curr_count, j, :, :]=get_complex_signal(x=imgs)
                pred_thetas[prev_count: curr_count, j]=preds_params[:, 0] * 80
                pred_alphas[prev_count: curr_count, j]=preds_params[:, 1] * 5
        prev_count=curr_count
    return new_img, gt_img, pred_thetas, pred_alphas

def get_complex_signal(x):
    x=x.squeeze()
    x=x[..., 0] + 1j*x[..., 1]
    return x

def power_cbf(n_sensors:int, sig_recv, grid:dict, wavelen, mic_dist, n_snap=30):
    theta_grid, alpha_grid=grid['theta_grid'], grid['alpha_grid']
    power_spec=np.zeros((len(theta_grid), len(alpha_grid)))
    const=-1j*2*np.pi*(mic_dist/wavelen)
    cov_mat=sig_recv.T[..., None] @ np.conjugate(sig_recv.T[:, None, :]) # (n_snap, n_sensors, n_sensors)
    cov_mat=cov_mat.numpy() if torch.is_tensor(cov_mat) else cov_mat
    for i in range(len(theta_grid)):
        for j in range(len(alpha_grid)):
            delta_alpha=alpha_grid[j]/(n_snap-1)
            doa=(theta_grid[i]+delta_alpha*np.arange(n_snap)).reshape(-1, 1) # (n_snap, 1)
            n_phi=np.sin(doa*np.pi/180.)*np.arange(n_sensors) # (n_snap, n_sensors)
            s_vec=np.exp(const*n_phi) # (n_snap, n_sensors)
            power_vec=np.abs(np.conjugate(s_vec[:, None, :]) @ cov_mat @ s_vec[:, :, None])
            power_spec[i, j]=power_vec.sum()/n_snap
    return power_spec.T

def get_power_spec(dataset, n_sensors:int, grid:dict, wavelen, mic_dist, n_snap=30):
    theta_grid, alpha_grid=grid['theta_grid'], grid['alpha_grid']
    power_spec=np.zeros((len(dataset), len(alpha_grid), len(theta_grid)))
    data_loader=data.DataLoader(dataset=dataset, batch_size=200, shuffle=False, 
        drop_last=False, pin_memory=True, num_workers=10)
    curr_count=0
    prev_count=0
    for idx, (imgs, _, _, _)in enumerate(data_loader):
        curr_count+=imgs.shape[0]
        for i in range(imgs.shape[0]):
            sig_recv=get_complex_signal(imgs[i])
            power_spec[int(prev_count+i), :, :]=power_cbf(n_sensors=n_sensors, sig_recv=sig_recv, grid=grid, wavelen=wavelen, mic_dist=mic_dist)
        prev_count=curr_count
    return power_spec

def normalize_data(x, axis=(2, 3), maxima=None):
    minima=np.min(x, axis=axis, keepdims=True)
    maxima=maxima if maxima is not None else np.max(x, axis=axis, keepdims=True)
    x=(x-minima)/(maxima-minima)
    return x, minima, maxima

def get_spectrums_2source(data_dir, file_name, n_sources=2):
    n_sensors= 8
    n_snap, freq, ss = 30, 300, 340  
    wavelen=ss/freq
    mic_dist=0.5*wavelen
    theta_grid=np.arange(-80, 81, 1)
    alpha_grid=np.arange(-5, 6, 1)
    grid = {'theta_grid': theta_grid, 'alpha_grid': alpha_grid}

    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    # test_tlcbf=np.load(os.path.join(data_dir, f'power_spec_tlcbf_{file_name}.npy'))
    test_label=pd.read_csv(test_label_path)
    test_img=loadmat(test_data_path)['recv_sig_with_noise']
    test_dataset=testdata_loss_acc.CustomDataset(
        dataset_path=test_data_path, sig_path=test_sig_path, label_path=test_label_path, 
        n_sources=n_sources)
    new_img, gt_img, pred_thetas, pred_alphas=getimg_nsources(
        net=base_model, dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
        training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    
    if os.path.isfile(os.path.join('original_spec', f'orig_spec_{file_name}.npy')):
        orig_spec=np.load(os.path.join('original_spec', f'orig_spec_{file_name}.npy'))
    else:
        orig_spec=get_power_spec(dataset=test_dataset, n_sensors=n_sensors, 
            grid=grid, wavelen=wavelen, mic_dist=mic_dist)
        os.makedirs('original_spec', exist_ok=True)
        np.save(os.path.join('original_spec', f'orig_spec_{file_name}.npy'), orig_spec)

    original_power_spec=np.zeros((100, 11, 161))
    pred_power_spec=np.zeros((100, 2, 11, 161))
    gt_thetas, gt_alphas, gt_snrs=np.zeros((100, 2)), np.zeros((100, 2)), np.zeros((100,))
    orig_img=np.zeros((100, 8, 30), dtype=np.complex128)
    x_img=np.zeros((100, 2, 8, 30), dtype=np.complex128)
    pthetas, palphas=np.zeros((100, n_sources)), np.zeros((100, n_sources))

    for i in range(20):
        orig_img[5*i: (i+1)*5, :, :]=test_img[200*i: 200*i +5, :, :]
        x_img[5*i: (i+1)*5, :, :, :]=new_img[200*i: 200*i +5, :, :, :]
        # gt_x_img[5*i: (i+1)*5, :, :, :]=gt_img[200*i: 200*i +5, :, :, :]
        original_power_spec[5*i: (i+1)*5, :, :]=orig_spec[200*i: 200*i+5, :, :]
        gt_thetas[5*i: (i+1)*5, :]=test_label[['src1_theta', 'src2_theta']][200*i: 200*i+5].to_numpy()
        gt_alphas[5*i: (i+1)*5, :]=test_label[['src1_alpha', 'src2_alpha']][200*i: 200*i+5].to_numpy()
        gt_snrs[5*i: (i+1)*5]=test_label['snr_db'][200*i: 200*i+5].to_numpy()
        pthetas[5*i: (i+1)*5, :]=pred_thetas[200*i: 200*i+5, :]
        palphas[5*i: (i+1)*5, :]=pred_alphas[200*i: 200*i+5, :]

    for i in tqdm(range(100)):
        for j in range(n_sources):
            pred_power_spec[i, j, :, :]=power_cbf(n_sensors=n_sensors, sig_recv=x_img[i, j, :, :], 
                wavelen=wavelen, grid=grid, mic_dist=mic_dist, n_snap=n_snap)
            # gt_power_spec[i, j, :, :]=power_cbf(n_sensors=n_sensors, sig_recv=gt_x_img[i, j, :, :], 
            #     wavelen=wavelen, grid=grid, mic_dist=mic_dist)
    
    figure_path_pdf=f'./Images/tlcbf_spec_{file_name}_pdf'
    os.makedirs(figure_path_pdf, exist_ok=True)
    figure_path_png=f'./Images/tlcbf_spec_{file_name}_png'
    os.makedirs(figure_path_png, exist_ok=True)

    original_power_spec, _, maxima=normalize_data(x=original_power_spec, axis=(1, 2), maxima=None)
    # gt_power_spec[:, 0, :, :]=gt_power_spec[:, 0, :, :] / maxima
    # gt_power_spec[:, 1, :, :]=gt_power_spec[:, 1, :, :] / maxima
    pred_power_spec[:, 0, :, :]=pred_power_spec[:, 0, :, :] / maxima
    pred_power_spec[:, 1, :, :]=pred_power_spec[:, 1, :, :] / maxima

    for i in tqdm(range(100)):
        fig, ax =plt.subplots(nrows=1, ncols=3, figsize=(18, 5), constrained_layout=True)
        ax[0].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k',
            labelsize=20)
        ax[0].imshow(original_power_spec[i, :, :], extent=(-82, 82, 6, -6), interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[0].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[0].scatter(pthetas[i, 0], palphas[i, 0], marker='o', s=90, color='r', facecolors='none')
        ax[0].set_title(f'(a) P={(abs(orig_img[i])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[0].set_xlabel(r'$\phi$', fontsize=20)
        ax[0].set_ylabel(r'$\alpha$', fontsize=20)
        ax[0].set_xlim([-80, 80])
        ax[0].set_ylim([5.25, -5.25])
        ax[0].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[0].set_yticks(np.arange(-5, 6, 2).tolist())

        ax[1].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k', 
            labelsize=20)
        ax[1].imshow(pred_power_spec[i, 0, :, :], extent=(-82, 82, 6, -6), interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[1].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[1].scatter(pthetas[i, 1], palphas[i, 1], marker='o', s=90, color='r', facecolors='none')
        ax[1].set_title(f'(b) P={(abs(x_img[i, 0])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[1].set_xlabel(r'$\phi$', fontsize=20)
        ax[1].set_ylabel(r'$\alpha$', fontsize=20)
        ax[1].set_xlim([-80, 80])
        ax[1].set_ylim([5.25, -5.25])
        ax[1].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[1].set_yticks(np.arange(-5, 6, 2).tolist())

        ax[2].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k', 
            labelsize=20)
        ax[2].imshow(pred_power_spec[i, 1, :, :], extent=(-82, 82, 6, -6), interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[2].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[2].set_title(f'(c) P={(abs(x_img[i, 1])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[2].set_xlabel(r'$\phi$', fontsize=20)
        ax[2].set_ylabel(r'$\alpha$', fontsize=20)
        ax[2].set_xlim([-80, 80])
        ax[2].set_ylim([5.25, -5.25])
        ax[2].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[2].set_yticks(np.arange(-5, 6, 2).tolist())

        im = ax[0].imshow(original_power_spec[i, :, :], extent=(-82, 82, 6, -6), interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        # fig.suptitle(f'{i}')
        cbar=plt.colorbar(im, ax=ax.ravel().tolist(), orientation='vertical', aspect=20)
        cbar.ax.tick_params(labelsize=20)
        # fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.1, right=1., top=0.9, bottom=0.1)
        if file_name=='2sources_1moving_10dbsnr':
            fig.savefig(os.path.join(figure_path_pdf, f'{i}_phi_{gt_thetas[i, 1]}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path_png, f'{i}_phi_{gt_thetas[i, 1]}.png'), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(figure_path_pdf, f'{i}_snrdb_{gt_snrs[i]}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path_png, f'{i}_snrdb_{gt_snrs[i]}.png'), bbox_inches='tight')
        plt.close()

def get_spectrums_3source(data_dir, file_name, n_sources=3):
    n_sensors= 8
    n_snap, freq, ss = 30, 300, 340  
    wavelen=ss/freq
    mic_dist=0.5*wavelen
    theta_grid=np.arange(-80, 81, 1)
    alpha_grid=np.arange(-5, 6, 1)
    grid = {'theta_grid': theta_grid, 'alpha_grid': alpha_grid}

    test_data_path=os.path.join(data_dir, f'recv_sig_{file_name}.mat')
    test_sig_path=os.path.join(data_dir, f'sig_amp_and_noise_{file_name}.mat')
    test_label_path=os.path.join(data_dir, f'testdata_{file_name}.csv')
    # test_tlcbf=np.load(os.path.join(data_dir, f'power_spec_tlcbf_{file_name}.npy'))
    test_label=pd.read_csv(test_label_path)
    test_img=loadmat(test_data_path)['recv_sig_with_noise']
    test_dataset=testdata_loss_acc.CustomDataset(
        dataset_path=test_data_path, sig_path=test_sig_path, label_path=test_label_path, 
        n_sources=n_sources)
    new_img, gt_img, pred_thetas, pred_alphas=getimg_nsources(
        net=base_model, dataset=test_dataset, device=device, dataset_cfg=dataset_cfg, 
        training_cfg=training_cfg, batch_size=200, n_source=n_sources)
    if os.path.isfile(os.path.join('original_spec', f'orig_spec_{file_name}.npy')):
        orig_spec=np.load(os.path.join('original_spec', f'orig_spec_{file_name}.npy'))
    else:
        orig_spec=get_power_spec(dataset=test_dataset, n_sensors=n_sensors, 
            grid=grid, wavelen=wavelen, mic_dist=mic_dist)
        os.makedirs('original_spec', exist_ok=True)
        np.save(os.path.join('original_spec', f'orig_spec_{file_name}.npy'), orig_spec)

    original_power_spec=np.zeros((100, 11, 161))
    pred_power_spec=np.zeros((100, n_sources, 11, 161))
    gt_thetas, gt_alphas, gt_snrs=np.zeros((100, n_sources)), np.zeros((100, n_sources)), np.zeros((100,))
    pthetas, palphas=np.zeros((100, n_sources)), np.zeros((100, n_sources))
    orig_img=np.zeros((100, 8, 30), dtype=np.complex128)
    x_img=np.zeros((100, n_sources, 8, 30), dtype=np.complex128)

    for i in range(20):
        orig_img[5*i: (i+1)*5, :, :]=test_img[200*i: 200*i +5, :, :]
        x_img[5*i: (i+1)*5, :, :, :]=new_img[200*i: 200*i+5, :, :, :]
        # gt_x_img[5*i: (i+1)*5, :, :, :]=gt_img[200*i: 200*i +5, :, :, :]
        original_power_spec[5*i: (i+1)*5, :, :]=orig_spec[200*i: 200*i+5, :, :]
        gt_thetas[5*i: (i+1)*5, :]=test_label[['src1_theta', 'src2_theta', 'src3_theta']][200*i: 200*i+5].to_numpy()
        gt_alphas[5*i: (i+1)*5, :]=test_label[['src1_alpha', 'src2_alpha', 'src3_alpha']][200*i: 200*i+5].to_numpy()
        gt_snrs[5*i: (i+1)*5]=test_label['snr_db'][200*i: 200*i+5].to_numpy()
        pthetas[5*i: (i+1)*5, :]=pred_thetas[200*i: 200*i+5, :]
        palphas[5*i: (i+1)*5, :]=pred_alphas[200*i: 200*i+5, :]
    
    for i in tqdm(range(100)):
        for j in range(n_sources):
            pred_power_spec[i, j, :, :]=power_cbf(n_sensors=n_sensors, sig_recv=x_img[i, j, :, :], 
                wavelen=wavelen, grid=grid, mic_dist=mic_dist, n_snap=n_snap)
            # gt_power_spec[i, j, :, :]=power_cbf(n_sensors=n_sensors, sig_recv=gt_x_img[i, j, :, :], 
            #     wavelen=wavelen, grid=grid, mic_dist=mic_dist)
    
    figure_path_pdf=f'./Images/tlcbf_spec_{file_name}_pdf'
    os.makedirs(figure_path_pdf, exist_ok=True)
    figure_path_png=f'./Images/tlcbf_spec_{file_name}_png'
    os.makedirs(figure_path_png, exist_ok=True)

    original_power_spec, _, maxima=normalize_data(x=original_power_spec, axis=(1, 2))
    # gt_power_spec[:, 0, :, :]=gt_power_spec[:, 0, :, :] / maxima
    # gt_power_spec[:, 1, :, :]=gt_power_spec[:, 1, :, :] / maxima
    pred_power_spec[:, 0, :, :]=pred_power_spec[:, 0, :, :] / maxima
    pred_power_spec[:, 1, :, :]=pred_power_spec[:, 1, :, :] / maxima
    pred_power_spec[:, 2, :, :]=pred_power_spec[:, 2, :, :] / maxima

    for i in tqdm(range(100)):
        fig, ax =plt.subplots(nrows=2, ncols=2, figsize=(12,8), constrained_layout=True)
        ax[0, 0].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k', labelsize=20)
        ax[0, 0].imshow(original_power_spec[i, :, :], extent=(-82, 82, 6, -6), 
            interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[0, 0].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[0, 0].scatter(pthetas[i, 0], palphas[i, 0], marker='o', s=90, color='r', facecolors='none')
        ax[0, 0].set_title(f'(a) P={(abs(orig_img[i])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[0, 0].set_xlabel(r'$\phi$', fontsize=20)
        ax[0, 0].set_ylabel(r'$\alpha$', fontsize=20)
        ax[0, 0].set_xlim([-80, 80])
        ax[0, 0].set_ylim([5.25, -5.25])
        ax[0, 0].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[0, 0].set_yticks(np.arange(-5, 6, 2).tolist())

        ax[0, 1].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k', labelsize=20)
        ax[0, 1].imshow(pred_power_spec[i, 0, :, :], extent=(-82, 82, 6, -6), 
            interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[0, 1].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[0, 1].scatter(pthetas[i, 1], palphas[i, 1], marker='o', s=90, color='r', facecolors='none')
        ax[0, 1].set_title(f'(b) P={(abs(x_img[i, 0])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[0, 1].set_xlabel(r'$\phi$', fontsize=20)
        ax[0, 1].set_ylabel(r'$\alpha$', fontsize=20)
        ax[0, 1].set_xlim([-80, 80])
        ax[0, 1].set_ylim([5.25, -5.25])
        ax[0, 1].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[0, 1].set_yticks(np.arange(-5, 6, 2).tolist())

        ax[1, 0].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k',labelsize=20)
        ax[1, 0].imshow(pred_power_spec[i, 1, :, :], extent=(-82, 82, 6, -6), 
            interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[1, 0].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[1, 0].scatter(pthetas[i, 2], palphas[i, 2], marker='o', s=90, color='r', facecolors='none')
        ax[1, 0].set_title(f'(c) P={(abs(x_img[i, 1])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[1, 0].set_xlabel(r'$\phi$', fontsize=20)
        ax[1, 0].set_ylabel(r'$\alpha$', fontsize=20)
        ax[1, 0].set_xlim([-80, 80])
        ax[1, 0].set_ylim([5.25, -5.25])
        ax[1, 0].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[1, 0].set_yticks(np.arange(-5, 6, 2).tolist())
        
        ax[1, 1].tick_params(direction='out', length=4, width=2, colors='k', grid_color='k', labelsize=20)
        ax[1, 1].imshow(pred_power_spec[i, 2, :, :], extent=(-82, 82, 6, -6), 
            interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax[1, 1].scatter(gt_thetas[i], gt_alphas[i], marker='x', s=90, color='r')
        ax[1, 1].set_title(f'(d) P={(abs(x_img[i, 2])**2).sum() / 240 :4.2f}', fontsize=20)
        ax[1, 1].set_xlabel(r'$\phi$', fontsize=20)
        ax[1, 1].set_ylabel(r'$\alpha$', fontsize=20)
        ax[1, 1].set_xlim([-80, 80])
        ax[1, 1].set_ylim([5.25, -5.25])
        ax[1, 1].set_xticks(np.arange(-80, 81, 40).tolist())
        ax[1, 1].set_yticks(np.arange(-5, 6, 2).tolist())

        im = ax[0, 0].imshow(original_power_spec[i, :, :], extent=(-82, 82, 6, -6), 
            interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        cbar=plt.colorbar(im, ax=ax.ravel().tolist(), orientation='vertical', aspect=20)
        cbar.ax.tick_params(labelsize=20)

        if file_name=='3sources_1moving_5dbsnr':
            fig.savefig(os.path.join(figure_path_pdf, f'{i}_phi_{gt_thetas[i, 1]}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path_png, f'{i}_phi_{gt_thetas[i, 1]}.png'), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(figure_path_pdf, f'{i}_snrdb_{gt_snrs[i]}.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path_png, f'{i}_snrdb_{gt_snrs[i]}.png'), bbox_inches='tight')
        plt.close()

if __name__=='__main__':
    device='cpu'
    print(f'Using device: {device}')
    
    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    os.environ["FIG_DIR"] = f'./Images/'
    os.environ['DUAL_SAVE'] = ''
    latex_fig.latexify()

    # 1. Average signal relative error.
    # plot_signal_relative_error()

    # 2. Average accuracy plots.
    # plot_accuracy()

    # 3. Plot Spectrums.
    config_file = './config.yml'
    dataset_cfg, model_cfg, training_cfg, metrics_cfg = utils.get_configurations(
        config_file=config_file)
    actfn_by_name = { 'relu': nn.ReLU, 'tanh': nn.Tanh, 'swish': nn.SiLU, 
        'sigmoid': nn.Sigmoid, 'identity': nn.Identity, 'leaky_relu': nn.LeakyReLU }
    
    base_model=utils.get_model(model_cfg=model_cfg, actfn_by_name=actfn_by_name)
    total_params=sum(p.numel() for p in base_model.parameters())
    print(f'Total parameters: {total_params}')

    base_model=testdata_loss_acc.get_best_model(training_cfg=training_cfg, 
        model_cfg=model_cfg, net=base_model, device=device)
    base_model.eval()

    data_dir='../dataset_folder/test_data'
    file_name='2sources_close_0_20snr' 
    n_sources=2
    # get_spectrums_2source(data_dir=data_dir, file_name=file_name, n_sources=n_sources)

    file_name='2sources_1moving_10dbsnr'
    n_sources=2
    # get_spectrums_2source(data_dir=data_dir, file_name=file_name, n_sources=n_sources)

    file_name='3sources_2close_1far_0_20snr'
    n_sources=3
    get_spectrums_3source(data_dir=data_dir, file_name=file_name, n_sources=n_sources)
    
    file_name='3sources_1moving_5dbsnr'
    n_sources=3
    get_spectrums_3source(data_dir=data_dir, file_name=file_name, n_sources=n_sources)

    
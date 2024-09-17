# Pytorch imports.
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import torch.optim as optim

# Standard libraries.
import os, random, json, yaml, time, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

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

def get_configurations(config_file):
    with open(config_file, 'r') as stream:
        cfg = yaml.safe_load(stream)
    (dataset_cfg, model_cfg, training_cfg, metrics_cfg) = (cfg['dataset'], cfg['model'], 
				cfg['training'], cfg['metrics'])
    return dataset_cfg, model_cfg, training_cfg, metrics_cfg

def get_annotation_file(dataset_cfg, shuffle=True):
	seed, label_path=dataset_cfg['seed'], dataset_cfg['label_path']
	test_size, val_size=dataset_cfg['test_size'], dataset_cfg['val_size']
	labels=pd.read_csv(label_path)
	file_num=np.arange(0, len(labels), 1)
	test_fnum, train_fnum = train_test_split(file_num, test_size=(1-test_size-val_size), random_state=seed, shuffle=shuffle)
	test_fnum, val_fnum = train_test_split(test_fnum, test_size=0.5, random_state=seed, shuffle=shuffle)
	train_df=pd.DataFrame(train_fnum, columns=['idx'])
	val_df=pd.DataFrame(val_fnum, columns=['idx'])
	test_df=pd.DataFrame(test_fnum, columns=['idx'])
	return train_df, val_df, test_df

class CustomDataset(data.Dataset):
	def __init__(self, annotation_file, dataset_cfg, is_traindataset=True):
		super().__init__()
		self.is_traindataset=is_traindataset
		self.dataset_cfg=dataset_cfg
		self.xy_idx = annotation_file
		self.x_img=np.load(dataset_cfg['dataset_path'])
		self.sig_amp=loadmat(dataset_cfg['sigdata_path'])['sig_amp']
		self.labels=pd.read_csv(dataset_cfg['label_path'])
		self.last_dim=dataset_cfg['last_dim']

	def __len__(self):
		return len(self.xy_idx)
	
	def __getitem__(self, index):
		idx=self.xy_idx.iloc[index]['idx']
		img=torch.as_tensor(self.convert_img(x=self.x_img[idx])).float()
		theta_gt=self.labels.iloc[idx][['src1_theta', 'src2_theta']].to_numpy()
		alpha_gt=self.labels.iloc[idx][['src1_alpha', 'src2_alpha']].to_numpy()
		snr_db=self.labels.iloc[idx][['snr_db']].to_numpy()
		sig_nsrc_gt=self.sig_amp[idx]

		if self.is_traindataset:
			sig_energy=np.sum(abs(sig_nsrc_gt)**2, axis=1)
			src_idx=np.argmax(sig_energy)
			sig_gt=torch.as_tensor(self.convert_sig(x=sig_nsrc_gt[src_idx])).float()
			doa_param=torch.as_tensor(np.array([theta_gt[src_idx], alpha_gt[src_idx]])).float()
		else:
			sig_gt=torch.as_tensor(self.convert_sig(x=sig_nsrc_gt)).float()
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

def get_dataloader(csv_file, dataset_cfg, dataset_name:str, is_traindataset: bool):
	dataset=CustomDataset(annotation_file=csv_file, dataset_cfg=dataset_cfg, is_traindataset=is_traindataset)
	print(f'{dataset_name} size: {len(dataset)}')
	batch_size, num_workers = dataset_cfg['batch_size'], dataset_cfg['num_workers']
	dataloader=data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
	print(f'Number of batches for {dataset_name} per epoch: {len(dataloader)}')
	return dataloader

def get_model(model_cfg, actfn_by_name):
	act_fn1=actfn_by_name[model_cfg['act_fn1']] if model_cfg['act_fn1'] in actfn_by_name.keys() else None
	act_fn2=actfn_by_name[model_cfg['act_fn2']] if model_cfg['act_fn2'] in actfn_by_name.keys() else None
	act_fn3=actfn_by_name[model_cfg['act_fn3']] if model_cfg['act_fn3'] in actfn_by_name.keys() else None

	c_in, c_outsum, c_out, c_red=model_cfg['c_in'], model_cfg['c_outsum'], model_cfg['c_out'], model_cfg['c_red']
	k_size=model_cfg['k_size']
	conv_stride, conv_padding, conv_dilation = model_cfg['conv_stride'], model_cfg['conv_padding'], model_cfg['conv_dilation']
	pool_stride, pool_padding, pool_dilation = model_cfg['pool_stride'], model_cfg['pool_padding'], model_cfg['pool_dilation']

	use_bias, use_bn=model_cfg['use_bias'], model_cfg['use_bn']
	is_1dkernel, is_2dkernel = model_cfg['is_1dkernel'], model_cfg['is_2dkernel']
	n_layers, n_inception = model_cfg['n_layers'], model_cfg['n_inception']
	resnet_stride1, resnet_stride2 = model_cfg['resnet_stride1'], model_cfg['resnet_stride2']
	resnet_subsample=model_cfg['resnet_subsample']
	is_concat, is_skip = model_cfg['is_concat'], model_cfg['is_skip']
	is_ampskip, is_doaskip = model_cfg['is_ampskip'], model_cfg['is_doaskip']
	is_se_net, squeeze_ratio= model_cfg['is_se_net'], model_cfg['squeeze_ratio']
	rnn_hidden_size, rnn_nlayers= model_cfg['rnn_hidden_size'], model_cfg['rnn_nlayers']
	n_snap, ndoa_param=model_cfg['n_snap'], model_cfg['ndoa_param']

	net=model.GridlessModel(
		act_fn1=act_fn1, act_fn2=act_fn2, act_fn3=act_fn3, c_in=c_in, c_outsum=c_outsum,
		c_out=c_out, c_red=c_red, k_size=k_size, conv_stride=conv_stride, conv_padding=conv_padding,
		conv_dilation=conv_dilation, pool_stride=pool_stride, pool_padding=pool_padding, 
		pool_dilation=pool_dilation, use_bias=use_bias, use_bn=use_bn, is_1dkernel=is_1dkernel, 
		is_2dkernel=is_2dkernel, n_layers=n_layers, n_inception=n_inception, 
		resnet_stride1=resnet_stride1, resnet_stride2=resnet_stride2, resnet_subsample=resnet_subsample, 
		is_concat=is_concat, is_skip=is_skip, is_ampskip=is_ampskip, is_doaskip=is_doaskip, 
		squeeze_ratio=squeeze_ratio, is_se_block=is_se_net, rnn_hid_size=rnn_hidden_size, 
		rnn_nlayers=rnn_nlayers, n_snap=n_snap
		)
	return net
	

# Functions for saving and loading the model.
# The hyperparameters are stored in a configuration file.

def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details.
	return os.path.join(model_path, model_name+'.config')

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
    
def optimizer_fn(optimizer_type, net_params, lr, weight_decay):
	if optimizer_type.lower()=='sgd':
		optimizer=optim.SGD(params=net_params, lr=lr, weight_decay=weight_decay)
	elif optimizer_type.lower()=='adam':
		optimizer=optim.Adam(params=net_params, lr=lr, weight_decay=weight_decay)
	elif optimizer_type.lower()=='adamw':
		optimizer=optim.AdamW(params=net_params, lr=lr, weight_decay=weight_decay)
	else:
		raise NotImplementedError(f'Enter correct optimizer_type: {optimizer_type}')
	return optimizer

def trajectory_loss_acc(preds_params, preds_traj, gts_params, 
	device, max_val=[80, 5], n_snap=30, rmse_threshold=2.4, max_angle=90, loss_type='loss4'):
	preds_phi, preds_alpha=max_val[0]*preds_params[:, 0].reshape(-1, 1), max_val[1]*preds_params[:, 1].reshape(-1, 1)
	gts_phi, gts_alpha=gts_params[:, 0].reshape(-1, 1), gts_params[:, 1].reshape(-1, 1)
	preds_track=(preds_phi + preds_alpha*torch.arange(n_snap, device=device) / (n_snap-1)) 
	gts_track=(gts_phi+gts_alpha*torch.arange(n_snap, device=device) / (n_snap-1)) 
	traj_e1 = f.mse_loss(input=preds_track/max_angle, target=gts_track/max_angle)
	traj_rmse = torch.sqrt(((preds_track-gts_track)**2).mean(-1))
	
	track_acc = ((traj_rmse < rmse_threshold).sum() / len(preds_params))*100
	traj_e2 = f.mse_loss(input=preds_traj, target=gts_track/max_angle)
	trajdiff_loss=f.mse_loss((preds_traj[:, 1:]-preds_traj[:, :-1]), (gts_track[:, 1:]-gts_track[:, :-1])/max_angle)
	track_dict={'gts_track': gts_track, 'preds_track': preds_track}
	traj_dict={'traj_e1': traj_e1, 'traj_e2': traj_e2, 
		'trajdiff_loss': trajdiff_loss, 'track_acc': track_acc, 'traj_rmse': traj_rmse}
	return track_dict, traj_dict

def scale_invariant_sdr(preds, labels, zero_mean=False):
	eps=torch.finfo(preds.dtype).eps
	if zero_mean:
		preds=preds-torch.mean(preds, dim=1, keepdim=True)
		labels=labels-torch.mean(labels, dim=1, keepdim=True)
	scale_factor=(torch.sum((preds*labels), dim=1, keepdim=True)+eps) / (torch.sum(labels**2, dim=1, keepdim=True)+eps)
	scaled_labels=scale_factor*labels
	error=scaled_labels-preds
	val=(torch.sum(scaled_labels**2, dim=1)+eps) / (torch.sum(error**2, dim=1)+eps)
	val_db=(10*torch.log10(val)).mean()
	return val_db

def scale_invariant_snr(preds, labels):
	eps=torch.finfo(preds.dtype).eps
	error=preds-labels
	val=(torch.sum(labels**2, dim=1)+eps) / (torch.sum(error**2, dim=1)+eps)
	val_db=(10*torch.log10(val)).mean()
	return val_db

def get_relative_error(preds, gts, dim):
    num=torch.norm((preds[..., 0]+1j*preds[..., 1])-(gts[..., 0]+1j*gts[..., 1]), dim=dim)
    deno=torch.norm((gts[..., 0]+1j*gts[..., 1]), dim=dim)
    rel_error=(num/deno).mean()*100
    return rel_error

def get_recvsig(sigs, doa_traj, device, n_snap=30, n_sensors=8):
    sigs=sigs[..., None] * torch.eye(n_snap, device=device)
    n_phi=torch.arange(n_sensors, device=device).reshape(1, -1, 1) * torch.sin(doa_traj*torch.pi/180.)[:, None, :]
    steer_mat=torch.exp(-1j*torch.pi*n_phi)
    recvsig=steer_mat @ sigs
    recvsig = torch.stack((recvsig.real, recvsig.imag), dim=-1)
    return recvsig

def signal_error(preds_amps, gts_amps, preds_track, gts_track, device, n_snap=30, n_sensors=8):
	sig_mse=f.mse_loss(input=preds_amps, target=gts_amps)
	preds_sigs=preds_amps[..., 0] + 1j*preds_amps[..., 1]
	gts_sigs=gts_amps[..., 0] + 1j*gts_amps[..., 1]
	preds_recvsig=get_recvsig(sigs=preds_sigs, doa_traj=preds_track, device=device, n_snap=n_snap,
		n_sensors=n_sensors)
	gts_recvsig=get_recvsig(sigs=gts_sigs, doa_traj=gts_track, device=device, n_snap=n_snap,
		n_sensors=n_sensors)
	recvsig_mse=f.mse_loss(input=preds_recvsig, target=gts_recvsig)
	recvsig_re= get_relative_error(preds=preds_recvsig, gts=gts_recvsig, dim=(1, 2))
	sig_re=get_relative_error(preds=preds_amps, gts=gts_amps, dim=1)
	preds_phase=torch.atan2(preds_amps[..., 1], preds_amps[..., 0])
	gts_phase=torch.atan2(gts_amps[..., 1], gts_amps[..., 0])
	phase_cossim=f.cosine_similarity(x1=preds_phase, x2=gts_phase, dim=1).mean()
	mag_e=f.mse_loss(torch.abs(preds_sigs), torch.abs(gts_sigs))
	si_sdr=scale_invariant_sdr(preds=preds_amps, labels=gts_amps)
	si_snr=scale_invariant_snr(preds=preds_amps, labels=gts_amps)
	recvsig_dict = {'recvsig_mse': recvsig_mse, 'recvsig_re': recvsig_re, 
        'preds_recvsig': preds_recvsig, 'gts_recvsig': gts_recvsig}
	sig_dict = {'sig_mse': sig_mse, 'sig_re': sig_re, 'phase_cossim': phase_cossim, 
        'mag_e': mag_e, 'si_sdr': si_sdr, 'si_snr': si_snr}
	return recvsig_dict, sig_dict

def compute_loss_and_acc(preds_sigs, preds_params, preds_traj, gts_sigs, gts_params, 
	device, loss_type='loss4', n_snap=30, max_val=[80, 5], loss_weights=[1, 1, 1], 
	rmse_threshold=2.4, max_angle=90):
	track_dict, traj_dict = trajectory_loss_acc(preds_params=preds_params, 
		preds_traj=preds_traj, gts_params=gts_params, device=device, max_val=max_val, 
		n_snap=n_snap, rmse_threshold=rmse_threshold, max_angle=max_angle, 
		loss_type=loss_type)
	recvsig_dict, sig_dict = signal_error(preds_amps=preds_sigs, gts_amps=gts_sigs, 
        preds_track=track_dict['preds_track'], gts_track=track_dict['gts_track'], 
        device=device, n_snap=n_snap)
	t_loss=get_total_loss(traj_dict=traj_dict,
        recvsig_dict=recvsig_dict, sig_dict=sig_dict, loss_type=loss_type, 
		loss_weights=loss_weights)
	return t_loss, track_dict, traj_dict, recvsig_dict, sig_dict
	
def get_total_loss(traj_dict, recvsig_dict, sig_dict, loss_type, loss_weights):
    if loss_type=='loss2':
        t_loss=sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']
    elif loss_type=='loss3':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*traj_dict['traj_e2'])
    elif loss_type=='loss3_1':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*traj_dict['trajdiff_loss'])
    elif loss_type=='loss3_2':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*recvsig_dict['recvsig_mse'])
    elif loss_type=='loss4':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*traj_dict['traj_e2']+loss_weights[2]*recvsig_dict['recvsig_mse'])
    elif loss_type=='loss4_1':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*traj_dict['trajdiff_loss']+loss_weights[2]*recvsig_dict['recvsig_mse'])
    elif loss_type=='loss5':
        t_loss=(sig_dict['sig_mse']+loss_weights[0]*traj_dict['traj_e1']+
			loss_weights[1]*traj_dict['traj_e2']+loss_weights[1]*traj_dict['trajdif']+
			loss_weights[2]*recvsig_dict['recvsig_mse'])
    else:
        raise NotImplementedError(f'Choose correct loss_type: {loss_type}')
    return t_loss	

# def overall_loss_acc(preds1, preds2, labels1, labels2, device, loss_type='l2', n_snap=30, max_val=[80, 5], 
# 	loss_weights=[1, 0.005, 1], rmse_threshold=1.2):
# 	# breakpoint()
# 	high_energy_src_idx=torch.argmax((labels1**2).sum(dim=(2, 3)), dim=-1)
# 	high_energy_src=labels1[range(len(high_energy_src_idx)), high_energy_src_idx]
# 	high_energy_doa_param=labels2[range(len(high_energy_src_idx)), :, high_energy_src_idx]
# 	sig_reconst_loss=loss_fn(preds=preds1, labels=high_energy_src)
# 	traj_loss, track_acc, gt_track, preds_track=trajectory_loss_fn(
# 		preds=preds2, labels=high_energy_doa_param, device=device, max_val=max_val, n_snap=n_snap, 
# 		rmse_threshold=rmse_threshold)
# 	total_loss=loss_weights[0]*sig_reconst_loss + loss_weights[1]*traj_loss
# 	# breakpoint()
# 	return total_loss, sig_reconst_loss, traj_loss, track_acc

# def get_other_signal_errors(preds, labels):
# 	pred_sigs=preds[:, :, 0]+1j*preds[:, :, 1]
# 	high_energy_src_idx=torch.argmax((labels**2).sum(dim=(2, 3)), dim=-1)
# 	high_energy_src=labels[range(len(high_energy_src_idx)), high_energy_src_idx]
# 	gt_sigs=high_energy_src[:, :, 0]+1j*high_energy_src[:, :, 1]
# 	sig_rel_error=((torch.norm((pred_sigs-gt_sigs), dim=1) / torch.norm(gt_sigs, dim=-1)).mean()*100)
# 	preds_phase=torch.atan2(preds[..., 1], preds[..., 0])
# 	gt_phase=torch.atan2(high_energy_src[..., 1], high_energy_src[..., 0])
# 	sig_phase_cosim=f.cosine_similarity(x1=preds_phase, x2=gt_phase, dim=1).mean()
# 	sig_mag_error=f.mse_loss(torch.abs(pred_sigs), torch.abs(gt_sigs))
# 	sig_sisdr=scale_invariant_sdr(preds=preds, labels=high_energy_src, zero_mean=False)
# 	sig_sisnr=scale_invariant_sdr(preds=preds, labels=high_energy_src, zero_mean=True)
# 	return sig_rel_error, sig_phase_cosim, sig_mag_error, sig_sisdr, sig_sisnr

def train_model(
	net:nn.Module, model_path:str, model_name:str, 
	train_loader: data.DataLoader, val_loader: data.DataLoader, test_loader: data.DataLoader, 
	device, max_epochs:int=50, lr=1e-4, loss_type='loss4', optimizer_type:str ='adamw', 
    weight_decay=0, rmse_threshold=2.4, loss_weights=[1, 1, 1], n_snap=30, max_val=[80, 5], 
	max_angle=90, is_norm=False):

	file_exists=os.path.isfile(_get_model_file(model_path=model_path, model_name=model_name))
	print(f'Is file exists?: {file_exists}')

	if file_exists:
		print(f'Model file already exists. Skip training....')
	else:
		print('Training....')
		print(f'loss_type: {loss_type}')
		print(f'max epochs: {max_epochs}')
		print(f'optimizer_type: {optimizer_type}')
		print(f'learning_rate: {lr}')
		print(f'rmse_threshold: {rmse_threshold}')

		# Define optimizer.
		optimizer=optimizer_fn(optimizer_type=optimizer_type, net_params=net.parameters(), 
			lr=lr, weight_decay=weight_decay)
		lr_scheduler=optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[30, 40, 43, 46, 49], gamma=0.5, verbose=True)

		train_hist = pd.DataFrame(
            columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
                'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
                'trajdiff_mse', 'track_acc'])
		val_hist1=pd.DataFrame(
            columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
                'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
                'trajdiff_mse', 'track_acc'])
		val_hist2=pd.DataFrame(
            columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
                'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
                'trajdiff_mse', 'track_acc'])
		avg_val_hist=pd.DataFrame(
            columns=['epoch', 'tloss', 'sig_mse', 'sig_re', 'phase_cossim', 'mag_e', 
                'si_sdr', 'si_snr', 'recvsig_mse', 'recvsig_re', 'traj_e1', 'traj_e2', 
                'trajdiff_mse', 'track_acc'])

		best_epochavgloss, best_epochavgacc =-1, -1
		
		net=net.to(device)
		print(f'Check if model is in cuda: {next(net.parameters()).is_cuda}')

		start_time=time.time()
		for epoch in range(max_epochs):
			# Training.
			epoch_st=time.time()
			net.train()

			epoch_log_dict = {'tloss': 0., 'sig_mse': 0., 'sig_re': 0., 'sig_mage': 0., 
				'phase_cossim': 0., 'si_sdr': 0., 'si_snr': 0., 
				'recvsig_mse': 0., 'recvsig_re': 0., 
				'traj_e1': 0., 'traj_e2': 0., 'trajdiff_mse': 0., 'track_acc': 0.}

			t=tqdm(iterable=train_loader, leave=False)
			# if ((epoch+1)>=30) and ((epoch+1)%5==0):
			# 	lr=lr*0.5
			# 	optimizer=optimizer_fn(optimizer_type=optimizer_type, net_params=net.parameters(), 
			# 		lr=lr, weight_decay=weight_decay)
			# 	print(f'\nLearning rate: {lr} changed after epoch: {epoch+1}')
			
			# for imgs, gt_sigs, gt_params, _ in train_loader:
			for imgs, gts_sigs, gts_params, snr_db in t:
				(imgs, gts_sigs, gts_params)=imgs.to(device), gts_sigs.to(device), gts_params.to(device)
				optimizer.zero_grad()
				preds_sigs, preds_doa_params, preds_doatracks=net(imgs)
				t_loss, track_dict, traj_dict, recvsig_dict, sig_dict=compute_loss_and_acc(
					preds_sigs=preds_sigs, preds_params=preds_doa_params, preds_traj=preds_doatracks, 
					gts_sigs=gts_sigs, gts_params=gts_params, device=device, loss_type=loss_type, 
					n_snap=n_snap, max_val=max_val, loss_weights=loss_weights, 
					rmse_threshold=rmse_threshold, max_angle=max_angle)
				t_loss.backward()
				optimizer.step()
				# t.set_description(f'Epoch: {epoch+1} -tloss: {total_loss.item():4.5f}, ')
				epoch_log_dict=update_epoch_log(log_dict=epoch_log_dict, t_loss=t_loss, 
					sig_dict=sig_dict, traj_dict=traj_dict, recvsig_dict=recvsig_dict)
			train_hist=update_hist(hist=train_hist, log_dict=epoch_log_dict, epoch=epoch, l=len(train_loader))

			epoch_et=time.time()
			epoch_tt=epoch_et-epoch_st
			do_print(epoch=epoch, hist=train_hist, epoch_tt=epoch_tt, stage='Training')
			
			# Validation step.
			val_hist1, val_hist2, avg_val_hist=validate_model(net=net, dataloader=val_loader, 
				device=device, epoch=epoch, loss_type=loss_type, n_snap=n_snap, 
				max_val=max_val, max_angle=max_angle, loss_weights=loss_weights, 
				rmse_threshold=rmse_threshold, hist1=val_hist1, hist2=val_hist2, 
				avg_hist=avg_val_hist)
			
			if ((len(avg_val_hist) == 1) or 
	   				(avg_val_hist['tloss'].iloc[epoch] < avg_val_hist['tloss'].iloc[best_epochavgloss])):
				print('\t New best performance using avgerage loss, saving model...')
				best_model_path_avgloss=os.path.join(model_path, f'best_model_avgloss')
				best_model_name_avgloss=model_name +f'_bm_avgloss'
				save_model(model=net, model_path=best_model_path_avgloss, model_name=best_model_name_avgloss)
				best_epochavgloss=epoch
			if ((len(avg_val_hist) == 1) or 
	   				(avg_val_hist['track_acc'].iloc[epoch] > avg_val_hist['track_acc'].iloc[best_epochavgacc])):
				print('\t New best performance using avgerage accuracy, saving model...')
				best_model_path_avgacc=os.path.join(model_path, f'best_model_avgacc')
				best_model_name_avgacc=model_name +f'_bm_avgacc'
				save_model(model=net, model_path=best_model_path_avgacc, model_name=best_model_name_avgacc)
				best_epochavgacc=epoch
			
			print(f'\tSaving model for each epoch: {epoch}.')
			diff_model_path=os.path.join(model_path, f'diff_model')
			diff_model_name=model_name+f'_epoch={epoch+1}'
			save_model(model=net, model_path=diff_model_path, model_name=diff_model_name)
				# best_val_epoch=epoch
			lr_scheduler.step()
		end_time=time.time()
	print(f'Total training time: {(end_time-start_time)/60} mins')
	print(f'Best epochs based on avg-loss: {best_epochavgloss}, avg-acc: {best_epochavgacc}')
	return train_hist, val_hist1, val_hist2, avg_val_hist

def val_compute_loss_acc(preds_sigs, preds_params, preds_traj, gts_sigs, gts_params, device, 
	loss_type='loss4', n_snap=30, max_val=[80, 5], loss_weights=[1, 1, 1], 
	rmse_threshold=2.4, max_angle=90, src_idx=None):
	if src_idx == None:
		src_idx=torch.argmax((gts_sigs**2).sum(dim=(2, 3)), dim=-1)
		all_idxes=torch.arange(2, device=device)
		mask=(all_idxes.unsqueeze(0) == src_idx.unsqueeze(1))
	else:
		all_idxes=torch.arange(2, device=device)
		mask=(all_idxes.unsqueeze(0) != src_idx.unsqueeze(1))
	t_loss, track_dict, traj_dict, recvsig_dict, sig_dict=compute_loss_and_acc(
		preds_sigs=preds_sigs, preds_params=preds_params, preds_traj=preds_traj, 
		gts_sigs=gts_sigs[mask], gts_params=gts_params.permute(0,2,1)[mask], 
		device=device, loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
		loss_weights=loss_weights, rmse_threshold=rmse_threshold, max_angle=max_angle)
	return t_loss, track_dict, traj_dict, recvsig_dict, sig_dict, src_idx

def validate_model(net:nn.Module, dataloader: data.DataLoader, device, epoch, loss_type='loss4', 
	n_snap=30, max_val=[80, 5], max_angle=90, loss_weights=[1, 1, 1], rmse_threshold=2.4, 
	hist1=None, hist2=None, avg_hist=None):
	net.eval()
	val_log_dict={'tloss': {0:0, 1:0}, 'sig_mse': {0:0, 1:0}, 'sig_re': {0:0, 1:0}, 
        'sig_mage': {0:0, 1:0}, 'phase_cossim': {0:0, 1:0}, 'si_sdr': {0:0, 1:0}, 
		'si_snr': {0:0, 1:0}, 'recvsig_mse': {0:0, 1:0}, 'recvsig_re': {0:0, 1:0}, 
		'traj_e1': {0:0, 1:0}, 'traj_e2': {0:0, 1:0}, 'trajdiff_mse': {0:0, 1:0}, 
		'track_acc': {0:0, 1:0}}
	t=tqdm(iterable=dataloader, leave=False)
	start_time=time.time()
	# for imgs, gt_sigs, gt_params, _ in dataloader:
	for imgs, gt_sigs, gt_params, _ in t:
		(imgs, gts_sigs, gts_params)=imgs.to(device), gts_sigs.to(device), gts_params.to(device)
		with torch.no_grad():
			for i in range(2):
				if i==0:
					preds_sigs, preds_params, preds_doatracks=net(imgs)
					(t_loss, track_dict, traj_dict, recvsig_dict, sig_dict, 
                        src_idx)=val_compute_loss_acc(preds_sigs=preds_sigs, 
							preds_params=preds_params, preds_traj=preds_doatracks, 
							gts_sigs=gts_sigs, gts_params=gts_params, device=device, 
                            loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
							loss_weights=loss_weights, rmse_threshold=rmse_threshold, 
							max_angle=max_angle, src_idx=None)
					imgs=imgs-recvsig_dict['preds_recvsig'].unsqueeze(1)
					val_log_dict=update_epoch_log(log_dict=val_log_dict, t_loss=t_loss, 
						sig_dict=sig_dict, traj_dict=traj_dict, recvsig_dict=recvsig_dict, iter=i)
				else:
					preds_sigs, preds_params, preds_doatracks=net(imgs)
					(t_loss, track_dict, traj_dict, recvsig_dict, sig_dict, 
                        src_idx)=val_compute_loss_acc(preds_sigs=preds_sigs, 
							preds_params=preds_params, preds_traj=preds_doatracks, 
							gts_sigs=gts_sigs, gts_params=gts_params, device=device, 
                            loss_type=loss_type, n_snap=n_snap, max_val=max_val, 
							loss_weights=loss_weights, rmse_threshold=rmse_threshold, 
							max_angle=max_angle, src_idx=src_idx)
					val_log_dict=update_epoch_log(log_dict=val_log_dict, t_loss=t_loss, 
						sig_dict=sig_dict, traj_dict=traj_dict, recvsig_dict=recvsig_dict, iter=i)
	end_time=time.time()
	val_epoch_tt=end_time-start_time
	
	hist1=update_val_hist(hist=hist1, log_dict=val_log_dict, epoch=epoch, 
		l=len(dataloader), src_num=0)
	do_print(epoch=epoch, hist=hist1, epoch_tt=val_epoch_tt, stage='Validation 1st src')
	hist2=update_val_hist(hist=hist2, log_dict=val_log_dict, epoch=epoch, 
		l=len(dataloader), src_num=1)
	do_print(epoch=epoch, hist=hist2, epoch_tt=val_epoch_tt, stage='Validation 2nd src')
	avg_hist=update_avg_hist(avg_hist=avg_hist, hist1=hist1, hist2=hist2, epoch=epoch)
	do_print(epoch=epoch, hist=avg_hist, epoch_tt=val_epoch_tt, stage='Validation Avg')
	return hist1, hist2, avg_hist

def update_epoch_log(log_dict, t_loss, sig_dict, traj_dict, recvsig_dict, iter=None):
	if iter==None:
		log_dict['tloss'] += t_loss.item()
		log_dict['sig_mse'] += sig_dict['sig_mse'].item()
		log_dict['sig_re'] += sig_dict['sig_re'].item()
		log_dict['phase_cossim'] += sig_dict['phase_cossim'].item()
		log_dict['sig_mage'] += sig_dict['mag_e'].item()
		log_dict['si_sdr'] += sig_dict['si_sdr'].item()
		log_dict['si_snr'] += sig_dict['si_snr'].item()
		log_dict['recvsig_mse'] += recvsig_dict['recvsig_mse'].item()
		log_dict['recvsig_re'] += recvsig_dict['recvsig_re'].item()
		log_dict['traj_e1'] += traj_dict['traj_e1'].item()
		log_dict['traj_e2'] += traj_dict['traj_e2'].item()
		log_dict['trajdiff_mse'] += traj_dict['trajdiff_loss'].item()
		log_dict['track_acc'] += traj_dict['track_acc'].item()
	else:
		log_dict['tloss'][iter] += t_loss.item()
		log_dict['sig_mse'][iter] += sig_dict['sig_mse'].item()
		log_dict['sig_re'][iter] += sig_dict['sig_re'].item()
		log_dict['phase_cossim'][iter] += sig_dict['phase_cossim'].item()
		log_dict['sig_mage'][iter] += sig_dict['mag_e'].item()
		log_dict['si_sdr'][iter] += sig_dict['si_sdr'].item()
		log_dict['si_snr'][iter] += sig_dict['si_snr'].item()
		log_dict['recvsig_mse'][iter] += recvsig_dict['recvsig_mse'].item()
		log_dict['recvsig_re'][iter] += recvsig_dict['recvsig_re'].item()
		log_dict['traj_e1'][iter] += traj_dict['traj_e1'].item()
		log_dict['traj_e2'][iter] += traj_dict['traj_e2'].item()
		log_dict['trajdiff_mse'][iter] += traj_dict['trajdiff_loss'].item()
		log_dict['track_acc'][iter] += traj_dict['track_acc'].item()
	return log_dict

def update_hist(hist, log_dict, epoch, l):
    hist=hist.append({
        'epoch': epoch, 'tloss': log_dict['tloss']/l, 'sig_mse': log_dict['sig_mse']/l,
        'sig_re': log_dict['sig_re']/l, 'phase_cossim': log_dict['phase_cossim']/l,
        'mag_e': log_dict['sig_mage']/l, 'si_sdr': log_dict['si_sdr']/l,
        'si_snr': log_dict['si_snr']/l, 'recvsig_mse': log_dict['recvsig_mse']/l, 
        'recvsig_re': log_dict['recvsig_re']/l, 'traj_e1': log_dict['traj_e1']/l, 
        'traj_e2': log_dict['traj_e2']/l, 'trajdiff_mse': log_dict['trajdiff_mse']/l, 
        'track_acc': log_dict['track_acc']/l}, ignore_index=True)
    return hist

def update_val_hist(hist, log_dict, epoch, l, src_num=0):
    # breakpoint()
    hist=hist.append({
        'epoch': epoch, 'tloss': log_dict['tloss'][src_num]/l, 'sig_mse': log_dict['sig_mse'][src_num]/l,
        'sig_re': log_dict['sig_re'][src_num]/l, 'phase_cossim': log_dict['phase_cossim'][src_num]/l,
        'mag_e': log_dict['sig_mage'][src_num]/l, 'si_sdr': log_dict['si_sdr'][src_num]/l,
        'si_snr': log_dict['si_snr'][src_num]/l, 'recvsig_mse': log_dict['recvsig_mse'][src_num]/l, 
        'recvsig_re': log_dict['recvsig_re'][src_num]/l, 'traj_e1': log_dict['traj_e1'][src_num]/l, 
        'traj_e2': log_dict['traj_e2'][src_num]/l, 'trajdiff_mse': log_dict['trajdiff_mse'][src_num]/l, 
        'track_acc': log_dict['track_acc'][src_num]/l}, ignore_index=True)
    return hist

def update_avg_hist(avg_hist, hist1, hist2, epoch):
    avg_hist=avg_hist.append({
        'epoch': epoch, 'tloss': (hist1['tloss'].iloc[epoch]+hist2['tloss'].iloc[epoch])/2,
        'sig_mse': (hist1['sig_mse'].iloc[epoch]+hist2['sig_mse'].iloc[epoch])/2,
        'sig_re': (hist1['sig_re'].iloc[epoch]+hist2['sig_re'].iloc[epoch])/2,
        'phase_cossim': (hist1['phase_cossim'].iloc[epoch]+hist2['phase_cossim'].iloc[epoch])/2,
        'mag_e': (hist1['mag_e'].iloc[epoch]+hist2['mag_e'].iloc[epoch])/2,
        'si_sdr': (hist1['si_sdr'].iloc[epoch]+hist2['si_sdr'].iloc[epoch])/2,
        'si_snr': (hist1['si_snr'].iloc[epoch]+hist2['si_snr'].iloc[epoch])/2,
        'recvsig_mse': (hist1['recvsig_mse'].iloc[epoch]+hist2['recvsig_mse'].iloc[epoch])/2,
        'recvsig_re': (hist1['recvsig_re'].iloc[epoch]+hist2['recvsig_re'].iloc[epoch])/2,
        'traj_e1': (hist1['traj_e1'].iloc[epoch]+hist2['traj_e1'].iloc[epoch])/2,
        'traj_e2': (hist1['traj_e2'].iloc[epoch]+hist2['traj_e2'].iloc[epoch])/2,
        'trajdiff_mse': (hist1['trajdiff_mse'].iloc[epoch]+hist2['trajdiff_mse'].iloc[epoch])/2,
        'track_acc': (hist1['track_acc'].iloc[epoch]+hist2['track_acc'].iloc[epoch])/2
        }, ignore_index=True)
    return avg_hist

def do_print(epoch, hist, epoch_tt, stage:str):
    print(f"Epoch: {epoch+1:2d}, {stage}, time: {epoch_tt/60:.2f}mins, "
		f"tloss: {hist['tloss'].iloc[epoch]:4.5f}, sig_mse: {hist['sig_mse'].iloc[epoch]:4.5f}, "
		f"sig_re: {hist['sig_re'].iloc[epoch]:4.5f}, phase_cossim: {hist['phase_cossim'].iloc[epoch]:4.5f}, "
		f"mag_e: {hist['mag_e'].iloc[epoch]:4.5f}, si_sdr: {hist['si_sdr'].iloc[epoch]:4.5f}, si_snr: {hist['si_snr'].iloc[epoch]:4.5f},"
		f"recvsig_mse: {hist['recvsig_mse'].iloc[epoch]:4.5f}, recvsig_re: {hist['recvsig_re'].iloc[epoch]:4.5f}, "
		f"traj_e1: {hist['traj_e1'].iloc[epoch]:4.5f}, traj_e2: {hist['traj_e2'].iloc[epoch]:4.5f}, "
		f"trajdiff_mse: {hist['trajdiff_mse'].iloc[epoch]:4.5f}, track_acc: {hist['track_acc'].iloc[epoch]:4.5f}")
    return 

if __name__=='__main__':
	config_file='./config1.yml'
	dataset_cfg, model_cfg, training_cfg, metrics_cfg = get_configurations(config_file=config_file)
	seed=dataset_cfg['seed']
	last_dim=dataset_cfg['last_dim']
	set_seed(seed)
	train_df, val_df, test_df = get_annotation_file(dataset_cfg=dataset_cfg)
	train_dataset = CustomDataset(annotation_file=train_df, dataset_cfg=dataset_cfg, is_traindataset=True)
	val_dataset=CustomDataset(annotation_file=val_df, dataset_cfg=dataset_cfg, is_traindataset=False)
	test_dataset=CustomDataset(annotation_file=test_df, dataset_cfg=dataset_cfg, is_traindataset=False)
	

	for idx, (img, sig_gt, doa_param, _) in enumerate(train_dataset):
		print(f'idx: {idx}')
		print(f'img.shape: {img.shape}')
		# print(f'tmap.shape: {img_gt.shape}')
		print(f'sig_gt.shape: {sig_gt.shape}')
		print(f'Doa parameter: {doa_param}')
		break
	
	for idx, (img, sig_gt, doa_param, _) in enumerate(val_dataset):
		print(f'idx: {idx}')
		print(f'img.shape: {img.shape}')
		# print(f'tmap.shape: {img_gt.shape}')
		print(f'sig_gt.shape: {sig_gt.shape}')
		print(f'Doa parameter: {doa_param}')
		break
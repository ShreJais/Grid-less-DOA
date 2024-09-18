import numpy as np
import pandas as pd
import os, time

import warnings
warnings.filterwarnings('ignore')

def theta_options(phi1, i, theta_grid, min_diff=5.0):
    if (phi1 < (min(theta_grid)+min_diff)) and (phi1 < 0):
        phi2_end=np.round(phi1 + min_diff, decimals=2)
        alt_theta_grid=theta_grid[theta_grid > phi2_end]
        print(i, phi1, 0, phi2_end, len(alt_theta_grid))
    elif (phi1 > (max(theta_grid)-min_diff)) and (phi1>0):
        phi2_start=np.round(phi1-min_diff, decimals=2)
        alt_theta_grid=theta_grid[theta_grid<phi2_start]
        print(i, phi1, phi2_start, 0, len(alt_theta_grid))
    else:
        phi2_start=np.round(phi1-min_diff, decimals=2)
        phi2_end=np.round(phi1 + min_diff, decimals=2)
        if (phi1>=0):
            left_grid=theta_grid[theta_grid < phi2_start]
            right_grid=theta_grid[theta_grid > phi2_end]
            alt_theta_grid=np.append(left_grid, right_grid)
            print(i, phi1, phi2_start, phi2_end, len(alt_theta_grid))
        else:
            left_grid=theta_grid[theta_grid < phi2_start]
            right_grid=theta_grid[theta_grid > phi2_end]
            alt_theta_grid=np.append(left_grid, right_grid)
            print(i, phi1, phi2_start, phi2_end, len(alt_theta_grid))
    return alt_theta_grid

def get_source_doaparams(file_name, theta_grid, alpha_grid, n_sources=2, n_sensors=8, n_snr=8):
    col_names=[]
    for i in range(n_sources):
        col_names.extend([f'src{i+1}_theta', f'src{i+1}_alpha'])
    col_names.extend(['snr_db'])
    df=pd.DataFrame(columns=col_names)
    count=[]
    for i in range(len(theta_grid)):
        phi_src1=theta_grid[i]
        phi_src2_grid=theta_options(phi1=phi_src1, i=i, theta_grid=theta_grid)
        count.append(len(phi_src2_grid))
        new_df=pd.DataFrame(columns=col_names)
        for j in range(len(phi_src2_grid)):
            phi_src2=phi_src2_grid[j]
            for _ in range(n_snr):
                alpha_src1=np.random.choice(alpha_grid)
                alpha_src2=np.random.choice(np.setdiff1d(alpha_grid, alpha_src1))
                snr_db = np.round(np.random.uniform(0, 20), 2)
                doa_params={
                    'src1_theta': phi_src1, 'src2_theta': phi_src2, 
                    'src1_alpha': alpha_src1, 'src2_alpha': alpha_src2, 'snr_db': snr_db
                    }
                curr_df=pd.DataFrame([doa_params])
                new_df=pd.concat([new_df, curr_df], ignore_index=True)
        df=pd.concat([df, new_df], ignore_index=True)
        print(sum(count)*n_snr)
    df.to_csv(f'{file_name}.csv')

def get_random_source_doaparams(file_name, n_examples=100, n_sources=2, 
    phi_range=[-80, 80], alpha_range=[-5, 5], snr_range=[0, 20], min_diff=5):
    print(f'n_examples: {n_examples}')
    col_names=[]
    for i in range(n_sources):
        col_names.extend([f'src{i+1}_theta', f'src{i+1}_alpha'])
    col_names.extend(['snr_db'])

    # df=pd.DataFrame(columns=col_names)
    all_data=[]
    for i in range(n_examples):
        np.random.seed(i)
        src1_phi=np.round(np.random.uniform(low=phi_range[0], high=phi_range[1]), decimals=2)
        src2_phi=get_phi2_given_phi1(phi1=src1_phi, seed=i, phi_range=phi_range, min_diff=min_diff)
        src1_alpha=np.round(np.random.uniform(low=alpha_range[0], high=alpha_range[1]), decimals=2)
        src2_alpha=np.round(np.random.uniform(low=alpha_range[0], high=alpha_range[1]), decimals=2)
        snr_db = np.round(np.random.uniform(low=snr_range[0], high=snr_range[1]), decimals=2)

        doa_params={
            'src1_theta': src1_phi, 'src2_theta': src2_phi, 'src1_alpha': src1_alpha,
            'src2_alpha': src2_alpha, 'snr_db': snr_db
        }
        all_data.append(doa_params)
    
    df=pd.DataFrame(all_data, columns=col_names)
    df.to_csv(f'{file_name}.csv', index=False)

def get_phi2_given_phi1(phi1, seed, phi_range=[-80, 80], min_diff=5):
    np.random.seed(seed)
    if (phi1 < (min(phi_range))+min_diff) and (phi1<0):
        return np.round(np.random.uniform(low=phi1+min_diff, high=phi_range[1]), decimals=2)
    elif (phi1 > (max(phi_range)-min_diff)) and (phi1>0):
        return np.round(np.random.uniform(low=phi_range[0], high=phi1-min_diff), decimals=2)
    else:
        left_end=phi1 - min_diff
        right_end=phi1 + min_diff
        if np.random.rand() < 0.5:
            return np.round(np.random.uniform(low=phi_range[0], high=left_end), decimals=2)
        else:
            return np.round(np.random.uniform(low=right_end, high=phi_range[1]), decimals=2)

def generate_signal(n_sources, n_sensors, snr_db, wavelen, mic_dist, doa_params, seed, n_snap):
    assert len(doa_params)==n_sources, ('Number of sources doesn"t match number of doa parameters.')
    
    # set seed for reproducibility.
    np.random.seed(seed=seed)

    # Generate gaussian distributed source signal.
    src1_var = 1
    src2_var = np.round(np.random.uniform(low=0.5, high=1), decimals=3)
    sig_pow = (src1_var**2 + src2_var**2)/n_sources
    
    src1_real, src1_imag = (src1_var*np.random.randn(n_sources-1, n_snap), 
        src1_var*np.random.randn(n_sources-1, n_snap))
    src1_sig=src1_real +1j*src1_imag
    src2_real, src2_imag = (src2_var*np.random.randn(n_sources-1, n_snap), 
        src2_var*np.random.randn(n_sources-1, n_snap))
    src2_sig=src2_real+1j*src2_imag

    sig=np.concatenate((src1_sig, src2_sig), axis=0)
    # different way to represent signal.
    new_sig=np.zeros((n_sources*n_snap, n_snap), dtype=np.complex128) # shape: (n_sources*n_snap , n_snap)
    for i in range(n_sources):
        new_sig[n_snap*i:n_snap*(i+1),:]=sig[i, :] * np.eye(n_snap)
    
    # generate gaussian noise using given snr_db and computed noise_std.
    noise_std=np.round(np.sqrt(10**(-snr_db/10)*sig_pow), decimals=3)
    noise=noise_std*(np.random.randn(n_sensors, n_snap) + 1j*np.random.randn(n_sensors, n_snap))
    
    # Generate sensing matrix A, which contains steering vectors.
    const=-1j*2*np.pi* (mic_dist/wavelen)
    A=np.zeros((n_sensors, n_sources*n_snap), dtype=np.complex128)
    sep_recv_sig=np.zeros((n_sources, n_sensors, n_snap), dtype=np.complex128)
    
    for i in range(n_sources):
        theta=doa_params[f'source{i+1}'][f'src{i+1}_theta']
        alpha=doa_params[f'source{i+1}'][f'src{i+1}_alpha']
        delta_alpha= alpha/(n_snap-1)
        doa = theta+delta_alpha*np.arange(n_snap)
        n_phi=np.arange(n_sensors).reshape(-1, 1) * np.sin(doa*np.pi/180.) # shape: (n_sensors, n_snap)
        A[:, i*n_snap:(i+1)*n_snap]=np.exp(const*n_phi)
        sep_recv_sig[i, :, :]=np.matmul(A[:, i*n_snap:(i+1)*n_snap], new_sig[i*n_snap:(i+1)*n_snap, :])

    recv_sig=np.matmul(A, new_sig)
    recvsig_noise=recv_sig+noise
    return recvsig_noise, recv_sig, noise_std, sig

def power_cbf(n_sensors:int, sig_recv, grid:dict, wavelen, mic_dist, n_snap=30):
    theta_grid, alpha_grid=grid['theta_grid'], grid['alpha_grid']
    power_spec=np.zeros((len(theta_grid), len(alpha_grid)))
    const=-1j*2*np.pi*(mic_dist/wavelen)
    cov_mat=sig_recv.T[..., None] @ np.conjugate(sig_recv.T[:, None, :]) # (n_snap, n_sensors, n_sensors)
    for i in range(len(theta_grid)):
        for j in range(len(alpha_grid)):
            delta_alpha=alpha_grid[j]/(n_snap-1)
            doa=(theta_grid[i]+delta_alpha*np.arange(n_snap)).reshape(-1, 1) # (n_snap, 1)
            n_phi=np.sin(doa*np.pi/180.)*np.arange(n_sensors) # (n_snap, n_sensors)
            s_vec=np.exp(const*n_phi) # (n_snap, n_sensors)
            power_vec=np.abs(np.conjugate(s_vec[:, None, :]) @ cov_mat @ s_vec[:, :, None])
            power_spec[i, j]=power_vec.sum()/n_snap
    return power_spec 

if __name__=="__main__":
    theta_grid=np.round(np.arange(-80, 80.01, 0.5), decimals=2)
    alpha_grid=np.round(np.arange(-5, 5.01, 0.5), decimals=2)

    n_sources, n_sensors = 2, 8
    n_snap, freq, ss = 30, 300, 340
    wavelen=ss/freq
    mic_dist=0.5*wavelen

    get_source_doaparams(file_name='traindata_01', theta_grid=theta_grid, alpha_grid=alpha_grid, 
        n_sensors=n_sensors, n_sources=2, n_snr=6)
    
    theta_grid=np.round(np.arange(-80, 80.01, 0.3), decimals=2)
    alpha_grid=np.round(np.arange(-5, 5.01, 0.2), decimals=2)
    
    get_source_doaparams(file_name='traindata_02', theta_grid=theta_grid, alpha_grid=alpha_grid, 
        n_sensors=n_sensors, n_sources=2, n_snr=8)

    n_examples=200000
    get_random_source_doaparams(file_name='traindata_random', n_examples=n_examples, n_sources=2, 
        phi_range=[-80, 80], alpha_range=[-5, 5], snr_range=[0, 20], min_diff=5)
    

    
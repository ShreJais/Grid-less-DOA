import numpy as np
import pandas as pd
import os, time

from multiprocessing import Pool
from scipy.io import loadmat, savemat

# 2 sources.
def get_2_sources_data(thetas, alphas, f_name):
    df=pd.DataFrame(columns=['src1_theta', 'src2_theta','src1_alpha', 'src2_alpha','snr_db'])
    snr_list=np.arange(0, 20, 1)
    for snr_val in snr_list:
        for i in range(200):
            snr_db=snr_val
            df=df.append({'src1_theta': thetas[0], 'src2_theta': thetas[1], 'src1_alpha': alphas[0], 
                'src2_alpha': alphas[1], 'snr_db': snr_db}, ignore_index=True)
    df.to_csv(f_name)
    return df

def get_2movingsources_data(static_thetas, static_alphas, snr_val, f_name):
    df=pd.DataFrame(columns=['src1_theta', 'src2_theta','src1_alpha', 'src2_alpha', 'snr_db'])
    changing_theta= np.arange(-60, 55, 6)
    snr_db=snr_val
    for cth in changing_theta:
        for i in range(200):
            df=df.append({'src1_theta': static_thetas[0], 'src2_theta': cth, 
                'src1_alpha': static_alphas[0], 'src2_alpha': static_alphas[1], 'snr_db': snr_db},
                ignore_index=True)
    df.to_csv(f_name)
    return df

# 3 sources.
def get_3_sources_data(thetas, alphas, f_name):
    df=pd.DataFrame(
        columns=['src1_theta', 'src2_theta', 'src3_theta',
            'src1_alpha', 'src2_alpha', 'src3_alpha', 'snr_db'])
    snr_list=np.arange(10, 20, 0.5)
    
    for snr_val in snr_list:
        for i in range(200):
            snr_db=snr_val
            df=df.append({'src1_theta': thetas[0], 'src2_theta': thetas[1], 'src3_theta': thetas[2],
                'src1_alpha': alphas[0], 'src2_alpha': alphas[1], 'src3_alpha': alphas[2], 'snr_db': snr_db},
                ignore_index=True)
    df.to_csv(f_name)
    return df

def get_3movingsources_data(static_thetas, static_alphas, snr_val, f_name):
    df=pd.DataFrame(
        columns=['src1_theta', 'src2_theta', 'src3_theta', 'src1_alpha', 
                 'src2_alpha', 'src3_alpha', 'snr_db'])
    changing_theta= np.arange(-60, 55, 6)
    snr_db=snr_val
    for cth in changing_theta:
        for i in range(200):
            df=df.append(
                {'src1_theta': static_thetas[0], 'src2_theta': static_thetas[1], 'src3_theta': cth,
                    'src1_alpha': static_alphas[0], 'src2_alpha': static_alphas[1], 
                        'src3_alpha': static_alphas[2], 'snr_db': snr_db},
                ignore_index=True)
    df.to_csv(f_name)
    return df

def generate_signal(src_var, n_sources, n_sensors, wavelen, mic_dist, snr_db, 
    true_doa_parameters, seed, n_snap=30):
    
    assert len(true_doa_parameters['thetas']) == n_sources, ('Either the number of sources', 
        'is entered wrong or true_doa_parameters is wrongly defined.')
    
    # set seed for reproducibility.
    np.random.seed(seed)
    # Generate gaussian distributed source signal.
    # breakpoint() 
    src_var=np.array(src_var)
    sig_pow=(src_var**2).sum() / n_sources
    
    src_sig_real=src_var.reshape(-1, 1) * np.random.randn(n_sources, n_snap)
    src_sig_imag=src_var.reshape(-1, 1) * np.random.randn(n_sources, n_snap)

    sig=src_sig_real+1j*src_sig_imag
    
    # different way to represent signal.
    new_sig=np.zeros((n_sources*n_snap, n_snap), dtype=np.complex128) # shape: (n_sources*n_snap , n_snap)
    for i in range(n_sources):
        new_sig[n_snap*i:n_snap*(i+1),:]=sig[i, :] * np.eye(n_snap)

    # generate gaussian noise using given snr_db and computed noise_std.
    noise_std=np.round(np.sqrt(10**(-snr_db/10)*sig_pow), decimals=2)
    noise=noise_std*(np.random.randn(n_sensors, n_snap) + 1j*np.random.randn(n_sensors, n_snap))

    # Generate sensing matrix A, which contains steering vectors.
    const=-1j*2*np.pi*(mic_dist/wavelen)
    A=np.zeros((n_sensors, n_sources*n_snap), dtype=np.complex128)
    # sep_recv_sig=np.zeros((n_sources, n_sensors, n_snap), dtype=np.complex128)

    for i in range(n_sources):
        theta=true_doa_parameters['thetas'][i]
        alpha=true_doa_parameters['alphas'][i]
        delta_alpha= alpha/(n_snap-1)
        doa = theta+delta_alpha*np.arange(n_snap)
        n_phi=np.arange(n_sensors).reshape(-1, 1) * np.sin(doa*np.pi/180.) # shape: (n_sensors, n_snap)
        A[:, i*n_snap:(i+1)*n_snap]=np.exp(const*n_phi)
    #     sep_recv_sig[i, :, :]=np.matmul(A[:, i*n_snap:(i+1)*n_snap], new_sig[i*n_snap:(i+1)*n_snap, :])

    recv_sig=np.matmul(A, new_sig)
    recv_sig_plus_noise=recv_sig+noise
    return (recv_sig_plus_noise, recv_sig, noise_std, sig)

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

def get_testdata(file_name):
    # For TL-CBF.
    theta_grid=np.arange(-80, 81, 1)
    alpha_grid=np.arange(-5, 6, 1)
    grid = {'theta_grid': theta_grid, 'alpha_grid': alpha_grid}

    csv_fp=os.path.join('test_data', f'testdata_{file_name}.csv')
    csv_data=np.array(pd.read_csv(csv_fp))[:, 1:]
    print(f'Shape: {csv_data.shape}')

    recvsig_with_noise=np.zeros((len(csv_data), 8, 30), dtype=np.complex128)
    recvsig_without_noise=np.zeros((len(csv_data), 8, 30), dtype=np.complex128)
    std_noise = np.zeros((len(csv_data),))
    sig_amplitude=np.zeros((len(csv_data), n_sources, n_snap), dtype=np.complex128)

    for i in range(len(csv_data)):
        params=csv_data[i]
        true_doa_params={}
        true_doa_params['thetas']=params[:n_sources]
        true_doa_params['alphas']=params[n_sources:-1]
        snr_db=params[-1]
        # recv_sig_plus_noise, recv_sig, noise_std, sig=
        (recvsig_with_noise[i, :, :], recvsig_without_noise[i, :, :], 
            std_noise[i], sig_amplitude[i, :, :])=generate_signal(src_var=src_var, n_sources=n_sources, n_sensors=n_sensors, 
                wavelen=wavelen, mic_dist=mic_dist, snr_db=snr_db, true_doa_parameters=true_doa_params, seed=i, n_snap=30)
    
    recv_sig = {'recvsig_with_noise': recvsig_with_noise, 'recvsig_without_noise': recvsig_without_noise}
    sig_amp_and_noise_std={'sig_amp': sig_amplitude, 'noise_std': std_noise}

    savemat('./test_data/'+f'recv_sig_{file_name}.mat', recv_sig)
    savemat('./test_data/'+f'sig_amp_and_noise_{file_name}.mat', sig_amp_and_noise_std)

    # 2D power spectrum - TL-CBF.
    recv_sig_path='./test_data/'+f'recv_sig_{file_name}.mat'
    recv_sig_noise=loadmat(recv_sig_path)['recvsig_with_noise']
    power_tlcbf=np.zeros((len(csv_data), len(alpha_grid), len(theta_grid)))

    global run_task
    def run_task(i):
        # print(i)
        power_spec=power_cbf(n_sensors=n_sensors, sig_recv=recv_sig_noise[i], 
            grid=grid, wavelen=wavelen, mic_dist=mic_dist, n_snap=n_snap)
        # power_spec=normalize_data(x=power_spec, axis=(0, 1))
        results={'power_spec': power_spec.T}
        return results

    start_time = time.time()
    NUM_PROCESS=10
    with Pool(NUM_PROCESS) as p:
        pool_result=p.map(run_task, list(range(len(csv_data)))) # range(len(csv_data))
    end_time=time.time()
    print(f'time taken: {end_time-start_time}')
    
    for i, out in enumerate(pool_result):
        power_tlcbf[i, :, :]=out['power_spec']
    
    np.save('./test_data/'+f'power_spec_tlcbf_{file_name}.npy', power_tlcbf)


if __name__=="__main__":
    # Generate data.
    n_sensors= 8
    n_snap, freq, ss = 30, 300, 340  
    wavelen=ss/freq
    mic_dist=0.5*wavelen
    
    os.makedirs('test_data', exist_ok=True)

    # Test-Data:1  2-sources
    n_sources=2
    file_name='2sources_close_0_20snr'
    theta=[60.4, 50.5]
    alpha=[3.5, -2.5]
    src_var=[1, 0.6]
    df=get_2_sources_data(thetas=theta, alphas=alpha, 
        f_name='./test_data/'+f'testdata_{file_name}.csv')
    get_testdata(file_name=file_name)

    # Test-Data:2 2-sources.
    n_sources=2
    file_name='2sources_1moving_10dbsnr'
    thetas=[20.4]
    alphas=[3.5, -2.5]
    src_var=[1, 0.75]
    df=get_2movingsources_data(static_thetas=thetas, static_alphas=alphas, snr_val=10, 
        f_name='./test_data/'+f'testdata_{file_name}.csv')
    get_testdata(file_name=file_name)

    # Test-Data:3  3-sources.
    n_sources=3
    file_name='3sources_2close_1far_10_20snr'
    thetas=[60.5, 5.5, -7.5]
    alphas=[2.5, 1.2, 3.5]
    src_var=[1, 0.8, 0.5]
    df=get_3_sources_data(thetas=thetas, alphas=alphas, f_name='./test_data/'+f'testdata_{file_name}.csv')
    get_testdata(file_name=file_name)

    # Test-Data:4 3-sources.
    n_sources=3
    file_name='3sources_1moving_5dbsnr'
    thetas=[60.5, 5.5]
    alphas=[2.5, 1.2, -3.8]
    src_var=[1, 1, 0.5]
    snr_val=5
    df=get_3movingsources_data(static_thetas=thetas, static_alphas=alphas, snr_val=snr_val, 
        f_name='./test_data/'+f'testdata_{file_name}.csv')
    get_testdata(file_name=file_name)
    

    
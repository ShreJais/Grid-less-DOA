import os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import utils

from scipy.io import savemat

if __name__=="__main__":
    # Small dataset.
    csv_fp='./traindata_01.csv'
    csv_data=pd.read_csv(csv_fp)

    dataset_path='/scratch/sj/dataset'
    img_path='/scratch/sj/img/'
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    recvsignoise_path=os.path.join(img_path, 'recv_sig_with_noise')
    recvsig_path=os.path.join(img_path, 'recv_sig')
    os.makedirs(recvsignoise_path, exist_ok=True)
    os.makedirs(recvsig_path, exist_ok=True)

    # Parameters for dataset generation.
    n_sources, n_sensors= 2, 8
    n_snap, freq, ss = 30, 300, 340  
    wavelen=ss/freq
    mic_dist=0.5*wavelen

    # for TL-CBF.
    theta_grid=np.arange(-80, 81, 1)
    alpha_grid=np.arange(-5, 6, 1)
    grid = {'theta_grid': theta_grid, 'alpha_grid': alpha_grid}

    global run_task
    def run_task(i):
    # for i in range(10):
        print(i)
        (src1_theta, src2_theta, src1_alpha, 
            src2_alpha, snr_db) = csv_data[['src1_theta', 'src2_theta', 'src1_alpha', 'src2_alpha', 'snr_db']].values[i]
        true_doa_params={
            'source1': {'src1_theta': src1_theta, 'src1_alpha': src1_alpha}, 
            'source2': {'src2_theta': src2_theta, 'src2_alpha': src2_alpha}
            }
        
        recvsig_noise, recvsig, noise_std, sig_amp=utils.generate_signal(n_sources=n_sources, 
            n_sensors=n_sensors, snr_db=snr_db, wavelen=wavelen, mic_dist=mic_dist, 
            doa_params=true_doa_params, seed=i, n_snap=n_snap)
        
        power_spec=utils.power_cbf(n_sensors=n_sensors, sig_recv=recvsig_noise, grid=grid, wavelen=wavelen, 
            mic_dist=mic_dist, n_snap=n_snap)
        
        results={'power_spec': power_spec.T, 'recv_sig_with_noise': recvsig_noise, 
            'recv_sig_without_noise': recvsig, 'noise_std': noise_std, 'sig_amp': sig_amp}
        return results
    
    start_time=time.time()
    NUM_PROCESS=60
    with Pool(NUM_PROCESS) as p:
        pool_result=p.map(run_task, list(range(len(csv_data))))
    end_time=time.time()
    print(f'time taken: {end_time - start_time}')

    power_spec = np.zeros((len(csv_data), len(alpha_grid), len(theta_grid)))
    recvsig_noise=np.zeros((len(csv_data), n_sensors, n_snap), dtype=np.complex128)
    recvsig=np.zeros((len(csv_data), n_sensors, n_snap), dtype=np.complex128)
    std_noise = np.zeros((len(csv_data),))
    sig_amplitude=np.zeros((len(csv_data), n_sources, n_snap), dtype=np.complex128)

    for i, out in enumerate(pool_result):
        power_spec[i, :, :]=out['power_spec']
        recvsig_noise[i, :, :]=out['recv_sig_with_noise']
        recvsig[i, :, :]=out['recv_sig_without_noise']
        std_noise[i]=out['noise_std']
        sig_amplitude[i, :, :]=out['sig_amp']
    
    recv_sig = {'recv_sig_with_noise': recvsig_noise, 'recv_sig_without_noise': recvsig_noise}
    sig_amp_and_noise_std={'sig_amp': sig_amplitude, 'noise_std': std_noise}

    np.save(f'{dataset_path}'+'power_spec_training05.npy', power_spec)
    savemat(dataset_path+'recv_sig_new.mat', recv_sig)
    savemat(dataset_path+'sig_amp_and_noise_std_05new.mat', sig_amp_and_noise_std)
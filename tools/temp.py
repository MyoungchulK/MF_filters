import numpy as np
import h5py

def temp_loader(c_path, shower):

    temp_path = f'{c_path}/temp/'
    if shower == 'EM':
        temp_name = f'AraOut.setup.template.S.N0.E18.D500.3A.O0.VH.NuE.Nu.CC.{shower}.El1.0.txt.run0.fft_w_band.h5'
    elif shower == 'HAD':
        temp_name = f'AraOut.setup.template.S.N0.E18.D500.3A.O0.VH.NuMu.Nu.NC.{shower}.El1.0.txt.run0.fft_w_band.h5'
    temp_file = h5py.File(temp_path+temp_name, 'r')
    
    temp_v = temp_file['temp_v'][:]
    theta_info = temp_file['num_theta'][:]
    n_theta = len(theta_info)
    theta_w = np.abs(theta_info[1] - theta_info[0])
    peak_i = temp_file['wf_peak_index'][:]
    del temp_path, temp_name, temp_file, theta_info
    
    print('Template loading is done!')

    return temp_v, n_theta, theta_w, peak_i#, theta_info

def temp_loader_ant(c_path, shower,ant_model):

    temp_path = f'{c_path}/temp/'
    if shower == 'EM':
        temp_name = 'AraOut.setup.template.S.N0.E18.D500.3A.O0.VH.NuE.Nu.CC.{shower}.El1.0.{ant_model}.txt.run0.fft_w_band.h5'
    elif shower == 'HAD':
        temp_name = 'AraOut.setup.template.S.N0.E18.D500.3A.O0.VH.NuMu.Nu.NC.{shower}.El1.0.{ant_model}.txt.run0.fft_w_band.h5'
    temp_file = h5py.File(temp_path+temp_name, 'r')

    temp_v = temp_file['temp_v'][:]
    theta_info = temp_file['num_theta'][:]
    n_theta = len(theta_info)
    theta_w = np.abs(theta_info[1] - theta_info[0])
    peak_i = temp_file['wf_peak_index'][:]
    del temp_path, temp_name, temp_file, theta_info

    print('Template loading is done!')

    return temp_v, n_theta, theta_w, peak_i#, theta_info

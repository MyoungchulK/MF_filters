import os
import numpy as np
from tqdm import tqdm
import h5py

def temp_sim_collector(Data, Station, Year):

    print('Collecting temp sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import get_path_info_v2

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_freq_pad = True, use_rfft = True, verbose = True, new_wf_time = wf_time)
    temp_time = wf_int.pad_zero_t
    wf_len = wf_int.pad_len
    temp_freq = wf_int.pad_zero_freq
    fft_len = wf_int.pad_fft_len

    # output arrays
    temp_temp = np.full((wf_len, num_ants, num_evts), 0, dtype = float)
    temp_temp_rfft = np.full((fft_len, num_ants, num_evts), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True)
        del wf_v
        temp_temp[:, :, evt] = wf_int.pad_v

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        temp_temp_rfft[:, :, evt] = wf_int.pad_fft
    del ara_root, num_evts, wf_int, wf_time

    # parameters
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    flavor = ['NuE', 'NuMu']
    charge = ['CC', 'NC']
    temp_param_len = 7 # Event_id Flavor CC_NC Elst Ant_Res Off_cone Useful_Ch
    temp_temp_param = np.full((temp_param_len, num_temps), 0, dtype = int)
    param_path = f'/home/mkim/analysis/MF_filters/sim/ARA0{Station}/sim_temp_setup_full/temp_A{Station}_R{config}_setup_parameter.txt' 
    with open(param_path, 'r') as f:
        counts = 0
        for lines in f:
            if counts == 0:
                continue
            line_p = lines.split()
            temp_temp_param[0, counts] = int(line_p[0])
            temp_temp_param[1, counts] = int(flavor.index(str(line_p[1])))
            temp_temp_param[2, counts] = int(charge.index(str(line_p[2])))
            temp_temp_param[3, counts] = int(float(line_p[3]) * 10)
            temp_temp_param[4, counts] = int(line_p[4])
            temp_temp_param[5, counts] = int(line_p[5])
            temp_temp_param[6, counts] = int(line_p[6])
            del line_p
            counts += 1
        del counts
    del config, flavor, charge, param_path

    # arrival time table
    

    # output arrays
    ant_idx = np.arange(num_ants, dtype = int)
    ele_idx = np.array([-60, -40, -20, 0, 20, 40, 60], dtype = int)
    ele_idx1 = np.copy(ele_idx)
    ele_idx1[4:] * -1
    phi_idx = np.array([-120, -60, 0, 60, 120,180], dtype = int) 
    off_idx = np.array([0, 2, 4], dtype = int)
    sho_idx = np.array([0, 1], dtype = int)
    num_temps = len(ele_idx) * len(phi_idx) * len(off_idx) * len(sho_idx)    
    param_len = 4 # EM/HAD, Ant_Res, Phi, Off
    temp_param = np.full((param_len, num_temps), 0, dtype = int)
    temp = np.full((wf_len, num_ants, num_temps), 0, dtype = float)
    temp_rfft = np.full((fft_len, num_ants, num_temps), 0, dtype = float)
    counts = 0
    for sho in range(len(sho_idx)):
        for ele in range(len(ele_idx)):
            for phi in range(len(phi_idx)):
                for off in range(len(off_idx)):
                    temp_param[0, counts] = sho_idx[sho]
                    temp_param[1, counts] = ele_idx[ele]
                    temp_param[2, counts] = phi_idx[phi]
                    temp_param[3, counts] = off_idx[off]
                    for ant in range(len(ant_idx)):
                        idxs = np.all((temp_temp_param[6] == ant_idx[ant], temp_temp_param[1] == int(sho_idx[sho] + 1), temp_temp_param[4] == ele_idx1[ele], temp_temp_param[5] == off_idx[off]), axis = 0)
                        params = temp_temp_param[:, idxs]
                        evts_idx = params[0]
                        ants_idx = params[-1]


                        temp[:, ant, counts] = temp_temp[:, ants_idx, evts_idx]
                        temp_rfft[:, ant, counts] = temp_temp_rfft[:, ants_idx, evts_idx]
                        del , params, evts_idx,, ants_idx
                    counts += 1     
    del wf_len, fft_len, counts, temp_temp_param, temp_param_len, param_len, num_temps, sho_idx, off_idx, phi_idx, ele_idx1, ele_idx, ant_idx, temp_temp, temp_temp_rfft

    print('Temp collecting is done!')

    return {'entry_num':entry_num,
            'temp_time':temp_time,
            'temp_freq':temp_freq,
            'num_temps':num_temps,
            'arr_time':arr_time,
            'temp_param':temp_param,
            'temp':temp,
            'temp_rfft':temp_rfft} 


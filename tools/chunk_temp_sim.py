import os, sys
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
    ara_root.get_sub_info(Data, get_temp_info = True)
    num_evts = ara_root.num_evts
    wf_time = ara_root.wf_time
    dt = ara_root.time_step[0]
    signal_bin = ara_root.signal_bin[0]
    rec_ang = ara_root.rec_ang[0] 

    # parameters
    flavor = ['NuE', 'NuMu']
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    param_path = f'/home/mkim/analysis/MF_filters/sim/ARA0{Station}/sim_temp_setup_full/temp_A{Station}_R{config}_setup_parameter.txt'
    temp_temp_param = np.full((4, num_evts), 0, dtype = int) # antenna ch, shower, antenna response, off-cone angle
    with open(param_path, 'r') as f:
        counts = 0
        for lines in f:
            if counts == 0:
                counts += 1
                continue
            line_p = lines.split()
            p_idx = int(counts - 1)
            temp_temp_param[0, p_idx] = int(line_p[6]) # antenna ch
            temp_temp_param[1, p_idx] = int(flavor.index(str(line_p[1]))) # shower 0 (EM) ot 1 (HAD)
            temp_temp_param[2, p_idx] = int(line_p[4]) # antenna response
            temp_temp_param[3, p_idx] = int(float(line_p[5])) # off-cone angle
            del line_p, p_idx
            counts += 1
        del counts
    del param_path, flavor

    # temp arrays
    sho_bin = np.array([0, 1], dtype = int)
    res_bin = np.array([-60, -40, -20, -10, 0], dtype = int)
    off_bin = np.array([0, 2, 4], dtype = int)
    sho_bin_len = len(sho_bin)
    res_bin_len = len(res_bin)
    off_bin_len = len(off_bin)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, verbose = True, new_wf_time = wf_time)
    temp_time = wf_int.pad_zero_t
    wf_len = wf_int.pad_len

    # temp arrays
    temp = np.full((wf_len, num_ants, sho_bin_len, res_bin_len, off_bin_len), 0, dtype = float)
    sig_shift = np.full((num_ants, sho_bin_len, res_bin_len, off_bin_len), np.nan, dtype = float)
    rec_angle = np.copy(sig_shift)
    snr = np.copy(sig_shift)
    del sho_bin_len, res_bin_len, off_bin_len

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        # indexs
        ant_ch = temp_temp_param[0, evt] 
        sho_loc = np.where(sho_bin == temp_temp_param[1, evt])[0]
        ele_loc = np.where(res_bin == temp_temp_param[2, evt])[0]
        off_loc = np.where(off_bin == temp_temp_param[3, evt])[0]
        if len(sho_loc) == 0 or len(ele_loc) == 0 or len(off_loc) == 0: 
            print('WRONG!! Error!!')
            print(evt)
            print(temp_temp_param[1, evt], temp_temp_param[2, evt], temp_temp_param[3, evt])
            sys.exit(1)
        sho_loc = sho_loc[0]
        ele_loc = ele_loc[0]
        off_loc = off_loc[0]    
        sig_bin_evt = signal_bin[ant_ch, evt]
        sig_bins = int(np.round(sig_bin_evt / dt))
        sig_shift[ant_ch, sho_loc, ele_loc, off_loc] = sig_bin_evt
        rec_angle[ant_ch, sho_loc, ele_loc, off_loc] = rec_ang[ant_ch, evt]
        del sig_bin_evt
        
        # sim wf
        wf_v = ara_root.get_rf_wfs(evt)
        p2p_evt = np.full((num_ants), np.nan, dtype = float)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_p2p = True, use_band_pass = True)
            p2p_evt[ant] = wf_int.int_p2p
        pad_v_ant = wf_int.pad_v[:, ant_ch]
        del wf_v

        # signal shift
        temp_sig_shift = np.full((wf_len), 0, dtype = float) 
        if sig_bins > 0:
            temp_sig_shift[:-sig_bins] = pad_v_ant[sig_bins:]
        elif sig_bins < 0: 
            temp_sig_shift[-sig_bins:] = pad_v_ant[:sig_bins] 
        else:
            temp_sig_shift[:] = pad_v_ant
        del sig_bins, pad_v_ant

        temp[:, ant_ch, sho_loc, ele_loc, off_loc] = temp_sig_shift
        snr[ant_ch, sho_loc, ele_loc, off_loc] = p2p_evt[ant_ch]
        del ant_ch, sho_loc, ele_loc, off_loc, temp_sig_shift, p2p_evt
    del ara_root, num_evts, num_ants, wf_int, wf_time, wf_len, dt, signal_bin, rec_ang, temp_temp_param

    snr /= 2
    r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_sim_merge/rms_A{Station}_R{config}.h5'
    print('rms path:', r_path)
    hf = h5py.File(r_path, 'r')
    rms_mean = hf['rms_mean'][:]
    snr /= rms_mean[:, np.newaxis, np.newaxis, np.newaxis]
    del config, r_path, hf, rms_mean

    print('Temp collecting is done!')

    return {'temp_time':temp_time,
            'temp':temp,
            'sig_shift':sig_shift,
            'rec_angle':rec_angle,
            'snr':snr,
            'sho_bin':sho_bin,
            'res_bin':res_bin,
            'off_bin':off_bin}


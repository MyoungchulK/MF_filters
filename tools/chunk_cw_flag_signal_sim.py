import os
import numpy as np
from tqdm import tqdm
import h5py

def cw_flag_signal_sim_collector(Data, Station, Year):

    print('Collectin cw flag signal sim starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_cw_filters import py_phase_variance
    from tools.ara_cw_filters import py_testbed
    from tools.ara_cw_filters import group_bad_frequency
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_utility import size_checker

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time

    # bad antenna
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
    ex_run = get_example_run(Station, config)
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, verbose = True, new_wf_time = wf_time)
    freq_range = wf_int.pad_zero_freq
    fft_len = len(freq_range)

    # cw class
    baseline_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/baseline_sim_merge/baseline_A{Station}_R{config}.h5' 
    print('baseline path:', baseline_path)
    cw_testbed = py_testbed(Station, ex_run, freq_range, verbose = True, use_st_pair = True, sim_path = baseline_path)
    testbed_params = np.array([cw_testbed.dB_cut, cw_testbed.dB_cut_broad, cw_testbed.num_coinc, cw_testbed.freq_range_broad, cw_testbed.freq_range_near])
    cw_phase = py_phase_variance(Station, ex_run, freq_range)
    evt_len = cw_phase.evt_len
    evt_len_s = int(evt_len - 1)
    phase_params = np.array([cw_phase.sigma_thres, evt_len])
    del config, sim_run, baseline_path

    phase_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/phase_sim/'
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    h5_file_name = f'phase_{data_name}.h5'    
    print('phase path: {phase_path}{h5_file_name}')
    hf = h5py.File(f'{phase_path}{h5_file_name}', 'r')
    phase_arr = hf['phase_arr'][:]
    num_phases = int(phase_arr.shape[-1])
    print(phase_arr.shape)
    del hf, h5_file_name, phase_path

    # output array
    sigma = []
    phase_idx = []
    testbed_idx = []
    phase_n_idx = np.full((evt_len_s, 2, num_evts), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:

        ## wf
        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True)
        del wf_v

        ## fft
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_phase = True, use_abs = True, use_norm = True, use_dBmHz = True)
        rfft_phase = wf_int.pad_phase
        rfft_dbmhz = wf_int.pad_fft

        ## testbed
        cw_testbed.get_bad_magnitude(rfft_dbmhz, 0)
        testbed_idxs = cw_testbed.bad_idx
        testbed_idx.append(testbed_idxs)

        ## signal + noise phase array
        ran_idx = np.random.choice(num_phases, size = evt_len_s * 2, replace=False) 
        ran_idx_f = ran_idx[:evt_len_s]
        ran_idx_b = ran_idx[evt_len_s:]
        phase_n_idx[:, 0, evt] = ran_idx_f
        phase_n_idx[:, 1, evt] = ran_idx_b
        phase_tot_front = np.full((fft_len, num_ants, evt_len), np.nan, dtype = float)
        phase_tot_back = np.copy(phase_tot_front) 
        phase_tot_front[:, :, :evt_len_s] = phase_arr[:, :, ran_idx_f]
        phase_tot_front[:, :, -1] = rfft_phase
        phase_tot_back[:, :, :evt_len_s] = phase_arr[:, :, ran_idx_b]
        phase_tot_back[:, :, -1] = rfft_phase
        del ran_idx, ran_idx_f, ran_idx_b

        ## pahse front
        cw_phase.get_phase_differences_at_once(phase_tot_front)
        cw_phase.get_bad_phase()
        sigmas = cw_phase.bad_sigma
        phase_idxs = cw_phase.bad_idx
        sigma.append(sigmas)
        phase_idx.append(phase_idxs)
        del phase_tot_front

        ## phase_back
        cw_phase.get_phase_differences_at_once(phase_tot_back)
        cw_phase.get_bad_phase()
        sigmas = cw_phase.bad_sigma
        phase_idxs = cw_phase.bad_idx
        sigma[evt] = np.concatenate((sigma[evt], sigmas))
        phase_idx[evt] = np.concatenate((phase_idx[evt], phase_idxs))
        del rfft_phase, rfft_dbmhz, phase_tot_back
        print(sigma[evt], phase_idx[evt], testbed_idx[evt]) # for debug
    del ara_root, num_ants, wf_time, wf_int, cw_testbed, cw_phase, evt_len, phase_arr, fft_len, evt_len_s, num_phases

    # to numpy array
    sigma = np.asarray(sigma, dtype=object)
    phase_idx = np.asarray(phase_idx, dtype=object)
    testbed_idx = np.asarray(testbed_idx, dtype=object)

    # group bad frequency
    cw_freq = group_bad_frequency(Station, ex_run, freq_range, verbose = True) # constructor for bad frequency grouping function
    del ex_run

    # output array
    bad_range = []

    # loop over the events
    for evt in tqdm(range(num_evts)):
       #if evt <100:
        
        bad_range_evt = cw_freq.get_pick_freqs_n_bands(sigma[evt], phase_idx[evt], testbed_idx[evt]).flatten()
        bad_range.append(bad_range_evt)
    del num_evts, cw_freq 
 
    # to numpy array
    bad_range = np.asarray(bad_range, dtype=object)

    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_band_sim/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'cw_band_{data_name}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('entry_num', data=entry_num, compression="gzip", compression_opts=9)
    try:
        hf.create_dataset('bad_range', data=bad_range, compression="gzip", compression_opts=9)
    except TypeError:
        dt = h5py.vlen_dtype(np.dtype(float))
        hf.create_dataset('bad_range', data=bad_range, dtype = dt, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))
    del slash_idx, dot_idx, data_name, h5_file_name

    print('CW flag signal sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'freq_range':freq_range,
            'sigma':sigma,
            'phase_idx':phase_idx,
            'phase_n_idx':phase_n_idx,
            'testbed_idx':testbed_idx,
            'bad_range':bad_range,
            'testbed_params':testbed_params,
            'phase_params':phase_params}

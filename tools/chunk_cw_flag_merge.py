import os
import numpy as np
import h5py
from tqdm import tqdm

def cw_flag_merge_collector(Data, Station, Run, analyze_blind_dat = False, no_tqdm = False):

    print('Cw merge starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_utility import size_checker
    from tools.ara_cw_filters import group_bad_frequency
    from tools.ara_quality_cut import get_bad_events

    # load cw flag
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_flag_old{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cw_dat = f'{output_path}cw_flag{blind_type}_A{Station}_R{Run}.h5'
    print(f'cw_flag_old_path:{cw_dat}', size_checker(f'{cw_dat}'))
    cw_hf = h5py.File(cw_dat, 'r')
    config = cw_hf['config'][:]
    evt_num = cw_hf['evt_num'][:]
    bad_ant = cw_hf['bad_ant'][:]
    freq_range = cw_hf['freq_range'][:]
    sigma = cw_hf['sigma'][:] # sigma value for phase variance
    phase_idx = cw_hf['phase_idx'][:] # bad frequency indexs by phase variance
    testbed_idx = cw_hf['testbed_idx'][:] # bad freqency indexs by testbed method
    bad_range = cw_hf['bad_range'][:]
    testbed_params = cw_hf['testbed_params'][:]
    phase_params = cw_hf['phase_params'][:]
    del cw_dat, cw_hf

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_st_dat = run_info.get_result_path(file_type = 'cw_flag_st', verbose = True, force_blind = True) # get the h5 file path
    cw_st_hf = h5py.File(cw_st_dat, 'r')
    testbed_idx_st = cw_st_hf['testbed_idx'][:]
    del cw_st_dat, cw_st_hf, run_info

    if len(testbed_idx) != len(testbed_idx_st):
        print('Wrong!!!!!:', len(testbed_idx), len(testbed_idx_st), Station, Run)

    daq_qual_cut = get_bad_events(Station, Run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, use_1st = True)[0]

    # group bad frequency
    cw_freq = group_bad_frequency(Station, Run, freq_range, verbose = True) # constructor for bad frequency grouping function

    # output array  
    testbed_idx_merge = []
    bad_range_merge = []    
    empty = np.full((0), 0, dtype = int)
    empty_float = np.full((0), np.nan, dtype = float)
    num_evts = len(evt_num)
    evt_check = np.full((num_evts), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
    
        # quality cut
        if daq_qual_cut[evt]:
            testbed_idx_merge.append(empty)
            bad_range_merge.append(empty_float)
            continue
 
        test_old = testbed_idx[evt].astype(int)
        test_st = testbed_idx_st[evt].astype(int)
        st_flag = np.any(~np.in1d(test_st, test_old))
        if ~st_flag: 
            testbed_idx_merge.append(test_old)
            bad_old = bad_range[evt]
            bad_range_merge.append(bad_old)
            continue
          
        evt_check[evt] = 1 
        cw_merge = np.unique(np.concatenate((test_old, test_st))).astype(int)
        testbed_idx_merge.append(cw_merge)
        bad_range_evt = cw_freq.get_pick_freqs_n_bands(sigma[evt], phase_idx[evt], cw_merge).flatten()
        bad_range_merge.append(bad_range_evt)

    print(f'tot_evt: {num_evts}, bad_evt: {np.sum(evt_check)}, bad_ratio{np.round(np.sum(evt_check)/num_evts, 2)}')
    testbed_idx_merge = np.asarray(testbed_idx_merge)
    bad_range_merge = np.asarray(bad_range_merge)

    hf_str = ['config', 'evt_num', 'bad_ant', 'freq_range', 'sigma', 'phase_idx', 'testbed_idx', 'bad_range', 'testbed_params', 'phase_params']
    hf_data = [config, evt_num, bad_ant, freq_range, sigma, phase_idx, testbed_idx_merge, bad_range_merge, testbed_params, phase_params]
    hf_len = len(hf_str)

    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_flag{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'cw_flag{blind_type}_A{Station}_R{Run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    for h in range(hf_len):
        print(hf_str[h], hf_data[h].shape)
        try:
            hf.create_dataset(hf_str[h], data=hf_data[h], compression="gzip", compression_opts=9) 
        except TypeError:
            dt = h5py.vlen_dtype(np.dtype(float))
            hf.create_dataset(hf_str[h], data=hf_data[h], dtype = dt, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))

    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_band{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'cw_band{blind_type}_A{Station}_R{Run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    try:
        hf.create_dataset('bad_range', data=bad_range_merge, compression="gzip", compression_opts=9)
    except TypeError:
        dt = h5py.vlen_dtype(np.dtype(float))
        hf.create_dataset('bad_range', data=bad_range_merge, dtype = dt, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))

    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_check{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'cw_check{blind_type}_A{Station}_R{Run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('evt_check', data=evt_check, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))

    print('cw merge is done!')

    return








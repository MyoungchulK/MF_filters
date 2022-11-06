import os
import numpy as np
from tqdm import tqdm
import h5py

def l2_temp_collector(Data, Ped, analyze_blind_dat = False, use_condor = False):

    print('Collecting l2 starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_utility import size_checker
    from tools.ara_run_manager import condor_info_loader
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    irs_block_number = ara_uproot.irs_block_number
    irs_block = []
    for evt in range(num_evts):
        blk_evt = irs_block_number[evt][num_ddas::num_ddas]
        blk_evt = blk_evt.astype(int)
        irs_block.append(blk_evt)
    irs_block = np.asarray(irs_block)
    st = ara_uproot.station_id
    run = ara_uproot.run
    year, month, date, unix = ara_uproot.get_run_time()
    ara_root = ara_root_loader(Data, Ped, st, unix)
    del num_ddas, irs_block_number

    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    config = run_info.get_config_number()
    run_config = np.array([st, run, num_evts, config, year, month, date, unix], dtype = int)
    del config, run_info

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    daq_sum = np.nansum(pre_qual.get_daq_structure_errors(), axis = 1)
    read_sum = np.nansum(pre_qual.get_readout_window_errors(), axis = 1)
    daq_cut = (daq_sum + read_sum).astype(int)
    del pre_qual, daq_sum, read_sum, ara_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True)

    # output
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    true_output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/l2{blind_type}/'
    if not os.path.exists(true_output_path):
        os.makedirs(true_output_path)

    condor_info = condor_info_loader(use_condor = use_condor, verbose = True)
    if use_condor:
        output_path = condor_info.local_path
    else:
        output_path = true_output_path   

    dt = h5py.vlen_dtype(np.dtype(int))
    dt_f = h5py.vlen_dtype(np.dtype(float))

    h5_file_name = f'l2{blind_type}_A{st}_R{run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('config', data=run_config, compression="gzip", compression_opts=9)
    hf.create_dataset(f'evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset(f'entry_num', data=entry_num, compression="gzip", compression_opts=9)
    hf.create_dataset(f'unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset(f'pps_number', data=pps_number, compression="gzip", compression_opts=9)
    hf.create_dataset(f'trig_type', data=trig_type, compression="gzip", compression_opts=9)
    hf.create_dataset(f'daq_cut', data=daq_cut, compression="gzip", compression_opts=9)
    if len(irs_block.shape) != 1:
        hf.create_dataset(f'irs_block', data=irs_block, compression="gzip", compression_opts=9)
    else:
        hf.create_dataset(f'irs_block', data=irs_block, dtype = dt, compression="gzip", compression_opts=9)
    del blind_type, st, run, output_path, run_config, evt_num, entry_num, unix_time, pps_number, trig_type, year, month, date, unix

    num_bins = np.full((num_ants, num_evts), 0, dtype = int)
    # loop over the events
    for evt in tqdm(range(num_evts)):
    #for evt in range(num_evts):
      #if evt <100:        
        
        if daq_cut[evt]:
            wfs = np.full((0, num_ants), np.nan, dtype = float)
            hf.create_dataset(f'entry{evt}', data=wfs, compression="gzip", compression_opts=9)
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        pad_num = wf_int.pad_num
        fil_wfs = wf_int.pad_v
        num_bins[:, evt] = pad_num

        max_num_bins = np.nanmax(pad_num)
        wfs = fil_wfs[:max_num_bins]
        hf.create_dataset(f'entry{evt}', data=wfs, compression="gzip", compression_opts=9)
    
        
        #wfs = []
        #for ant in range(num_ants):
        #    new_wf = fil_wfs[:pad_num[ant], ant]
        #    new_wf = new_wf.astype(float)
        #    wfs.append(new_wf)
        #wfs = np.asarray(wfs)
        #if len(wfs.shape) != 1:       
        #    hf.create_dataset(f'entry{evt}', data=wfs, compression="gzip", compression_opts=9)
        #else:
        #    hf.create_dataset(f'entry{evt}', data=wfs, dtype = dt_f, compression="gzip", compression_opts=9)
        #del wfs, pad_num, fil_wfs
    hf.create_dataset(f'num_bins', data=num_bins, compression="gzip", compression_opts=9)
    del dt, dt_f, ara_root, num_evts, num_ants, wf_int, daq_cut
    hf.close()

    Output = condor_info.get_condor_to_target_path(h5_file_name, true_output_path)
    print(f'output is {Output}')
    del true_output_path, condor_info, h5_file_name

    # quick size check
    size_checker(Output)
    del Output

    print('L2 collecting is done!')

    return









import os
import numpy as np
from tqdm import tqdm
import h5py

def l2_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting l2 starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_utility import size_checker

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)

    # pre quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    daq_sum = np.nansum(pre_qual.get_daq_structure_errors(), axis = 1)
    read_sum = np.nansum(pre_qual.get_readout_window_errors(), axis = 1)
    daq_qual_cut_sum = (daq_sum + read_sum).astype(int)
    del pre_qual, daq_sum, read_sum, ara_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # output
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/l2{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'{output_path}l2{blind_type}_A{st}_R{run}.h5'
    hf = h5py.File(h5_file_name, 'w')
    #evts = np.full((int(48*20/0.5), num_ants, num_evts), np.nan, dtype = float)
    del st, run, output_path

    # loop over the events
    for evt in tqdm(range(num_evts)):
    #for evt in range(num_evts):
      #if evt <100:        
        
        if daq_qual_cut_sum[evt]:
            wfs = np.full((0, num_ants), np.nan, dtype = float)
            hf.create_dataset(f'entry{evt}', data=wfs, compression="gzip", compression_opts=9)
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        max_num_bins = np.nanmax(wf_int.pad_num)
        wfs = wf_int.pad_v[:max_num_bins]
        #evts[:max_num_bins, :, evt] = wfs
        #print(max_num_bins)
        #print(wfs.shape)
        hf.create_dataset(f'entry{evt}', data=wfs, compression="gzip", compression_opts=9)
        del max_num_bins, wfs
    del ara_root, num_evts, num_ants, wf_int, daq_qual_cut_sum
    #hf.create_dataset(f'evts', data=evts, compression="gzip", compression_opts=9)
    hf.close()

    # quick size check
    size_checker(h5_file_name)
    del h5_file_name

    print('L2 collecting is done!')

    return









import numpy as np
from tqdm import tqdm
import h5py

def snr_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting snr starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
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
    time_bins, sec_per_min = ara_uproot.get_event_rate(use_time_bins = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    pre_qual_cut_sum = daq_hf['pre_qual_cut_sum'][:]
    daq_qual_cut_sum = daq_hf['tot_qual_cut_sum'][:]
    del ara_uproot, run_info, daq_dat, daq_hf

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # output array  
    rms = np.full((num_ants, num_evts), np.nan, dtype = float)
    p2p = np.copy(rms)

    # loop over the events
    #for evt in tqdm(range(num_evts)):
    for evt in range(num_evts):
      #if evt <100:        
  
        if daq_qual_cut_sum[evt]:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_p2p = True)
            p2p[ant, evt] = wf_int.int_p2p
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        rms[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
    del ara_root, num_evts, num_ants, wf_int

    clean_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 0)
    if np.count_nonzero(clean_idx) == 0:
        clean_idx = np.logical_and(daq_qual_cut_sum == 0, trig_type == 0)
        print(f'no clean rf events! all # of rf events: {np.count_nonzero(clean_idx)}')
    rms_copy = np.copy(rms)
    rms_copy[:, ~clean_idx] = np.nan
    rms_mean = np.nanmean(rms_copy, axis = 1)
    snr = p2p / 2 / rms_mean[:, np.newaxis]
    del rms_copy, daq_qual_cut_sum, pre_qual_cut_sum

    print('SNR collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins':time_bins,
            'sec_per_min':sec_per_min,
            'snr':snr,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean}









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
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year, use_l2 = True, run = ara_uproot.run, analyze_blind_dat = analyze_blind_dat, irs_block_number = ara_uproot.irs_block_number)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:] 
    daq_qual_cut = qual_hf['daq_qual_cut_sum'][:] != 0
    tot_qual_cut = qual_hf['tot_qual_cut_sum'][:] != 0
    daq_qual_cut_sum = np.in1d(evt_num, qual_evt[daq_qual_cut]).astype(int)
    tot_qual_cut_sum = np.in1d(evt_num, qual_evt[tot_qual_cut]).astype(int)
    del ara_uproot, qual_dat, qual_hf, qual_evt, daq_qual_cut, tot_qual_cut, run_info

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_l2 = True)

    # output array  
    rms = np.full((num_ants, num_evts), np.nan, dtype = float)
    p2p = np.copy(rms)

    # loop over the events
    for evt in tqdm(range(num_evts)):
    #for evt in range(num_evts):
      #if evt <100:        
  
        if daq_qual_cut_sum[evt]:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_p2p = True)
            p2p[ant, evt] = wf_int.int_p2p
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        rms[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
    del ara_root, num_evts, num_ants, wf_int

    clean_idx = np.logical_and(tot_qual_cut_sum == 0, trig_type == 2)
    if np.count_nonzero(clean_idx) == 0:
        clean_idx = np.logical_and(daq_qual_cut_sum == 0, trig_type == 2)
        print(f'no clean soft events! all # of soft events: {np.count_nonzero(clean_idx)}')
    rms_copy = np.copy(rms)
    rms_copy[:, ~clean_idx] = np.nan
    rms_mean = np.nanmean(rms_copy, axis = 1)
    snr = p2p / 2 / rms_mean[:, np.newaxis]
    del rms_copy, daq_qual_cut_sum, tot_qual_cut_sum

    print('SNR collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'snr':snr,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean}









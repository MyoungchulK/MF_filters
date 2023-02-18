import numpy as np
from tqdm import tqdm
import h5py

def snr_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting snr starts!')
    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
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
    if use_l2:
        ara_root = ara_l2_loader(Data)
        ara_root.get_sub_info()
        num_evts = ara_root.num_evts
        evt_num = ara_root.evt_num
        trig_type = ara_root.trig_type
        st = ara_root.station_id
        run = ara_root.run
    else:
        ara_uproot = ara_uproot_loader(Data)
        ara_uproot.get_sub_info()
        evt_num = ara_uproot.evt_num
        num_evts = ara_uproot.num_evts
        trig_type = ara_uproot.get_trig_type()
        st = ara_uproot.station_id
        run = ara_uproot.run
        ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
        blk_len = (ara_uproot.read_win // 4) - 1
        del ara_uproot

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:] 
    daq_qual_cut = qual_hf['daq_qual_cut_sum'][:] != 0
    tot_qual_cut = qual_hf['tot_qual_cut_sum'][:] != 0
    daq_qual_cut_sum = np.in1d(evt_num, qual_evt[daq_qual_cut]).astype(int)
    tot_qual_cut_sum = np.in1d(evt_num, qual_evt[tot_qual_cut]).astype(int)
    del qual_dat, qual_hf, qual_evt, daq_qual_cut, tot_qual_cut, run_info

    for t in range(3):
        print(np.count_nonzero(trig_type == t))

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)
    del st, run

    # output array  
    rms = np.full((num_ants, num_evts), np.nan, dtype = float)
    p2p = np.copy(rms)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt <100:        

        if daq_qual_cut_sum[evt]:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        #ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True, use_p2p = True, evt = evt)
            p2p[ant, evt] = wf_int.int_p2p
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        rms[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
    del ara_root, num_ants, wf_int

    rms_qual = np.full((num_evts, 4), False, dtype = bool)
    rms_qual[:, 0] = np.logical_and(tot_qual_cut_sum == 0, trig_type == 2)
    rms_qual[:, 1] = np.logical_and(tot_qual_cut_sum == 0, trig_type == 0)
    rms_qual[:, 2] = np.logical_and(daq_qual_cut_sum == 0, trig_type == 2)
    rms_qual[:, 3] = np.logical_and(daq_qual_cut_sum == 0, trig_type == 0)

    clean_idx = np.full((num_evts), False, dtype = bool)
    for q in range(rms_qual.shape[1]):
        if np.count_nonzero(rms_qual[:, q]) > 0:
            clean_idx = rms_qual[:, q]
            break
    rms_qual = np.count_nonzero(rms_qual, axis = 0)
    print(f'rms quality: {rms_qual}')
    rms_copy = np.copy(rms)
    rms_copy[:, ~clean_idx] = np.nan
    rms_mean = np.nanmean(rms_copy, axis = 1)
    snr = p2p / 2 / rms_mean[:, np.newaxis]
    del num_evts, rms_copy, daq_qual_cut_sum

    print('SNR collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'clean_idx':clean_idx,
            'snr':snr,
            'p2p':p2p,
            'rms':rms,
            'rms_mean':rms_mean,
            'rms_qual':rms_qual,#}
            'tot_qual_cut_sum':tot_qual_cut_sum,
            'blk_len':blk_len}









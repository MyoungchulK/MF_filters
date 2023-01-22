import numpy as np
from tqdm import tqdm
import h5py

def baseline_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False):

    print('Collecting baseline starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_trigs = ara_const.TRIGGER_TYPE
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    del ara_uproot

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:]
    daq_qual_cut = np.in1d(evt_num, qual_evt[qual_hf['daq_qual_cut_sum'][:] != 0])
    tot_qual_cut = np.in1d(evt_num, qual_evt[qual_hf['tot_qual_cut_sum'][:] != 0])
    del st, run, run_info, qual_dat, qual_hf, qual_evt

    # clean events
    clean_idx = np.full(num_evts, 0, dtype = int)
    num_cleans = np.full((num_trigs), 0, dtype = int)
    for trig in range(num_trigs):
        cleans = np.logical_and(tot_qual_cut == 0, trig_type == trig)
        if np.count_nonzero(cleans) == 0:
            cleans = np.logical_and(daq_qual_cut == 0, trig_type == trig)
        num_cleans[trig] = np.count_nonzero(cleans)
        clean_idx += cleans.astype(int)
        del cleans
    print(f'number of clean events by trigger: {num_cleans}')
    del entry_num, daq_qual_cut, tot_qual_cut 

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)
    freq_range = wf_int.pad_zero_freq

    # output array  
    baseline = np.full((wf_int.pad_fft_len, num_ants, num_trigs), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 0:        
       
        # quality cut
        if clean_idx[evt] == 0:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        rffts = wf_int.pad_fft
        baseline[:, :, trig_type[evt]] += rffts
        del rffts
    del ara_root, num_evts, num_ants, num_trigs, wf_int

    baseline /= num_cleans.astype(float)[np.newaxis, np.newaxis, :]

    print('Baseline collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'clean_idx':clean_idx,
            'num_cleans':num_cleans, 
            'freq_range':freq_range,
            'baseline':baseline}









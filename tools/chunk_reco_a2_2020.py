import numpy as np
from tqdm import tqdm
import h5py

def reco_a2_2020_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting reco a2 2020 starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_py_interferometers import get_products
    from tools.ara_run_manager import run_info_loader
    from tools.ara_quality_cut import get_bad_events

    # data config
    if use_l2:
        ara_root = ara_l2_loader(Data)
        num_evts = ara_root.num_evts
        evt_num = ara_root.evt_num
        trig_type = ara_root.trig_type
        daq_qual_cut_sum = ara_root.daq_cut
        st = ara_root.station_id
        yr = ara_root.year
        run = ara_root.run
    else:
        ara_uproot = ara_uproot_loader(Data)
        trig_type = ara_uproot.get_trig_type()
        evt_num = ara_uproot.evt_num
        num_evts = ara_uproot.num_evts
        st = ara_uproot.station_id
        yr = ara_uproot.year
        run = ara_uproot.run
        ara_root = ara_root_loader(Data, Ped, st, yr)
        daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num)[0]
        del ara_uproot
    
    # snr info
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    wei_dat = run_info.get_result_path(file_type = 'snr', verbose = True)
    wei_hf = h5py.File(wei_dat, 'r')
    weights = wei_hf['snr'][:]
    del run_info, wei_dat, wei_hf

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # interferometers
    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, st, yr, run = run, get_sub_file = True)
    lags = ara_int.lags
    pairs = ara_int.pairs
    ants = np.array([6, 7], dtype = int)
    ch_idx = np.arange(len(pairs[:, 0]), dtype = int)[np.logical_and(pairs[:, 0] == ants[0], pairs[:, 1] == ants[1])][0]
    print(f'pair index for Ch {ants[0]} & Ch {ants[1]} is {ch_idx}!!!')
    wei_pairs = get_products(weights, pairs, ara_int.v_pairs_len)[ch_idx]
    del st, yr, run, pairs, weights

    # output array  
    coef = np.full((num_evts), np.nan, dtype = float)
    lag = np.copy(coef)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
        
        if daq_qual_cut_sum[evt] or trig_type[evt] != 1:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in ants:
            raw_t, raw_v = ara_root.get_rf_ch_wf(int(ant))
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        ara_int.get_padded_wf(wf_int.pad_v)
        ara_int.get_cross_correlation()
        corr = ara_int.corr[:, ch_idx] * wei_pairs[evt]
        try:
            max_idx = np.nanargmax(corr)
            coef[evt] = corr[max_idx]
            lag[evt] = lags[max_idx]
        except ValueError:
            pass
        del corr, max_idx
    del ara_root, num_evts, wf_int, ara_int, daq_qual_cut_sum, wei_pairs, lags, ch_idx, ants

    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'coef':coef,
            'lag':lag}










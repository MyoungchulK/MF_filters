import numpy as np
from tqdm import tqdm
import h5py

def l2_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_quality_cut import get_bad_events

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    evt_num = ara_uproot.evt_num
    num_evts = ara_uproot.num_evts
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    yr = ara_uproot.year
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, yr)
    ara_uproot.get_sub_info()
    read_win = np.nansum(ara_uproot.read_win // 4)
    read_win *= 40
    read_win *= num_ants
    read_10 = int(float(read_win) * 0.1)
    read_win += read_10 
    del ara_uproot, yr, read_10

    # pre quality cut
    daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, qual_type = 2)[0]

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)
    del st, run 

    wf_len = np.full((num_evts * num_ants), 0, dtype = int)
    wf_v = np.full((read_win), np.nan, dtype = float)
    wf_t = np.full((num_evts * num_ants), np.nan, dtype = float)
    print(f'filtered wfs size: ~{np.round(wf_v.nbytes/1024/1024)} MB')
    del read_win

    counts = 0
    bin_conunts = 0
    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
        
        if daq_qual_cut_sum[evt]:
            for ant in range(num_ants):
                counts += 1
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        pad_num = wf_int.pad_num
        pad_v = wf_int.pad_v
        pad_t = wf_int.pad_t
        for ant in range(num_ants):
            wf_len[counts] = pad_num[ant]
            wf_t[counts] = pad_t[0, ant]
            wf_v[bin_conunts:bin_conunts + pad_num[ant]] = pad_v[:pad_num[ant], ant]
            bin_conunts += pad_num[ant] 
            counts += 1
        del pad_num, pad_v, pad_t
    del num_ants, num_evts, ara_root, daq_qual_cut_sum, wf_int, counts 

    wf_v = wf_v[:bin_conunts]

    print('l2 collecting is done!')

    return {'evt_num':evt_num, 
            'trig_type':trig_type,
            'wf_len':wf_len,
            'wf_t':wf_t,
            'wf_v':wf_v}









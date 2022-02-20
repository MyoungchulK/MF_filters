import numpy as np
from tqdm import tqdm

def reco_collector(Data, Ped):

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    from tools.ara_quality_cut import pre_qual_cut_loader
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, pre_qual_cut_temp

    clean_evt_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 1)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del pre_qual_cut_sum

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # interferometers
    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, ara_uproot.station_id, ara_uproot.year, ara_uproot.run)

    # output arr
    corr_v = []
    corr_h = []

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if clean_evt_idx[evt] == False:
            continue 

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_time_pad = True, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   
 
        corr_v_evt, corr_h_evt = ara_int.get_sky_map(wf_int.pad_v)
        corr_v.append(corr_v_evt)
        corr_h.append(corr_h_evt)
    del ara_root, ara_uproot, num_evts, num_ants, clean_evt_idx, wf_int, ara_int

    corr_v = np.asarray(corr_v)
    corr_h = np.asarray(corr_h)

    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'corr_v':corr_v,
            'corr_h':corr_h}









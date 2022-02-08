import numpy as np
from tqdm import tqdm

def dupl_temp_collector(Data, Ped):

    print('dupl check starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const 

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    from tools.ara_data_load import analog_buffer_info_loader
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)

    evt_n = np.full((1), 0, dtype = int)
    trig = np.copy(evt_n)
    unix = np.copy(evt_n)
    unix[0] = ara_uproot.unix_time[0]
    wf_all = np.full((3200, 2, 16, 1), np.nan, dtype = float)
    wf_all_time = np.copy(wf_all)
    wf_all_calib = np.copy(wf_all)

    samp_idx = np.full((64, 50, num_ants, 1), np.nan, dtype=float)
    time_arr = np.copy(samp_idx)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:

        if trig_type[evt] != 1:
            continue
        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        if blk_idx_arr[0] != 138 and blk_idx_arr[-1] != 164:
            continue
        print(blk_idx_arr)

        evt_n[0] = evt_num[evt]
        trig[0] = trig_type[evt]
        print(evt_n[0], trig[0])

        samp_idx[:, :len(blk_idx_arr), :, 0] = buffer_info.get_samp_idx(blk_idx_arr)
        time_arr[:, :len(blk_idx_arr), :, 0] = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True)

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kNoCalib)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            raw_len = len(raw_t)
            wf_all[:raw_len, 0, ant, 0] = raw_t
            wf_all[:raw_len, 1, ant, 0] = raw_v

            del raw_t, raw_v, raw_len
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
       
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            raw_len = len(raw_t)
            wf_all_time[:raw_len, 0, ant, 0] = raw_t
            wf_all_time[:raw_len, 1, ant, 0] = raw_v

            del raw_t, raw_v, raw_len
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            raw_len = len(raw_t)
            wf_all_calib[:raw_len, 0, ant, 0] = raw_t
            wf_all_calib[:raw_len, 1, ant, 0] = raw_v

            del raw_t, raw_v, raw_len
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

 
        break    

    del ara_root, num_evts, num_ants

    print('dupl check is done!')

    return {'evt_num':evt_n,
            'trig_type':trig,
            'unix_time':unix,
            'wf_all':wf_all,
            'wf_all_time':wf_all_time,
            'wf_all_calib':wf_all_calib,
            'samp_idx':samp_idx,
            'time_arr':time_arr}





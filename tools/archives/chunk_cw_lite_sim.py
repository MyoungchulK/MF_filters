import numpy as np
from tqdm import tqdm

def cw_lite_sim_collector(Data, Station, Year):

    print('Collecting cw starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_data_load import sin_subtract_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    evt_num = np.arange(num_evts, dtype = int)
    ara_root.get_sub_info(Data)
    dt = ara_root.time_step

    # wf analyzer
    num_params = 3 # 04, 025, 0125
    cw_freq = np.full((num_params, 2), np.nan, dtype = float)
    cw_freq[0, 0] = 0.125
    cw_freq[0, 1] = 0.15
    cw_freq[1, 0] = 0.24
    cw_freq[1, 1] = 0.26
    cw_freq[2, 0] = 0.395
    cw_freq[2, 1] = 0.415
    cw_thres = np.full((num_params, num_ants), 0.02, dtype = float)
    sol_pad = 200
    sin_sub = sin_subtract_loader(cw_freq, cw_thres, 3, num_params, dt, sol_pad)
    wf_int = wf_analyzer(dt = dt, use_time_pad = True)
    del cw_freq, cw_thres 

    # output
    sub_ratio = np.full((sol_pad, num_params, num_ants, num_evts), np.nan, dtype = float)
    sub_power = np.copy(sub_ratio)
    del num_params, sol_pad

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt == 100:        

        # get entry and wf
        ara_root.get_entry(evt)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            int_v, int_num = wf_int.get_int_wf(raw_t, raw_v, ant, use_unpad = True)[1:]           
            sin_sub.get_sin_subtract_wf(int_v, int_num, ant, return_none = True)
            sub_ratio[:, :, ant, evt] = sin_sub.sub_ratios
            sub_power[:, :, ant, evt] = sin_sub.sub_powers
            del raw_t, raw_v, int_v, int_num
            ara_root.del_TGraph()
    del ara_root, num_evts, wf_int, sin_sub

    sub_sum = np.nansum(sub_power, axis = 0)
    sub_weight = sub_power / sub_sum[np.newaxis, :, :]
    del sub_sum

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'sub_ratio':sub_ratio,
            'sub_power':sub_power,
            'sub_weight':sub_weight}



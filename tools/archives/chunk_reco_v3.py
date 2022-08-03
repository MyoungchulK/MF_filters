import numpy as np
from tqdm import tqdm

def reco_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers

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
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, add_double_pad = True)

    # interferometers
    ara_int = py_interferometers(41, 0, wf_int.pad_len, wf_int.dt, ara_uproot.station_id, ara_uproot.year, ara_uproot.run)
    del ara_uproot

    # output array  
    snr = np.full((num_ants, num_evts), np.nan, dtype = float)
    coef = np.full((2, 2, 2, num_evts), np.nan, dtype = float) # pol, rad, sol
    coord = np.full((2, 2, 2, 2, num_evts), np.nan, dtype = float) # thephi, pol, rad, sol

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
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
 
        coef[:, :, :, evt], coord[:, :, :, :, evt] = ara_int.get_sky_map(wf_int.pad_v)
    del ara_root, ara_uproot, num_evts, num_ants, wf_int, ara_int

    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'snr':snr,
            'coef':coef,
            'coord':coord}









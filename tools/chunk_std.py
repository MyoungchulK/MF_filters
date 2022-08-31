import numpy as np
from tqdm import tqdm
import h5py

def std_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_constant import ara_const
    
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    time_bins, sec_per_min = ara_uproot.get_event_rate(use_time_bins = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # qulity cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    qual_hf = h5py.File(qual_dat, 'r')
    daq_qual_sum = qual_hf['daq_qual_cut_sum'][:]
    del run_info, qual_dat, qual_hf, ara_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # output array
    std = np.full((num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    #for evt in tqdm(range(num_evts)):
    for evt in range(num_evts):
      #if evt <100:        
   
        if daq_qual_sum[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        std[:, evt] = np.nanstd(wf_int.pad_v, axis = 0)
    del ara_root, num_evts, daq_qual_sum, num_ants, wf_int

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins':time_bins,
            'sec_per_min':sec_per_min,
            'std':std}




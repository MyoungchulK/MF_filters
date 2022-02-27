import numpy as np
from tqdm import tqdm

def rayl_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting freq starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_Hk_uproot_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist = ara_Hk_uproot.get_sensor_hist()
    del run_info, Data, ara_Hk_uproot

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_entry = entry_num[clean_evt_idx]
    clean_evt = evt_num[clean_evt_idx]
    clean_len = len(clean_evt)   
    print(f'Number of clean event is {clean_len}') 
    del qual_cut_sum, clean_evt_idx, ara_qual, entry_num

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)

    # output arr
    freq = wf_int.pad_zero_freq
    print(len(freq))

    # loop over the events
    for evt in tqdm(range(clean_len)):
      #if evt <100:        
   
        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        wf_int.pad_fft       
 
  
    del ara_root, ara_uproot, num_evts, wf_int

    print('Freq collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'total_qual_cut':total_qual_cut,
            'freq':freq}








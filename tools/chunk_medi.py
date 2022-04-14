import numpy as np
from tqdm import tqdm
import h5py

def medi_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting median starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_run_manager import run_info_loader

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
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}')
    del qual_cut_sum, ara_qual

    # sensor info
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = True)
    sensor_dat = run_info.get_result_path(file_type = 'sensor', verbose = True)
    sensor_hf = h5py.File(sensor_dat, 'r')
    sensor_unix = sensor_hf['unix_time'][:]
    dda_temp = sensor_hf['dda_temp'][:]
    tda_temp = sensor_hf['tda_temp'][:]
    print(dda_temp.shape)
    del run_info, sensor_dat, sensor_hf, ara_uproot

    # output arr
    adc_medi = np.full((num_ants, num_evts), np.nan, dtype = float)
    sub_medi = np.copy(adc_medi)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            adc_medi[ant, evt] = np.nanmedian(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_root.get_useful_evt(ara_root.cal_type.kJustPedWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            sub_medi[ant, evt] = np.nanmedian(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 
    del ara_root, num_evts, daq_qual_sum, num_ants

    #std
    adc_std = np.nanstd(adc_medi, axis = 1)
    adc_rf_copy = np.copy(adc_medi)
    adc_rf_copy[:, trig_type != 0] = np.nan
    adc_rf_std = np.nanstd(adc_rf_copy, axis = 1)
    adc_rf_cut_copy = np.copy(adc_medi)
    adc_rf_cut_copy[:, ~clean_evt_idx] = np.nan
    adc_rf_cut_std = np.nanstd(adc_rf_cut_copy, axis = 1)

    sub_std = np.nanstd(sub_medi, axis = 1)
    sub_rf_copy = np.copy(sub_medi)
    sub_rf_copy[:, trig_type != 0] = np.nan
    sub_rf_std = np.nanstd(sub_rf_copy, axis = 1)
    sub_rf_cut_copy = np.copy(sub_medi)
    sub_rf_cut_copy[:, ~clean_evt_idx] = np.nan
    sub_rf_cut_std = np.nanstd(sub_rf_cut_copy, axis = 1)

    dda_std = np.nanstd(dda_temp, axis = 0)
    tda_std = np.nanstd(tda_temp, axis = 0)

    # first/end diff
    min_2nd_idx = np.logical_and(unix_time > unix_time[0] + 59, unix_time < unix_time[0] + 180)
    min_last_idx = unix_time > unix_time[-1] - 120

    def get_sub_diff(dat, min_2nd_idx, min_last_idx):
        sub_2nd_min = np.copy(dat)
        sub_last_min = np.copy(dat)
        sub_2nd_min[:, ~min_2nd_idx] = np.nan
        sub_last_min[:, ~min_last_idx] = np.nan
        sub_2nd_min_medi = np.nanmedian(sub_2nd_min, axis = 1)
        sub_last_min_medi = np.nanmedian(sub_last_min, axis = 1)
        sub_diff = np.abs(sub_2nd_min_medi - sub_last_min_medi)
        del sub_2nd_min, sub_last_min, sub_2nd_min_medi, sub_last_min_medi
        return sub_diff

    adc_diff = get_sub_diff(adc_medi, min_2nd_idx, min_last_idx)
    adc_rf_diff = get_sub_diff(adc_rf_copy, min_2nd_idx, min_last_idx)
    adc_rf_cut_diff = get_sub_diff(adc_rf_cut_copy, min_2nd_idx, min_last_idx)
    del adc_rf_copy, adc_rf_cut_copy

    sub_diff = get_sub_diff(sub_medi, min_2nd_idx, min_last_idx)
    sub_rf_diff = get_sub_diff(sub_rf_copy, min_2nd_idx, min_last_idx)
    sub_rf_cut_diff = get_sub_diff(sub_rf_cut_copy, min_2nd_idx, min_last_idx)
    del sub_rf_copy, sub_rf_cut_copy, min_2nd_idx, min_last_idx

    # first/end diff
    min_2nd_idx = np.logical_and(sensor_unix > sensor_unix[0] + 59, sensor_unix < sensor_unix[0] + 180)
    min_last_idx = sensor_unix > sensor_unix[-1] - 120

    def get_sub_diff(dat, min_2nd_idx, min_last_idx):
        sub_2nd_min = np.copy(dat)
        sub_last_min = np.copy(dat)
        sub_2nd_min[~min_2nd_idx] = np.nan
        sub_last_min[~min_last_idx] = np.nan
        sub_2nd_min_medi = np.nanmedian(sub_2nd_min, axis = 0)
        sub_last_min_medi = np.nanmedian(sub_last_min, axis = 0)
        sub_diff = np.abs(sub_2nd_min_medi - sub_last_min_medi)
        del sub_2nd_min, sub_last_min, sub_2nd_min_medi, sub_last_min_medi
        return sub_diff

    dda_diff = get_sub_diff(dda_temp, min_2nd_idx, min_last_idx)
    tda_diff = get_sub_diff(tda_temp, min_2nd_idx, min_last_idx)
    del min_2nd_idx, min_last_idx  
 
    print('Median collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'sensor_unix':sensor_unix,
            'dda_temp':dda_temp,
            'tda_temp':tda_temp,
            'adc_medi':adc_medi,
            'sub_medi':sub_medi,
            'adc_std':adc_std,
            'adc_rf_std':adc_rf_std,
            'adc_rf_cut_std':adc_rf_cut_std,
            'sub_std':sub_std,
            'sub_rf_std':sub_rf_std,
            'sub_rf_cut_std':sub_rf_cut_std,
            'dda_std':dda_std,
            'tda_std':tda_std,
            'adc_diff':adc_diff,
            'adc_rf_diff':adc_rf_diff,
            'adc_rf_cut_diff':adc_rf_cut_diff,
            'sub_diff':sub_diff,
            'sub_rf_diff':sub_rf_diff,
            'sub_rf_cut_diff':sub_rf_cut_diff,
            'dda_diff':dda_diff,
            'tda_diff':tda_diff}



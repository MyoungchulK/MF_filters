import numpy as np
from tqdm import tqdm

def std_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import pre_qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts

    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    pre_qual_cut_temp[:, -2] = 0
    pre_qual_cut_temp[:, -3] = 0
    pre_qual_cut_sum_wo_bias = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, pre_qual_cut_temp

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_Hk_uproot_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = False)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist = ara_Hk_uproot.get_sensor_hist()
    del run_info, Data, ara_Hk_uproot

    clean_evt_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del pre_qual_cut_sum
    wo_bias_rf_evt_idx = np.logical_and(pre_qual_cut_sum_wo_bias == 0, trig_type == 0)
    del pre_qual_cut_sum_wo_bias

    # output array
    std = np.full((num_eles, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
    
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # loop over the antennas
        for ant in range(num_eles):
            raw_v = ara_root.get_ele_ch_wf(ant)[1]
            std[ant, evt] = np.nanstd(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_root, ara_uproot, num_evts

    std_rf = np.copy(std)
    std_rf[:, trig_type != 0] = np.nan
    std_rf_wo_bias_cut = np.copy(std)
    std_rf_wo_bias_cut[:, ~wo_bias_rf_evt_idx] = np.nan
    std_rf_w_cut = np.copy(std)
    std_rf_w_cut[:, ~clean_evt_idx] = np.nan
    del clean_evt_idx, wo_bias_rf_evt_idx

    std_range = np.arange(0,100,0.1)
    std_bins = np.linspace(0, 100, 1000 + 1)
    std_hist = np.full((num_eles, len(std_range)), 0, dtype = int)
    std_rf_hist = np.copy(std_hist)
    std_rf_wo_bias_cut_hist = np.copy(std_hist)
    std_rf_w_cut_hist = np.copy(std_hist)
    for ant in tqdm(range(num_eles)):
        std_hist[ant] = np.histogram(std[ant], bins = std_bins)[0].astype(int)
        std_rf_hist[ant] = np.histogram(std_rf[ant], bins = std_bins)[0].astype(int)
        std_rf_wo_bias_cut_hist[ant] = np.histogram(std_rf_wo_bias_cut[ant], bins = std_bins)[0].astype(int)
        std_rf_w_cut_hist[ant] = np.histogram(std_rf_w_cut[ant], bins = std_bins)[0].astype(int)
    del num_eles

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'clean_evt':clean_evt,
            'std':std,
            'std_rf':std_rf,
            'std_rf_wo_bias_cut':std_rf_wo_bias_cut,
            'std_range':std_range,
            'std_bins':std_bins,
            'std_hist':std_hist,
            'std_rf_hist':std_rf_hist,
            'std_rf_wo_bias_cut_hist':std_rf_wo_bias_cut_hist,
            'std_rf_w_cut_hist':std_rf_w_cut_hist,
            'std_rf_w_cut':std_rf_w_cut, 
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}





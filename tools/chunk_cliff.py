import numpy as np
from tqdm import tqdm

def cliff_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts

    #output array
    cliff_adc = np.full((num_ants, num_evts), np.nan, dtype = float)
    cliff = np.copy(cliff_adc)
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    from tools.ara_quality_cut import pre_qual_cut_loader
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
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = False)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    if Data is None:
        print('There is no sensorHk file!')
        dda_volt = np.full((1,4), np.nan, dtype = float)
        dda_curr = np.copy(dda_volt)
        dda_temp = np.copy(dda_volt)
        tda_volt = np.copy(dda_volt)
        tda_curr = np.copy(dda_volt)
        tda_temp = np.copy(dda_volt)
        atri_volt = np.full((1), np.nan, dtype = float)
        atri_curr = np.copy(atri_volt)
    else:
        from tools.ara_data_load import ara_Hk_uproot_loader
        ara_Hk_uproot = ara_Hk_uproot_loader(Data)
        if ara_Hk_uproot.empty_file_error == True:
            print('There is empty sensorHk file!')
            dda_volt = np.full((1,4), np.nan, dtype = float)
            dda_curr = np.copy(dda_volt)
            dda_temp = np.copy(dda_volt)
            tda_volt = np.copy(dda_volt)
            tda_curr = np.copy(dda_volt)
            tda_temp = np.copy(dda_volt)
            atri_volt = np.full((1), np.nan, dtype = float)
            atri_curr = np.copy(atri_volt)
        else:
            ara_Hk_uproot.get_sub_info()
            atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_Hk_uproot.get_daq_sensor_info()
        del ara_Hk_uproot
    del run_info


    if ara_uproot.station_id == 2:
        dda_volt_range = np.arange(3, 3.5, 0.005)
        dda_volt_bins = np.linspace(3, 3.5, 100+1)
        dda_curr_range = np.arange(0, 0.5, 0.005)
        dda_curr_bins = np.linspace(0, 0.5, 100+1)
        tda_volt_range = np.arange(3, 3.5, 0.005)
        tda_volt_bins = np.linspace(3, 3.5, 100+1)
        tda_curr_range = np.arange(0.16, 0.21, 0.0005)
        tda_curr_bins = np.linspace(0.16, 0.21, 100+1)
        temp_range = np.arange(-20,20,0.4)
        temp_bins = np.linspace(-20,20,100+1)
        atri_range = np.arange(0, 4.5, 0.045)
        atri_bins = np.linspace(0, 4.5, 100+1) 
    else:
        dda_volt_range = np.arange(0,10,0.1)
        dda_volt_bins = np.linspace(0, 10, 100+1)
        dda_curr_range = np.arange(0,1,0.01)
        dda_curr_bins = np.linspace(0, 1, 100+1)
        tda_volt_range = np.arange(0,5,0.05)
        tda_volt_bins = np.linspace(0, 5, 100+1)
        tda_curr_range = np.arange(0,0.3,0.003)
        tda_curr_bins = np.linspace(0, 0.3, 100+1)
        temp_range = np.arange(-20,20,0.4)
        temp_bins = np.linspace(-20,20,100+1)
        atri_range = np.arange(0,15,0.15)
        atri_bins = np.linspace(0,15,100+1)   
 
    dda_volt_hist = np.full((len(dda_volt_range), 4), 0, dtype = int)
    dda_curr_hist = np.copy(dda_volt_hist)
    tda_volt_hist = np.copy(dda_volt_hist)
    tda_curr_hist = np.copy(dda_volt_hist)
    dda_temp_hist = np.copy(dda_volt_hist)
    tda_temp_hist = np.copy(dda_volt_hist)
    for d in range(4):
        dda_volt_hist[:,d] = np.histogram(dda_volt[:,d], bins = dda_volt_bins)[0].astype(int)    
        dda_curr_hist[:,d] = np.histogram(dda_curr[:,d], bins = dda_curr_bins)[0].astype(int)    
        dda_temp_hist[:,d] = np.histogram(dda_temp[:,d], bins = temp_bins)[0].astype(int)    
        tda_volt_hist[:,d] = np.histogram(tda_volt[:,d], bins = tda_volt_bins)[0].astype(int)    
        tda_curr_hist[:,d] = np.histogram(tda_curr[:,d], bins = tda_curr_bins)[0].astype(int)    
        tda_temp_hist[:,d] = np.histogram(tda_temp[:,d], bins = temp_bins)[0].astype(int)    
    atri_volt_hist = np.histogram(atri_volt, bins = atri_bins)[0].astype(int)
    atri_curr_hist = np.histogram(atri_curr, bins = atri_bins)[0].astype(int)
    del atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp

    clean_rf_evt_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 0)
    clean_rf_evt = evt_num[clean_rf_evt_idx]   
    print(f'Number of clean event is {len(clean_rf_evt)}') 
    del pre_qual_cut_sum
    wo_bias_rf_evt_idx = np.logical_and(pre_qual_cut_sum_wo_bias == 0, trig_type == 0)
    del pre_qual_cut_sum_wo_bias

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
    
        # sample index
        blk_idx_arr, blk_len_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)
        if blk_len_arr == 0:
            cliff_adc[:, evt] = 0
            cliff[:, evt] = 0
            del blk_idx_arr, blk_len_arr
            continue
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        samp_in_blk = buffer_info.samp_in_blk
        del blk_idx_arr, blk_len_arr

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            cliff_adc[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
        del samp_in_blk
    del ara_root, ara_uproot, buffer_info, num_evts

    print(np.nanmax(cliff_adc),np.nanmin(cliff_adc))
    print(np.nanmax(cliff),np.nanmin(cliff))

    cliff_adc_rf = np.copy(cliff_adc)
    cliff_adc_rf[:, trig_type != 0] = np.nan
    cliff_adc_rf_wo_bias_cut = np.copy(cliff_adc)
    cliff_adc_rf_wo_bias_cut[:, ~wo_bias_rf_evt_idx] = np.nan
    cliff_adc_rf_w_cut = np.copy(cliff_adc)
    cliff_adc_rf_w_cut[:, ~clean_rf_evt_idx] = np.nan

    cliff_rf = np.copy(cliff)
    cliff_rf[:, trig_type != 0] = np.nan
    cliff_rf_wo_bias_cut = np.copy(cliff)
    cliff_rf_wo_bias_cut[:, ~wo_bias_rf_evt_idx] = np.nan
    cliff_rf_w_cut = np.copy(cliff)
    cliff_rf_w_cut[:, ~clean_rf_evt_idx] = np.nan
    del clean_rf_evt_idx, wo_bias_rf_evt_idx

    cliff_range = np.arange(-2500, 2500, 5)
    cliff_bins = np.linspace(-2500, 2500, 1000 + 1)

    cliff_adc_hist = np.full((num_ants, len(cliff_range)), 0, dtype = int)
    cliff_adc_rf_hist = np.copy(cliff_adc_hist)
    cliff_adc_rf_hist_wo_bias_cut = np.copy(cliff_adc_hist)
    cliff_adc_rf_hist_w_cut = np.copy(cliff_adc_hist)
    
    cliff_hist = np.copy(cliff_adc_hist)
    cliff_rf_hist = np.copy(cliff_adc_hist)
    cliff_rf_hist_wo_bias_cut = np.copy(cliff_adc_hist)
    cliff_rf_hist_w_cut = np.copy(cliff_adc_hist)
    for ant in tqdm(range(num_ants)):

        cliff_adc_hist[ant] = np.histogram(cliff_adc[ant], bins = cliff_bins)[0].astype(int)
        cliff_adc_rf_hist[ant] = np.histogram(cliff_adc_rf[ant], bins = cliff_bins)[0].astype(int)
        cliff_adc_rf_hist_wo_bias_cut[ant] = np.histogram(cliff_adc_rf_wo_bias_cut[ant], bins = cliff_bins)[0].astype(int)
        cliff_adc_rf_hist_w_cut[ant] = np.histogram(cliff_adc_rf_w_cut[ant], bins = cliff_bins)[0].astype(int)

        cliff_hist[ant] = np.histogram(cliff[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist[ant] = np.histogram(cliff_rf[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist_wo_bias_cut[ant] = np.histogram(cliff_rf_wo_bias_cut[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist_w_cut[ant] = np.histogram(cliff_rf_w_cut[ant], bins = cliff_bins)[0].astype(int)
    del num_ants

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'clean_rf_evt':clean_rf_evt,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,    
            'cliff_range':cliff_range,
            'cliff_adc':cliff_adc,
            'cliff_adc_hist':cliff_adc_hist,
            'cliff_adc_rf_hist':cliff_adc_rf_hist,
            'cliff_adc_rf_hist_wo_bias_cut':cliff_adc_rf_hist_wo_bias_cut,
            'cliff_adc_rf_hist_w_cut':cliff_adc_rf_hist_w_cut, 
            'cliff':cliff,
            'cliff_hist':cliff_hist,
            'cliff_rf_hist':cliff_rf_hist,
            'cliff_rf_hist_wo_bias_cut':cliff_rf_hist_wo_bias_cut,
            'cliff_rf_hist_w_cut':cliff_rf_hist_w_cut,
            'dda_volt_range':dda_volt_range,
            'dda_curr_range':dda_curr_range,
            'tda_volt_range':tda_volt_range,
            'tda_curr_range':tda_curr_range,
            'temp_range':temp_range,
            'atri_range':atri_range,
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}








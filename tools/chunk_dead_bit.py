import numpy as np
from tqdm import tqdm

def dead_bit_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts

    #output array
    cliff = np.full((num_ants, num_evts), np.nan, dtype = float)
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    dead_bit_hist = np.full((num_ants, num_bits), 0, dtype = int)
    dead_bit_rf_hist = np.copy(dead_bit_hist)
    dead_bit_rf_hist_wo_bias_cut = np.copy(dead_bit_hist)
    dead_bit_rf_hist_w_cut = np.copy(dead_bit_hist)
    dead_bit_bins = np.linspace(0, num_bits, num_bits+1)
    dead_bit_range = np.arange(num_bits)

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

    is_sensor = np.full((1), 1, dtype = int)

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
        is_sensor[:] = 0
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
            is_sensor[:] = 0
        else:
            ara_Hk_uproot.get_sub_info()
            atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_Hk_uproot.get_daq_sensor_info()
        del ara_Hk_uproot
    del run_info

    dda_volt_range = np.arange(0,10,0.1)
    dda_volt_bins = np.linspace(0, 10, 100+1)
    dda_volt_hist = np.full((len(dda_volt_range), 4), 0, dtype = int)
    dda_curr_range = np.arange(0,1,0.01)
    dda_curr_bins = np.linspace(0, 1, 100+1)
    dda_curr_hist = np.copy(dda_volt_hist)
    tda_volt_range = np.arange(0,5,0.05)
    tda_volt_bins = np.linspace(0, 5, 100+1)
    tda_volt_hist = np.copy(dda_volt_hist)
    tda_curr_range = np.arange(0,0.3,0.003)
    tda_curr_bins = np.linspace(0, 0.3, 100+1)
    tda_curr_hist = np.copy(dda_volt_hist)
    temp_range = np.arange(-20,20,0.4)
    temp_bins = np.linspace(-20,20,100+1)
    dda_temp_hist = np.copy(dda_volt_hist)
    tda_temp_hist = np.copy(dda_volt_hist)
    for d in range(4):
        dda_volt_hist[:,d] = np.histogram(dda_volt[:,d], bins = dda_volt_bins)[0].astype(int)    
        dda_curr_hist[:,d] = np.histogram(dda_curr[:,d], bins = dda_curr_bins)[0].astype(int)    
        dda_temp_hist[:,d] = np.histogram(dda_temp[:,d], bins = temp_bins)[0].astype(int)    
        tda_volt_hist[:,d] = np.histogram(tda_volt[:,d], bins = tda_volt_bins)[0].astype(int)    
        tda_curr_hist[:,d] = np.histogram(tda_curr[:,d], bins = tda_curr_bins)[0].astype(int)    
        tda_temp_hist[:,d] = np.histogram(tda_temp[:,d], bins = temp_bins)[0].astype(int)    
    atri_range = np.arange(0,15,0.15)
    atri_bins = np.linspace(0,15,100+1)
    atri_volt_hist = np.histogram(atri_volt, bins = atri_bins)[0].astype(int)
    atri_curr_hist = np.histogram(atri_curr, bins = atri_bins)[0].astype(int)
    del atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp

    trig_ratio = np.full((3), np.nan, dtype = float)
    trig_ratio[0] = np.count_nonzero(trig_type == 0)
    trig_ratio[1] = np.count_nonzero(trig_type == 1)
    trig_ratio[2] = np.count_nonzero(trig_type == 2)
    trig_ratio /= num_evts
    print(trig_ratio)

    bv_idx = pre_qual_cut[:,-3] == 0
    trig_type_wo_bv = trig_type[bv_idx]
    if np.any(bv_idx):
        num_evts_wo_bv = len(trig_type_wo_bv)
        trig_ratio_wo_bv = np.full((3), np.nan, dtype = float)
        trig_ratio_wo_bv[0] = np.count_nonzero(trig_type_wo_bv == 0)
        trig_ratio_wo_bv[1] = np.count_nonzero(trig_type_wo_bv == 1)
        trig_ratio_wo_bv[2] = np.count_nonzero(trig_type_wo_bv == 2)
        trig_ratio_wo_bv /= num_evts_wo_bv
        del num_evts_wo_bv
    else:
        trig_ratio_wo_bv = np.full((3), np.nan, dtype = float)
    del bv_idx
    print(trig_ratio_wo_bv)

    cal_idx = (pre_qual_cut[:,-3] + pre_qual_cut[:,-2]) == 0
    trig_type_wo_cal = trig_type[cal_idx]
    if np.any(cal_idx):
        num_evts_wo_cal = len(trig_type_wo_cal)
        trig_ratio_wo_cal = np.full((3), np.nan, dtype = float)
        trig_ratio_wo_cal[0] = np.count_nonzero(trig_type_wo_cal == 0)
        trig_ratio_wo_cal[1] = np.count_nonzero(trig_type_wo_cal == 1)
        trig_ratio_wo_cal[2] = np.count_nonzero(trig_type_wo_cal == 2)
        trig_ratio_wo_cal /= num_evts_wo_cal
        del num_evts_wo_cal
    else:
        trig_ratio_wo_cal = np.full((3), np.nan, dtype = float)
    del cal_idx
    print(trig_ratio_wo_cal)

    clean_rf_evt_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 0)
    clean_rf_evt = evt_num[clean_rf_evt_idx]   
    print(f'Number of clean event is {len(clean_rf_evt)}') 

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
    
        #if pre_qual_cut_sum[evt] != 0 or trig_type[evt] != 0:
        #    continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)

        # sample index
        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        samp_in_blk = buffer_info.samp_in_blk

        # loop over the antennas
        for ant in range(num_ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            if len(raw_v) == 0:
                cliff[ant, evt] = 0
                continue
            cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            dead_bit_hist_wf = np.histogram(raw_v, bins = dead_bit_bins)[0].astype(int)
            dead_bit_hist[ant] += dead_bit_hist_wf
            if trig_type[evt] == 0:
                dead_bit_rf_hist[ant] += dead_bit_hist_wf
            if pre_qual_cut_sum_wo_bias[evt] == 0 and trig_type[evt] == 0:    
                dead_bit_rf_hist_wo_bias_cut[ant] += dead_bit_hist_wf
            if pre_qual_cut_sum[evt] == 0 and trig_type[evt] == 0:
                dead_bit_rf_hist_w_cut[ant] += dead_bit_hist_wf
            del raw_v, dead_bit_hist_wf
            ara_root.del_TGraph()
        del blk_idx_arr, samp_in_blk
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, buffer_info, num_evts, dead_bit_bins, pre_qual_cut_sum

    wo_bias_rf_evt_idx = np.logical_and(pre_qual_cut_sum_wo_bias == 0, trig_type == 0)
    del pre_qual_cut_sum_wo_bias

    cliff_rf = np.copy(cliff)
    cliff_rf[:, trig_type != 0] = np.nan
    cliff_rf_wo_bias_cut = np.copy(cliff)
    cliff_rf_wo_bias_cut[:, ~wo_bias_rf_evt_idx] = np.nan
    cliff_rf_w_cut = np.copy(cliff)
    cliff_rf_w_cut[:, ~clean_rf_evt_idx] = np.nan
    del clean_rf_evt_idx, wo_bias_rf_evt_idx

    cliff_range = np.arange(-2500, 2500, 5)
    cliff_bins = np.linspace(-2500, 2500, 1000 + 1)

    cliff_hist = np.full((num_ants, len(cliff_range)), 0, dtype = int)
    cliff_rf_hist = np.copy(cliff_hist)
    cliff_rf_hist_wo_bias_cut = np.copy(cliff_hist)
    cliff_rf_hist_w_cut = np.copy(cliff_hist)
    for ant in tqdm(range(num_ants)):
        cliff_hist[ant] = np.histogram(cliff[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist[ant] = np.histogram(cliff_rf[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist_wo_bias_cut[ant] = np.histogram(cliff_rf_wo_bias_cut[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist_w_cut[ant] = np.histogram(cliff_rf_w_cut[ant], bins = cliff_bins)[0].astype(int)
    del num_ants, num_bits

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'clean_rf_evt':clean_rf_evt,
            'trig_type':trig_type,
            'trig_type_wo_bv':trig_type_wo_bv,
            'trig_type_wo_cal':trig_type_wo_cal,
            'trig_ratio':trig_ratio,
            'trig_ratio_wo_bv':trig_ratio_wo_bv,
            'trig_ratio_wo_cal':trig_ratio_wo_cal,
            'is_sensor':is_sensor,
            'cliff':cliff,
            'cliff_rf':cliff_rf,
            'cliff_rf_wo_bias_cut':cliff_rf_wo_bias_cut,
            'cliff_rf_w_cut':cliff_rf_w_cut,
            'cliff_range':cliff_range,
            'cliff_hist':cliff_hist,
            'cliff_rf_hist':cliff_rf_hist,
            'cliff_rf_hist_wo_bias_cut':cliff_rf_hist_wo_bias_cut,
            'cliff_rf_hist_w_cut':cliff_rf_hist_w_cut,
            'dead_bit_range':dead_bit_range,
            'dead_bit_hist':dead_bit_hist,
            'dead_bit_rf_hist':dead_bit_rf_hist,
            'dead_bit_rf_hist_wo_bias_cut':dead_bit_rf_hist_wo_bias_cut,
            'dead_bit_rf_hist_w_cut':dead_bit_rf_hist_w_cut,
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
            'atri_curr_hist':atri_curr_hist,
            'pre_qual_cut':pre_qual_cut}








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
    num_bits = ara_const.BUFFER_BIT_RANGE
    del ara_const

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

    from tools.ara_quality_cut import pre_qual_cut_loader
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, pre_qual_cut_temp

    is_sensor = np.full((1), 1, dtype = int)

    from tools.ara_run_manager import run_info_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = False)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    if Data is None:
        print('There is no sensorHk file!')
        dda_volt = np.full((1,4), np.nan, dtype = float)
        tda_volt = np.copy(dda_volt)
        atri_volt = np.full((1), np.nan, dtype = float)
        atri_curr = np.copy(atri_volt)
        is_sensor[:] = 0
    else:
        from tools.ara_data_load import ara_Hk_uproot_loader
        ara_Hk_uproot = ara_Hk_uproot_loader(Data)
        ara_Hk_uproot.get_sub_info()
        dda_volt_curr = ara_Hk_uproot.dda_volt_curr
        dda_volt = ara_Hk_uproot.get_voltage(dda_volt_curr)
        tda_volt_curr = ara_Hk_uproot.tda_volt_curr
        tda_volt = ara_Hk_uproot.get_voltage(tda_volt_curr)
        atri_volt, atri_curr = ara_Hk_uproot.get_atri_voltage_current()
        del ara_Hk_uproot, dda_volt_curr, tda_volt_curr
    del run_info

    dda_range = np.arange(0,20,0.01)
    dda_bins = np.linspace(0, 20, 2000+1)
    dda_volt_hist = np.full((len(dda_range), 4), 0, dtype = int)
    tda_volt_hist = np.full((len(dda_range), 4), 0, dtype = int)
    for d in range(4):
        dda_volt_hist[:,d] = np.histogram(dda_volt[:,d], bins = dda_bins)[0].astype(int)    
        tda_volt_hist[:,d] = np.histogram(tda_volt[:,d], bins = dda_bins)[0].astype(int)    
    atri_volt_hist = np.histogram(atri_volt, bins = dda_bins)[0].astype(int)
    atri_curr_hist = np.histogram(atri_curr, bins = dda_bins)[0].astype(int)
    del dda_volt, tda_volt, dda_bins, atri_volt, atri_curr

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
      #if evt < 100:        
    
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
            else:
                cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            del raw_v
            ara_root.del_TGraph()
        del blk_idx_arr, samp_in_blk
        ara_root.del_usefulEvt()
    del ara_root, ara_uproot, buffer_info, num_evts, pre_qual_cut_sum

    cliff_rf = cliff[:, trig_type == 0]
    cliff_rf_w_cut = cliff[:, clean_rf_evt_idx]
    del clean_rf_evt_idx

    cliff_range = np.arange(-num_bits, num_bits)
    cliff_bins = np.linspace(-num_bits, num_bits, num_bits*2 + 1)

    cliff_hist = np.full((num_ants, num_bits*2), 0, dtype = int)
    cliff_rf_hist = np.copy(cliff_hist)
    cliff_rf_hist_w_cut = np.copy(cliff_hist)
    for ant in tqdm(range(num_ants)):
        cliff_hist[ant] = np.histogram(cliff[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist[ant] = np.histogram(cliff_rf[ant], bins = cliff_bins)[0].astype(int)
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
            'cliff_rf_w_cut':cliff_rf_w_cut,
            'cliff_range':cliff_range,
            'cliff_bins':cliff_bins,
            'cliff_hist':cliff_hist,
            'cliff_rf_hist':cliff_rf_hist,
            'cliff_rf_hist_w_cut':cliff_rf_hist_w_cut,
            'dda_range':dda_range,
            'dda_volt_hist':dda_volt_hist,
            'tda_volt_hist':tda_volt_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist,
            'pre_qual_cut':pre_qual_cut}








import numpy as np
from tqdm import tqdm

def std_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    ara_geom = ara_geom_loader(ara_uproot.station_id, ara_uproot.year, verbose = True)
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    ele_ch = ara_geom.get_ele_ch_idx()
    del ara_geom

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_Hk_uproot_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist = ara_Hk_uproot.get_sensor_hist()
    del run_info, Data, ara_Hk_uproot

    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    del ara_qual

    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum

    # output array
    std = np.full((num_eles, num_evts), np.nan, dtype = float)
    cliff_adc = np.full((num_ants, num_evts), np.nan, dtype = float)
    cliff = np.copy(cliff_adc)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # sample index
        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        samp_in_blk = buffer_info.samp_in_blk
        del blk_idx_arr 

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
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            del raw_v
            ara_root.del_TGraph()

        # loop over the antennas
        for ant in range(num_eles):
            raw_v = ara_root.get_ele_ch_wf(ant)[1]
            std[ant, evt] = np.nanstd(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_root, ara_uproot, buffer_info, num_evts, daq_qual_sum

    std_rf = np.copy(std)
    std_rf[:, trig_type != 0] = np.nan
    std_rf_w_cut = np.copy(std)
    std_rf_w_cut[:, ~clean_evt_idx] = np.nan

    std_range = np.arange(0,100,0.1)
    std_bins = np.linspace(0, 100, 1000 + 1)
    std_hist = np.full((num_eles, len(std_range)), 0, dtype = int)
    std_rf_hist = np.copy(std_hist)
    std_rf_w_cut_hist = np.copy(std_hist)
    for ant in range(num_eles):
        std_hist[ant] = np.histogram(std[ant], bins = std_bins)[0].astype(int)
        std_rf_hist[ant] = np.histogram(std_rf[ant], bins = std_bins)[0].astype(int)
        std_rf_w_cut_hist[ant] = np.histogram(std_rf_w_cut[ant], bins = std_bins)[0].astype(int)
    del num_eles, std_rf, std_rf_w_cut

    cliff_adc_rf = np.copy(cliff_adc)
    cliff_adc_rf[:, trig_type != 0] = np.nan
    cliff_adc_rf_w_cut = np.copy(cliff_adc)
    cliff_adc_rf_w_cut[:, ~clean_evt_idx] = np.nan
    cliff_rf = np.copy(cliff)
    cliff_rf[:, trig_type != 0] = np.nan
    cliff_rf_w_cut = np.copy(cliff)
    cliff_rf_w_cut[:, ~clean_evt_idx] = np.nan
    del clean_evt_idx

    cliff_range = np.arange(-2500, 2500, 5)
    cliff_bins = np.linspace(-2500, 2500, 1000 + 1)
    cliff_adc_hist = np.full((num_ants, len(cliff_range)), 0, dtype = int)
    cliff_adc_rf_hist = np.copy(cliff_adc_hist)
    cliff_adc_rf_hist_w_cut = np.copy(cliff_adc_hist)
    cliff_hist = np.copy(cliff_adc_hist)
    cliff_rf_hist = np.copy(cliff_adc_hist)
    cliff_rf_hist_w_cut = np.copy(cliff_adc_hist)
    for ant in range(num_ants):
        cliff_adc_hist[ant] = np.histogram(cliff_adc[ant], bins = cliff_bins)[0].astype(int)
        cliff_adc_rf_hist[ant] = np.histogram(cliff_adc_rf[ant], bins = cliff_bins)[0].astype(int)
        cliff_adc_rf_hist_w_cut[ant] = np.histogram(cliff_adc_rf_w_cut[ant], bins = cliff_bins)[0].astype(int)
        cliff_hist[ant] = np.histogram(cliff[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist[ant] = np.histogram(cliff_rf[ant], bins = cliff_bins)[0].astype(int)
        cliff_rf_hist_w_cut[ant] = np.histogram(cliff_rf_w_cut[ant], bins = cliff_bins)[0].astype(int)
    del num_ants, cliff_adc_rf, cliff_adc_rf_w_cut, cliff_rf, cliff_rf_w_cut

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'ele_ch':ele_ch,
            'total_qual_cut':total_qual_cut,
            'clean_evt':clean_evt,
            'std':std,
            'std_range':std_range,
            'std_bins':std_bins,
            'std_hist':std_hist,
            'std_rf_hist':std_rf_hist,
            'std_rf_w_cut_hist':std_rf_w_cut_hist,
            'cliff_adc':cliff_adc,
            'cliff':cliff,
            'cliff_range':cliff_range,
            'cliff_bins':cliff_bins,
            'cliff_adc_hist':cliff_adc_hist,
            'cliff_adc_rf_hist':cliff_adc_rf_hist,
            'cliff_adc_rf_hist_w_cut':cliff_adc_rf_hist_w_cut,
            'cliff_hist':cliff_hist,
            'cliff_rf_hist':cliff_rf_hist,
            'cliff_rf_hist_w_cut':cliff_rf_hist_w_cut,
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}





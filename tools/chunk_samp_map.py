import numpy as np
from tqdm import tqdm
import h5py

def samp_map_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import sample_map_loader
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    trig_type = ara_uproot.get_trig_type()
    
    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = ara_qual.tot_cut_sum
    daq_qual_sum = ara_qual.daq_cut_sum
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}')
    del qual_cut_sum, ara_qual

    # sensor info
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = True)    
    sensor_dat = run_info.get_result_path(file_type = 'sensor', verbose = True)
    sensor_hf = h5py.File(sensor_dat, 'r')
    sensor_unix = sensor_hf['unix_time'][:]
    dda_volt = sensor_hf['dda_volt'][:]
    dda_curr = sensor_hf['dda_curr'][:]
    dda_temp = sensor_hf['dda_temp'][:]
    tda_volt = sensor_hf['tda_volt'][:]
    tda_curr = sensor_hf['tda_curr'][:]
    tda_temp = sensor_hf['tda_temp'][:]
    atri_volt = sensor_hf['atri_volt'][:]
    atri_curr = sensor_hf['atri_curr'][:]
    del run_info, sensor_dat, sensor_hf

    # output array
    adc_medi = np.full((num_ants, num_evts), np.nan, dtype = float)
    ped_medi = np.copy(adc_medi)
    sub_medi = np.copy(adc_medi)
    cliff = np.copy(adc_medi)
    adc_max_min = np.full((2, num_ants, num_evts), np.nan, dtype = float)
    mv_max_min = np.copy(adc_max_min)

    # 2d map
    ara_hist = sample_map_loader()
    buffer_bit_range = ara_hist.y_range
    buffer_sample_range = ara_hist.x_range

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt < 100:
       
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        
        # sub off
        ara_root.get_useful_evt(ara_root.cal_type.kJustPedWithOut1stBlockAndBadSamples)
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            sub_medi[ant, evt] = np.nanmedian(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        # ped medi
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyPedWithOut1stBlockAndBadSamples)
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            ped_medi[ant, evt] = np.nanmedian(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        # blk index
        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        samp_in_blk = buffer_info.samp_in_blk
        samp_idx = buffer_info.get_samp_idx(blk_idx_arr, ch_shape = True)
        del blk_idx_arr

        # adc medi
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            adc_medi[ant, evt] = np.nanmedian(raw_v)
            adc_max_min[0, ant, evt] = np.nanmax(raw_v)
            adc_max_min[1, ant, evt] = np.nanmin(raw_v)
            
            if clean_evt_idx[evt]:
                samp_idx_ant = samp_idx[:,ant][~np.isnan(samp_idx[:,ant])].astype(int)
                ara_hist.stack_in_hist(samp_idx_ant, raw_v.astype(int), ant)
                del samp_idx_ant            
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
        del samp_idx
 
        # cliff
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            mv_max_min[0, ant, evt] = np.nanmax(raw_v)
            mv_max_min[1, ant, evt] = np.nanmin(raw_v)
            cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 
        del samp_in_blk
    del ara_root, ara_uproot, buffer_info, num_ants, num_evts, daq_qual_sum, clean_evt_idx

    samp_medi = ara_hist.get_median_est()
    samp_map = ara_hist.hist_map
    del ara_hist

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'trig_type':trig_type,
            'total_qual_cut':total_qual_cut,
            'clean_evt':clean_evt,
            'sensor_unix':sensor_unix,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'dda_temp':dda_temp,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'tda_temp':tda_temp,
            'atri_volt':atri_volt,
            'atri_curr':atri_curr,
            'adc_medi':adc_medi,
            'ped_medi':ped_medi,
            'sub_medi':sub_medi,
            'cliff':cliff,
            'adc_max_min':adc_max_min,
            'mv_max_min':mv_max_min,
            'buffer_bit_range':buffer_bit_range,
            'buffer_sample_range':buffer_sample_range,
            'samp_medi':samp_medi,
            'samp_map':samp_map}









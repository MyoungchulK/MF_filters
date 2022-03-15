import numpy as np
from tqdm import tqdm

def mean_blk_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time

    from tools.ara_run_manager import run_info_loader
    from tools.ara_data_load import ara_Hk_uproot_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = False)
    Data = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist = ara_Hk_uproot.get_sensor_hist()
    del run_info, Data, ara_Hk_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_band_pass = True)

    ara_qual = qual_cut_loader()
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1) 
    del ara_qual, total_qual_cut

    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]   
    clean_entry = entry_num[clean_evt_idx]
    clean_unix = unix_time[clean_evt_idx]
    clean_len = len(clean_evt)   
    print(f'Number of clean event is {clean_len}') 
    del qual_cut_sum, clean_evt_idx, evt_num, entry_num, trig_type, unix_time

    # output array
    blk_est_range = 50
    blk_idx = np.full((blk_est_range, clean_len), np.nan, dtype = float)
    blk_mean = np.full((blk_est_range, num_ants, clean_len), np.nan, dtype = float)
    bp_blk_mean = np.copy(blk_mean)
    del blk_est_range
    amp_range = np.arange(-100,100)
    amp_bin = np.linspace(-100,100,200+1)
    blk_mean_evt = np.full((len(amp_range), num_ants, clean_len), 0, dtype = int)
    bp_blk_mean_evt = np.copy(blk_mean_evt)

    # loop over the events
    for evt in tqdm(range(clean_len)):
      #if evt <100:        
    
        # block index
        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(clean_entry[evt], trim_1st_blk = True)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        buffer_info.get_num_samp_in_blk(blk_idx_arr, use_int_dat = True)

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            mb = buffer_info.get_mean_blk(ant, raw_v)
            blk_mean[:blk_idx_len, ant, evt] = mb
            blk_mean_evt[:, ant, evt] = np.histogram(mb, bins = amp_bin)[0].astype(int)

            bp_v = wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)[1]
            bp_mb = buffer_info.get_mean_blk(ant, bp_v, use_int_dat = True)
            bp_blk_mean[:blk_idx_len, ant, evt] = bp_mb
            bp_blk_mean_evt[:, ant, evt] = np.histogram(bp_mb, bins = amp_bin)[0].astype(int)
            del raw_t, raw_v, bp_v, mb, bp_mb
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
        del blk_idx_arr, blk_idx_len
    del ara_root, ara_uproot, buffer_info, wf_int, clean_entry, clean_len

    def ant_hist(dat):
        bin_width = 200
        dat_bins = np.linspace(-100, 100, bin_width+1)
        dat_hist = np.full((bin_width, num_ants), 0, dtype = int)
        for ant in range(num_ants):
            dat_ant = dat[:, ant].flatten()
            dat_hist[:,ant] = np.histogram(dat_ant, bins = dat_bins)[0].astype(int)
        return dat_hist

    def ant_hist_2d(dat):
        dat_idx_fla = blk_idx.flatten()
        bin_width = 200
        amp_bins = np.linspace(-100, 100, bin_width+1)
        blk_bins = np.linspace(0, 512, 512+1)
        dat_hist_2d = np.full((512, bin_width, num_ants), 0, dtype = int)
        for ant in range(num_ants):
            dat_ant = dat[:, ant].flatten()
            dat_hist_2d[:, :, ant] = np.histogram2d(dat_idx_fla, dat_ant, bins = (blk_bins, amp_bins))[0].astype(int)
        return dat_hist_2d

    blk_mean_hist = ant_hist(blk_mean)
    bp_blk_mean_hist = ant_hist(bp_blk_mean)

    blk_mean_hist_2d = ant_hist_2d(blk_mean)
    bp_blk_mean_hist_2d = ant_hist_2d(bp_blk_mean)
    del num_ants

    print('WF collecting is done!')

    return {'clean_evt':clean_evt,
            'clean_unix':clean_unix,
            'blk_idx':blk_idx,
            'blk_mean':blk_mean,
            'bp_blk_mean':bp_blk_mean,
            'blk_mean_evt':blk_mean_evt,
            'bp_blk_mean_evt':bp_blk_mean_evt,
            'blk_mean_hist':blk_mean_hist,
            'bp_blk_mean_hist':bp_blk_mean_hist,
            'blk_mean_hist_2d':blk_mean_hist_2d,
            'bp_blk_mean_hist_2d':bp_blk_mean_hist_2d,
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}





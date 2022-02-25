import numpy as np
from tqdm import tqdm

def freq_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting freq starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    ara_geom = ara_geom_loader(ara_uproot.station_id, ara_uproot.year, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    del ara_geom

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
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1) 
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual

    # wf analyzer
    wf_int = wf_analyzer()
    dt = wf_int.dt

    # output arr
    freq = np.full((num_eles, num_evts), np.nan, dtype = float)
    peak = np.copy(freq)    

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_eles):
            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            int_v = wf_int.get_int_wf(raw_t, raw_v, ant)[1]

            fft = np.abs(np.fft.rfft(int_v)) / np.sqrt(len(int_v))
            peak_idx = np.nanargmax(fft)
            freq[ant, evt] = peak_idx / (len(int_v) * dt)
            peak[ant, evt] = fft[peak_idx]
            del raw_t, raw_v, int_v, fft, peak_idx
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   
    del ara_root, ara_uproot, num_evts, daq_qual_sum, wf_int, dt

    freq_rf = np.copy(freq)
    freq_rf[:, trig_type != 0] = np.nan
    freq_rf_w_cut = np.copy(freq)
    freq_rf_w_cut[:, ~clean_evt_idx] = np.nan

    freq_range = np.arange(0, 1, 0.01)
    freq_bins = np.linspace(0, 1, 100 + 1)
    freq_hist = np.full((num_eles, len(freq_range)), 0, dtype = int)
    freq_rf_hist = np.copy(freq_hist)
    freq_rf_w_cut_hist = np.copy(freq_hist)
    for ant in range(num_eles):
        freq_hist[ant] = np.histogram(freq[ant], bins = freq_bins)[0].astype(int)
        freq_rf_hist[ant] = np.histogram(freq_rf[ant], bins = freq_bins)[0].astype(int)
        freq_rf_w_cut_hist[ant] = np.histogram(freq_rf_w_cut[ant], bins = freq_bins)[0].astype(int)

    peak = np.log10(peak)
    peak_rf = np.copy(peak)
    peak_rf[:, trig_type != 0] = np.nan
    peak_rf_w_cut = np.copy(peak)
    peak_rf_w_cut[:, ~clean_evt_idx] = np.nan

    peak_range = np.arange(-10, 10, 0.2)
    peak_bins = np.linspace(-10, 10, 100 + 1)
    peak_hist = np.full((num_eles, len(peak_range)), 0, dtype = int)
    peak_rf_hist = np.copy(peak_hist)
    peak_rf_w_cut_hist = np.copy(peak_hist)
    for ant in range(num_eles):
        peak_hist[ant] = np.histogram(peak[ant], bins = peak_bins)[0].astype(int)
        peak_rf_hist[ant] = np.histogram(peak_rf[ant], bins = peak_bins)[0].astype(int)
        peak_rf_w_cut_hist[ant] = np.histogram(peak_rf_w_cut[ant], bins = peak_bins)[0].astype(int)
    
    def ant_hist_2d(freq, peak):
        freq_bins = np.linspace(0, 1, 100 + 1)
        peak_bins = np.linspace(-10, 10, 100 + 1)
        freq_peak_hist_2d = np.full((100, 100, num_eles), 0, dtype = int)
        for ant in range(num_eles):
            peak_ant = peak[ant].flatten()
            freq_ant = freq[ant].flatten()
            freq_peak_hist_2d[:, :, ant] = np.histogram2d(freq_ant, peak_ant, bins = (freq_bins, peak_bins))[0].astype(int)
        return freq_peak_hist_2d
   
    freq_peak_hist_2d = ant_hist_2d(freq, peak)
    freq_peak_rf_hist_2d = ant_hist_2d(freq_rf, peak_rf)
    freq_peak_rf_w_cut_hist_2d = ant_hist_2d(freq_rf_w_cut, peak_rf_w_cut)
    del num_eles, clean_evt_idx, peak_rf, peak_rf_w_cut, freq_rf, freq_rf_w_cut

    print('Freq collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'ele_ch':ele_ch,
            'total_qual_cut':total_qual_cut,
            'freq':freq,
            'freq_range':freq_range,
            'freq_bins':freq_bins,
            'freq_hist':freq_hist,
            'freq_rf_hist':freq_rf_hist,
            'freq_rf_w_cut_hist':freq_rf_w_cut_hist,
            'peak':peak,
            'peak_range':peak_range,
            'peak_bins':peak_bins,
            'peak_hist':peak_hist,
            'peak_rf_hist':peak_rf_hist,
            'peak_rf_w_cut_hist':peak_rf_w_cut_hist,
            'freq_peak_hist_2d':freq_peak_hist_2d,
            'freq_peak_rf_hist_2d':freq_peak_rf_hist_2d,
            'freq_peak_rf_w_cut_hist_2d':freq_peak_rf_w_cut_hist_2d,
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}








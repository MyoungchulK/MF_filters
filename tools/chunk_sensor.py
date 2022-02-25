import numpy as np

def sensor_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting sensor info starts!')

    from tools.ara_data_load import ara_Hk_uproot_loader

    # load data 
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    
    atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_Hk_uproot.get_daq_sensor_info()
    unix_time = ara_Hk_uproot.unix_time
    
    bin_width = 100
    try:
        if ara_Hk_uproot.station_id == 2:
            dda_volt_bins = np.linspace(3, 3.5, bin_width+1)
            dda_curr_bins = np.linspace(0, 0.5, bin_width+1)
            tda_volt_bins = np.linspace(3, 3.5, bin_width+1)
            tda_curr_bins = np.linspace(0.16, 0.21, bin_width+1)
            temp_bins = np.linspace(-25, 25, bin_width+1)
            atri_bins = np.linspace(0, 4.5, bin_width+1)
        else:
            dda_volt_bins = np.linspace(0, 10, bin_width+1)
            dda_curr_bins = np.linspace(0, 1, bin_width+1)
            tda_volt_bins = np.linspace(0, 5, bin_width+1)
            tda_curr_bins = np.linspace(0, 0.3, bin_width+1)
            temp_bins = np.linspace(-25, 25, bin_width+1)
            atri_bins = np.linspace(0, 15, bin_width+1)
    except AttributeError:
        dda_volt_bins = np.full((1), np.nan, dtype = float)        
        dda_curr_bins = np.copy(dda_volt_bins)
        tda_volt_bins = np.copy(dda_volt_bins)
        tda_curr_bins = np.copy(dda_volt_bins)
        temp_bins = np.copy(dda_volt_bins)
        atri_bins = np.copy(dda_volt_bins)

    atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist = ara_Hk_uproot.get_sensor_hist()
    del Data, ara_Hk_uproot

    print('Sensor info collecting is done!')

    return {'unix_time':unix_time,
            'atri_volt':atri_volt,
            'atri_curr':atri_curr,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'dda_temp':dda_temp,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'tda_temp':tda_temp,
            'dda_volt_bins':dda_volt_bins,
            'dda_curr_bins':dda_curr_bins,
            'tda_volt_bins':tda_volt_bins,
            'tda_curr_bins':tda_curr_bins,
            'temp_bins':temp_bins,
            'atri_bins':atri_bins,
            'dda_volt_hist':dda_volt_hist,
            'dda_curr_hist':dda_curr_hist,
            'dda_temp_hist':dda_temp_hist,
            'tda_volt_hist':tda_volt_hist,
            'tda_curr_hist':tda_curr_hist,
            'tda_temp_hist':tda_temp_hist,
            'atri_volt_hist':atri_volt_hist,
            'atri_curr_hist':atri_curr_hist}







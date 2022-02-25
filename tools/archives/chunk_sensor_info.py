import numpy as np

def sensor_info_collector(Data, Ped):

    print('Collecting sensor info starts!')

    from tools.ara_data_load import ara_Hk_uproot_loader

    # data config
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    if ara_Hk_uproot.empty_file_error == True:
        print('There is empty sensorHk file!')
        unix_time = np.full((1), np.nan, dtype = float)
        atri_volt = np.copy(unix_time)
        atri_curr = np.copy(unix_time)
        dda_volt = np.full((1,4), np.nan, dtype = float)
        dda_curr = np.copy(dda_volt)
        dda_temp = np.copy(dda_volt)
        tda_volt = np.copy(dda_volt)
        tda_curr = np.copy(dda_volt)
        tda_temp = np.copy(dda_volt)

    else:
        ara_Hk_uproot.get_sub_info()
        unix_time = ara_Hk_uproot.unix_time
        atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_Hk_uproot.get_daq_sensor_info()
    del ara_Hk_uproot

    print('Sensor info collecting is done!')

    return {'unix_time':unix_time,
            'atri_volt':atri_volt,
            'atri_curr':atri_curr,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'dda_temp':dda_temp,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'tda_temp':tda_temp}







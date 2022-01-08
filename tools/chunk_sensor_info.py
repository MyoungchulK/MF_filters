import numpy as np

def sensor_info_collector(Data, Ped):

    print('Collecting sensor info starts!')

    from tools.ara_data_load import ara_Hk_uproot_loader

    # data config
    ara_Hk_uproot = ara_Hk_uproot_loader(Data)
    ara_Hk_uproot.get_sub_info()

    #output array
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







import numpy as np
from tqdm import tqdm
from datetime import datetime

def run_time_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting run time starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_run_manager import config_info_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    del ara_uproot

    evt_unix = np.full((2), np.nan, dtype = float)
    evt_date = np.copy(evt_unix)
    evt_unix[0] = unix_time[0]
    evt_unix[1] = unix_time[-1]
    date_start = datetime.fromtimestamp(unix_time[0])
    date_start = date_start.strftime('%Y%m%d%H%M%S')
    evt_date[0] = int(date_start)
    date_stop = datetime.fromtimestamp(unix_time[-1])
    date_stop = date_stop.strftime('%Y%m%d%H%M%S')
    evt_date[1] = int(date_stop)
    del date_start, date_stop

    # config file
    ara_config = config_info_loader(verbose = True)
    slash_idx = Data.rfind('/')
    run_path = Data[:slash_idx] + '/'
    config_unix, config_date = ara_config.get_run_start_n_stop(run_path)
    del ara_config, slash_idx, run_path

    run_time = np.full((2), np.nan, dtype = float)
    run_time[0] = unix_time[-1] - unix_time[0]
    run_time[1] = config_unix[1] - config_unix[0]

    print('Run time collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'evt_unix':evt_unix,
            'evt_date':evt_date,
            'config_unix':config_unix,
            'config_date':config_date,
            'run_time':run_time}






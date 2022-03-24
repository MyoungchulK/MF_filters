import numpy as np

def l1_collector(Data, Ped, Station, Year, analyze_blind_dat = False):

    print('Collecting L1 info starts!')

    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_eventHk_uproot_loader

    # load data 
    ara_eventHk_uproot = ara_eventHk_uproot_loader(Data)
    
    l1_rate, l1_thres = ara_eventHk_uproot.get_l1_info()
    unix_time = ara_eventHk_uproot.unix_time

    if ara_eventHk_uproot.empty_file_error: 
        ara_geom = ara_geom_loader(Station, Year, verbose = True)
    else:
        ara_geom = ara_geom_loader(ara_eventHk_uproot.station_id, ara_eventHk_uproot.year, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom, ara_eventHk_uproot

    print('l1 info collecting is done!')

    return {'unix_time':unix_time,
            'l1_rate':l1_rate,
            'l1_thres':l1_thres,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch}






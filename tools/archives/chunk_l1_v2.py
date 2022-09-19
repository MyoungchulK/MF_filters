import numpy as np

def l1_collector(Data, Ped, Station, Run, Year, analyze_blind_dat = False):

    print('Collecting L1 info starts!')

    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_eventHk_uproot_loader
    
    # load data 
    ara_eventHk_uproot = ara_eventHk_uproot_loader(Data)
    l1_rate, l2_rate, l3_rate, l4_rate, l1_thres, readout_err, readout_avg, readout_min, blk_buff_avg, blk_buff_max, dig_dead, buff_dead, tot_dead = ara_eventHk_uproot.get_eventHk_info(use_prescale = True)
    unix_time = ara_eventHk_uproot.unix_time
    pps_counter = ara_eventHk_uproot.pps_counter
    clock_counter = ara_eventHk_uproot.clock_counter

    if ara_eventHk_uproot.empty_file_error: 
        ara_geom = ara_geom_loader(Station, Year, verbose = True)
    else:
        ara_geom = ara_geom_loader(ara_eventHk_uproot.station_id, ara_eventHk_uproot.year, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom, ara_eventHk_uproot

    print('l1 info collecting is done!')

    return {'unix_time':unix_time,
            'pps_counter':pps_counter,
            'clock_counter':clock_counter,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch,
            'l1_rate':l1_rate,
            'l2_rate':l2_rate,
            'l3_rate':l3_rate,
            'l4_rate':l4_rate,
            'l1_thres':l1_thres,
            'readout_err':readout_err,
            'readout_avg':readout_avg,
            'readout_min':readout_min,
            'blk_buff_avg':blk_buff_avg,
            'blk_buff_max':blk_buff_max,
            'dig_dead':dig_dead,
            'buff_dead':buff_dead,
            'tot_dead':tot_dead}




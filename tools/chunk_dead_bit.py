import numpy as np
from tqdm import tqdm

def dead_bit_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    entry_num = ara_uproot.entry_num

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    dead_bit_hist = np.full((num_ants, num_bits), 0, dtype = int)
    dead_bit_bins = np.linspace(0, num_bits, num_bits+1)
    dead_bit_range = np.arange(num_bits)

    from tools.ara_quality_cut import pre_qual_cut_loader
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del pre_qual, pre_qual_cut

    from tools.ara_run_manager import run_info_loader
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run)
    Data = run_info.get_data_path(file_type = 'sensorHk', manual_stop = True)[0]
    if Data is None:
        print('There is no sensorHk file!')
        dda_volt = np.full((1,4), np.nan, dtype = float)
    else:
        from tools.ara_data_load import ara_Hk_uproot_loader
        ara_Hk_uproot = ara_Hk_uproot_loader(Data)
        ara_Hk_uproot.get_sub_info()
        dda_volt_curr = ara_Hk_uproot.dda_volt_curr
        dda_volt = ara_Hk_uproot.get_voltage(dda_volt_curr)
        del ara_Hk_uproot, dda_volt_curr
    del run_info

    dda_range = np.arange(0,20,0.01)
    dda_bins = np.linspace(0, 20, 2000+1)
    dda_volt_hist = np.full((len(dda_range), 4), 0, dtype = int)
    for d in range(4):
        dda_volt_hist[:,d] = np.histogram(dda_volt[:,d], bins = dda_bins)[0].astype(int)    
    del dda_volt

    num_evts = ara_uproot.num_evts
    trig_ratio = np.full((3), np.nan, dtype = float)
    trig_ratio[0] = np.count_nonzero(trig_type == 0)
    trig_ratio[1] = np.count_nonzero(trig_type == 1)
    trig_ratio[2] = np.count_nonzero(trig_type == 2)
    trig_ratio /= num_evts
    del num_evts

    # loop over the events
    for evt in tqdm(range(len(entry_num))):
      #if evt <100:        
    
        if pre_qual_cut_sum[evt] != 0 or trig_type[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyGoodADC)

        # loop over the antennas
        for ant in range(num_ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1].astype(int)
            dead_bit_hist[ant] += np.histogram(raw_v, bins = dead_bit_bins)[0].astype(int)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, entry_num, dead_bit_bins, pre_qual_cut_sum, num_ants, num_bits

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'trig_ratio':trig_ratio,
            'dead_bit_range':dead_bit_range,
            'dead_bit_hist':dead_bit_hist,
            'dda_range':dda_range,
            'dda_volt_hist':dda_volt_hist}








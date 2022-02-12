import numpy as np
from tqdm import tqdm

def qual_cut_temp_collector(Data, Ped):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import pre_qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const 

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    del pre_qual, ara_uproot

    timing_err = np.full((num_ants, num_evts), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t = ara_root.get_rf_ch_wf(ant)[0]
            if len(raw_t) == 0:
                pass
            else:
                timing_err[ant, evt] = int(np.any(np.diff(raw_t)<0))
            del raw_t
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_root, num_evts, num_ants

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'timing_err':timing_err}





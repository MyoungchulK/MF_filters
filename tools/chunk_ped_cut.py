import os
import numpy as np
from tqdm import tqdm
import h5py

def ped_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Pedestal cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import ped_qual_cut_loader
    from tools.ara_quality_cut import get_time_smearing
    from tools.ara_run_manager import run_info_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time

    # daq & cw quality cuts
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'daq_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    daq_cut = daq_hf['total_daq_cut'][:]
    daq_qual_cut_sum = daq_hf['daq_qual_cut_sum'][:]

    force_unblind = True
    cw_dat = run_info.get_result_path(file_type = 'cw_cut', verbose = True, force_unblind = force_unblind)
    cw_hf = h5py.File(cw_dat, 'r')
    cw_cut = cw_hf['total_cw_cut'][:]
    rp_evts = cw_hf['rp_evts'][:]
    rp_evts = np.repeat(rp_evts[:, np.newaxis], 1, axis = 1)
    cw_cut += rp_evts
    print(np.sum(cw_cut))
    if force_unblind:
        cw_cut = np.nansum(cw_cut, axis = 1)
        cw_pps = cw_hf['pps_number'][:]
        cw_smear_time = get_time_smearing(cw_pps[cw_cut != 0]) 
        cw_cut = np.in1d(pps_number, cw_smear_time).astype(int)
        cw_cut = np.repeat(cw_cut[:, np.newaxis], 1, axis = 1)
        print(np.sum(cw_cut))
        del cw_pps, cw_smear_time
    daq_cw_cut = np.append(daq_cut, cw_cut, axis = 1)
    daq_cw_cut_sum = np.nansum(daq_cw_cut, axis = 1)
    del run_info, daq_dat, daq_hf, daq_cut, cw_dat, cw_hf, cw_cut, rp_evts

    # ped quailty cut
    ped_qual = ped_qual_cut_loader(ara_uproot, daq_cw_cut, daq_qual_cut_sum, analyze_blind_dat = analyze_blind_dat, verbose = True)
    ped_qual_evt_num, ped_qual_type, ped_qual_num_evts, ped_blk_usage, ped_low_blk_usage, ped_qualities, ped_counts, ped_final_type = ped_qual.get_pedestal_information()
    total_ped_cut = ped_qual.run_ped_qual_cut()
    total_ped_cut_sum = ped_qual.ped_qual_cut_sum
    del ara_uproot, ped_qual, daq_cw_cut

    print('Pedestal cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'daq_cw_cut_sum':daq_cw_cut_sum,
            'total_ped_cut':total_ped_cut,
            'total_ped_cut_sum':total_ped_cut_sum,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type}





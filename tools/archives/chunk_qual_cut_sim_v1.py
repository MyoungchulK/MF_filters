import os
import numpy as np
from tqdm import tqdm
import h5py

def qual_cut_sim_collector(Data, Station, Year):

    print('Quality cut 3rd starts!')

    #from tools.ara_constant import ara_const   
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import filt_qual_cut_loader
    from tools.ara_run_manager import get_example_run
    from tools.ara_run_manager import get_file_name

    ## file name
    exponent = int(get_path_info_v2(Data, '_E', '_F'))
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    flavor = int(get_path_info_v2(Data, '_F', '_A'))
    sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
    ex_run = get_example_run(Station, config)
    h5_file_name = get_file_name(Data)

    ## result paths
    cw_r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/cw_ratio_sim/cw_ratio_{h5_file_name}.h5'
    rms_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_sim/rms_{h5_file_name}.h5'
    reco_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_sim/reco_{h5_file_name}.h5'
    print('cw ratio path:', cw_r_path)
    print('rms path:', rms_path)
    print('reco path:', reco_path)
    del h5_file_name

    ## entry number
    hf = h5py.File(cw_r_path, 'r')
    entry_num = hf['entry_num'][:]
    del hf

    ## cw ratio 
    pre_qual = pre_qual_cut_loader(0, sim_st = Station, sim_run = ex_run, sim_evt = entry_num)
    pre_qual_cut = np.full((len(entry_num), 1), 0, dtype = int) 
    pre_qual_cut[:, 0] = pre_qual.get_cw_ratio_events(sim_path = cw_r_path)
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del cw_r_path, pre_qual

    ## spark and cal, surface
    filt_qual = filt_qual_cut_loader(Station, ex_run, entry_num, verbose = True, sim_spark_path = rms_path, sim_cal_sur_path = reco_path)
    filt_qual_cut = filt_qual.run_filt_qual_cut()
    filt_qual_cut_sum = filt_qual.filt_qual_cut_sum
    del filt_qual, ex_run, rms_path, reco_path

    ## total quality cut
    tot_qual_cut = np.append(pre_qual_cut, filt_qual_cut, axis = 1)
    tot_qual_cut_sum = np.nansum(tot_qual_cut, axis = 1)

    ## one weight
    signal_key = 'signal'
    if Data.find(signal_key) != -1:
        wei_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/One_Weight_Pad_mass_A{Station}.h5'
        print('weight path:', wei_path)
        wei_hf = h5py.File(wei_path, 'r') 
        one_weight_tot = wei_hf['one_weight'][:] 
        evt_rate_tot = wei_hf['evt_rate'][:]
        flavor_tot = wei_hf['flavor'][:]
        config_tot = wei_hf['config'][:]
        sim_run_tot = wei_hf['sim_run'][:]
        exponent_tot = wei_hf['exponent'][:, 0]
        idxs = np.all((sim_run_tot == sim_run, flavor_tot == flavor, config_tot == config, exponent_tot == int(exponent - 9)), axis = 0)       
        one_weight = one_weight_tot[idxs]
        evt_rate = evt_rate_tot[idxs]
        del wei_path, wei_hf, flavor_tot, config_tot, sim_run_tot, one_weight_tot, evt_rate_tot, exponent_tot, idxs
    else:
        one_weight = np.full((len(entry_num)), 1, dtype = float)        
        evt_rate = np.copy(one_weight)
    del exponent, config, flavor, sim_run, signal_key

    print('Quality cut sim is done!')

    return {'entry_num':entry_num,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'filt_qual_cut':filt_qual_cut,
            'filt_qual_cut_sum':filt_qual_cut_sum,
            'tot_qual_cut':tot_qual_cut,
            'tot_qual_cut_sum':tot_qual_cut_sum,
            'one_weight':one_weight,
            'evt_rate':evt_rate}





##
# @file chunk_rayl.py
#
# @section Created on 06/29/2022, mkim@icecube.wisc.edu
#
# @brief loads data by AraRoot, make FFTs, and perform Rayleigh fitting for given run

import os, sys
import h5py
import numpy as np
from tqdm import tqdm

def rayl_collector(Data):
    """! rayleigh fitting 
    
    @param Data  String. data path ex)/data/exp/ARA/2014/unblinded/L1/ARA02/1027/run004434/event004434.root
    """

    from tools import ara_root_loader
    from tools import wf_analyzer
    from tools import get_rayl_distribution
    from tools import get_config_info
    from tools import get_qual_cut

    ## scrap info from data path
    Ped, st, run, year = get_config_info(Data) 

    ## data loading by usiung araroot
    ara_root = ara_root_loader(Data, Ped, st, year)
    num_evts = ara_root.num_evts
    num_ants = 16 # number of rf channels

    ## quality cut
    bad_evt = get_qual_cut(st, run)

    ## wf analyzer. zero padding, band pass and fft
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = True)

    ## fft arrays
    rf_rffts = []
    soft_rffts = []

    ## loop over the events
    for evt in tqdm(range(num_evts)):

        ## quality cut
        if bad_evt[evt]: # quality cut result for each event. 0: pass, 1: cut
            continue

        ## get entry, trigger, and calibrated wf
        ara_root.get_entry(evt)
        trig_type = ara_root.get_trig_type() # 0:rf, 1:cal, 2:soft
        if trig_type == 1: #pass calpulser
            continue
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        ## loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant) # wf info. time[ns] and amplitudep[mV]
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True) # interpolation, band pass, and zero padding
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ## rfft with normalization
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)
        rfft_evt = wf_int.pad_fft # rfft array dim: (rfft length, number of channels)
        if trig_type == 0:
            rf_rffts.append(rfft_evt)
        elif trig_type == 2:
            soft_rffts.append(rfft_evt)
        del trig_type
    del num_evts, num_ants, ara_root, wf_int
  
    ## to numpy array
    rf_rffts = np.asarray(rf_rffts)
    soft_rffts = np.asarray(soft_rffts)
 
    ## rayl fit. sigma, 2d fft distribution, binsedges for each frequency 
    num_bins = np.array([1000], dtype = int)
    rf_rayl, rf_rfft_hist2d, rf_bin_edges = get_rayl_distribution(rf_rffts, binning = num_bins[0])
    soft_rayl, soft_rfft_hist2d, soft_bin_edges = get_rayl_distribution(soft_rffts, binning = num_bins[0])
    del rf_rffts, soft_rffts

    print('Rayl. fitting is done!')

    return {'num_bins':num_bins,                    # number of binning for 2d rfft distribution
            'rf_bin_edges':rf_bin_edges,            # array of bin edges for RF channel
            'soft_bin_edges':soft_bin_edges,        # array of bin edges for Software channel
            'rf_rfft_hist2d':rf_rfft_hist2d,        # rfft distribution for RF channel
            'soft_rfft_hist2d':soft_rfft_hist2d,    # rfft distribution for Software channel
            'rf_rayl':rf_rayl,                      # rayl. fit parameters for RF channel
            'soft_rayl':soft_rayl}                  # rayl. fit parameters for Software channel


def script_loader(Data, Output):
    """! load wrapping script and save the result in output path

    @param Data  String. data path ex)/data/exp/ARA/2014/unblinded/L1/ARA02/1027/run004434/event004434.root
    @param Output  String. desired output path
    """
    from tools import get_path_info    

    ## run the wrapping code
    results = rayl_collector(Data)

    ## create output dir
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    Station = int(get_path_info(Data, '/ARA0', '/'))
    Run = int(get_path_info(Data, '/run', '/'))
    h5_file_name = f'{Output}rayl_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')

    ## saving result
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    hf.close()
    print(f'Output is {h5_file_name}')

if __name__ == "__main__":

    if len (sys.argv) < 2:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Data path ex)/data/exp/ARA/2014/unblinded/L1/ARA02/1027/run004434/event004434.root>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/rayl/>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    data = str(sys.argv[1])
    output = str(sys.argv[2])
    script_loader(data, output)

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
import click # 'pip3 install click' will make you very happy

@click.command()
@click.option('-d', '--data', type = str, help = 'ex) /data/exp/ARA/2014/blinded/L1/ARA02/1026/run004434/event004434.root')
@click.option('-p', '--ped', type = str, help = 'ex) /data/user/mkim/OMF_filter/ARA02/ped_full/ped_values_full_A2_R4434.dat')
@click.option('-o', '--output', type = str, help = 'ex) /home/mkim/')
@click.option('-q', '--qual', default = '', type = str, help = 'ex) ~.txt This text file should contain array that size is as same as number of events and value should be 0 (pass) or 1 (cut)')
def main(data, ped, output, qual):
    """! rayleigh fitting, and gain extraction 
    
    @param data  String. data path
    @param ped  String. pedestal path
    @param output  String. output path
    @param qual  String. quality cut path
    """

    from tools import ara_root_loader
    from tools import wf_analyzer
    from tools import get_config_info
    from tools import get_rayl_distribution
    from tools import get_signal_chain_gain

    ## scrap data info
    st, run, year, soft_blk_len = get_config_info(data) 

    ## data loading by using araroot
    ara_root = ara_root_loader(data, ped, st, year)
    num_evts = ara_root.num_evts
    num_ants = 16 # number of rf channels

    ## quality cut. if user have it...
    if len(qual) != 0:
        bad_evt = np.loadtxt(qual)
        bad_evt = bad_evt.astype(int)
    else:
        bad_evt = np.full((num_evts), 0, dtype = int)

    ## wf analyzer. zero padding, band pass and fft
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = True, use_cw = True)
    dt = wf_int.dt
    freq_range = wf_int.pad_zero_freq

    ## fft arrays
    soft_rffts = []

    ## loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt < 100: # debug

        ## quality cut
        if bad_evt[evt]: # quality cut result for each event. 0: pass, 1: cut
            continue

        ## get entry, trigger, and block length
        ara_root.get_entry(evt)
        trig_type = ara_root.get_trig_type() # 0:rf, 1:cal, 2:soft
        blk_len = ara_root.get_block_length()
        if trig_type != 2 or blk_len != soft_blk_len: # using only good software triggered event
            continue
        del trig_type, blk_len

        ## calibration
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        ## loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant) # wf info. time[ns] and amplitudep[mV]
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True) # interpolation, band pass, cw, and zero padding
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ## storing rffs
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True) # rfft with normalization
        rfft_evt = wf_int.pad_fft # rfft array dim: (rfft length, number of channels)
        soft_rffts.append(rfft_evt)
    del num_evts, num_ants, ara_root, wf_int, soft_blk_len, year, bad_evt
  
    ## to numpy array
    soft_rffts = np.asarray(soft_rffts)
    print(f'rfft array dim. (evt, rfft, chs): {soft_rffts.shape}')
 
    ## rayl fit. sigma, 2d fft distribution, binsedges for each frequency 
    num_bins = np.array([1000], dtype = int)
    soft_rayl, soft_rfft_hist2d, soft_bin_edges = get_rayl_distribution(soft_rffts, binning = num_bins[0])
    del soft_rffts

    ## signal chain gain extraction
    soft_sc = get_signal_chain_gain(np.nansum(soft_rayl, axis = 0), freq_range, dt, st) # let use loc + scale

    ## create output path
    if not os.path.exists(output):
        os.makedirs(output)
    hf_file_name = f'{output}Gain_A{st}_R{run}.h5'
    hf = h5py.File(hf_file_name, 'w')
    hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
    hf.create_dataset('num_bins', data=num_bins, compression="gzip", compression_opts=9) # number of binning for 2d rfft distribution
    hf.create_dataset('soft_rayl', data=soft_rayl, compression="gzip", compression_opts=9) # array of bin edges for Software channel
    hf.create_dataset('soft_rfft_hist2d', data=soft_rfft_hist2d, compression="gzip", compression_opts=9) # rfft distribution for Software channel
    hf.create_dataset('soft_bin_edges', data=soft_bin_edges, compression="gzip", compression_opts=9) # rayl. fit parameters for Software channel
    hf.create_dataset('soft_sc', data=soft_sc, compression="gzip", compression_opts=9) # array of signal chain gain
    hf.close()

    print(f'Output is {hf_file_name}')
    print('Done!')

if __name__ == "__main__":

    main()

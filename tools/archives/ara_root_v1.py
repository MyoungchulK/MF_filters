import numpy as np
import sys, os
import uproot
import h5py

#custom lib.
from tools.array import arr_2d

def chunk_range_maker(num_evts, chunk_size = 5000, chunk_i = None):

    if chunk_i is not None:
        event_i = chunk_i * chunk_size
        event_f = chunk_i * chunk_size + chunk_size
        if event_f >= num_evts:
            event_f = num_evts
            final_chunk = True
            print('It is final chunk!')
        else:
            final_chunk = False
    else:
        event_i = 0
        event_f = num_evts
        final_chunk = True
    print(f'Analyzing event# {event_i} to {event_f}')

    return event_i, event_f, final_chunk

# import root and ara root lib
def ara_root_lib():
    
    # general cern root lib
    import ROOT

    # cvmfs ara lib
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

    return ROOT

def AraGeom_loader(ROOT, st, ant_ch, yrs):

    # create a geomtool
    geomTool = ROOT.AraGeomTool.Instance()

    # trigger channel info
    pol_type = np.zeros((ant_ch), dtype=int)
    trig_ch = np.copy(pol_type)
    ele_ch = np.copy(pol_type)
    for ant in range(ant_ch):
        pol_type[ant] = geomTool.getStationInfo(st, yrs).getAntennaInfo(ant).polType
        trig_ch[ant] = geomTool.getStationInfo(st, yrs).getAntennaInfo(ant).getTrigChan()
        ele_ch[ant] = geomTool.getStationInfo(st, yrs).getAntennaInfo(ant).daqChanNum

    print('electronic channel:',ele_ch)
    print('trigger channel:',trig_ch)
    print('polarization type:',pol_type)
    del geomTool

    return trig_ch, pol_type, ele_ch

def qual_loader(ROOT):

    # open general quilty cut
    q = ROOT.AraQualCuts.Instance()

    return q

# general data loading procedure by araroot
def ara_raw_to_qual(ROOT, data, ped, st, ant_ch):

    # open a data file
    file = ROOT.TFile.Open(data)

    # load in the event free for this file
    evtTree = file.Get("eventTree")

    # set the tree address to access our raw data type
    rawEvt = ROOT.RawAtriStationEvent()
    evtTree.SetBranchAddress("event",ROOT.AddressOf(rawEvt))

    # get the number of entries in this file
    num_evts = int(evtTree.GetEntries())
    print('total events:', num_evts)

    # open a pedestal file
    cal = ROOT.AraEventCalibrator.Instance()
    cal.setAtriPedFile(ped, st)

    return file, evtTree, rawEvt, num_evts, cal#not sure need to return the 'cal'

def useful_evt_maker(ROOT, evtTree, rawEvt, evt, cal): #not sure need to input the 'cal'

    # get the event
    evtTree.GetEntry(evt)

    usefulEvt = ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)

    return usefulEvt

#return ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)

#def useful_evt_maker_del(ROOT):
#
#    ROOT.~UsefulAtriStationEvent()

# make a useful event
#usefulEvt = ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)
#return usefulEvt

def trig_checker(rawEvt):

    # default trig tag
    trig = -1#'Unknown'

    # trigger tag check
    if rawEvt.isSoftwareTrigger() == 1:
        trig = 2#'Soft'
    elif rawEvt.isCalpulserEvent() == 1:
        trig = 1#'Cal'
    elif rawEvt.isSoftwareTrigger() == 0 and rawEvt.isCalpulserEvent() == 0:
        trig = 0#'RF'
    else:
        pass

    return trig

def trig_ch_checker(rawEvt, trig_chs, num_Ants):

    # trigger info
    high_trig_ch = np.full((num_Ants), 0, dtype = int)

    for ant in range(num_Ants):

        # high triggered ch
        if rawEvt.isTriggerChanHigh(int(trig_chs[ant])) == True:
            high_trig_ch[ant] = 1
        else:
            pass  

    return high_trig_ch

def evt_num_checker(rawEvt):

    # actual event number
    evt_n = rawEvt.eventNumber
    
    return evt_n

def unix_time_checker(rawEvt):
  
    #unix time info
    unix_t = rawEvt.unixTime
    unix_t_us = rawEvt.unixTimeUs
    unix_time = np.array([unix_t, unix_t_us])
    del unix_t, unix_t_us

    return unix_time

def read_win_checker(rawEvt):

    # print readout window
    read_win = rawEvt.numReadoutBlocks

    return read_win

def time_stamp_checker(rawEvt):

    # print readout window
    time_stamp = rawEvt.timeStamp

    return time_stamp

def qual_checker(q, usefulEvt):

    # default qual tag
    qual = -1#'Unknown'

    # qual tag check
    if q.isGoodEvent(usefulEvt) == 1:
        qual = 0#'Pass'
    elif q.isGoodEvent(usefulEvt) == 0:
        qual = 1#'Cut'
    else:
        pass

    return qual

def sub_info_checker(rawEvt, q, usefulEvt, ch_index, num_Ants):

    # trig type
    trig_type = trig_checker(rawEvt)

    # actual event number
    evt_num = evt_num_checker(rawEvt)

    #unix time info
    unix_time = unix_time_checker(rawEvt)

    # trigger info
    trig_ant = trig_ch_checker(rawEvt, trig_chs, num_Ants)

    # print readout window
    time_stamp = rawEvt.timeStamp

    # print readout window
    read_win = rawEvt.numReadoutBlocks

    return trig_type, evt_num, unix_time, trig_ant, time_stamp, read_win


def ant_xyz(ROOT, Station, num_ant, Years = None):

    # create a geomtool
    geomTool = ROOT.AraGeomTool.Instance()

    # array for xyz coord
    ant_xyz = arr_2d(num_ant, 3, np.nan, float)

    # the x-y coordinates of channels 0-3 are enough for a top down view
    for ant in range(num_ant):

        if Years is not None:
            ant_xyz[ant] = geomTool.getStationInfo(Station,Years).getAntennaInfo(ant).antLocation
        else:
            ant_xyz[ant] = geomTool.getStationInfo(Station).getAntennaInfo(ant).antLocation

    del geomTool

    return ant_xyz

def cal_length(Station):

    pulserTime = 0
    
    if Station == 1:
        pulserTime=254
    elif Station == 2:
        pulserTime=245
    elif Station == 3:
        pulserTime=245
    elif Station == 4:
        pulserTime=400
    elif Station == 5:
        pulserTime=400
    
    return pulserTime
"""
def uproot_loader(Data, Station, num_Ants, num_evts, ch_index):

    #custom lib
    from tools.qual import first_five_event_error_uproot
    from tools.qual import block_gap_error_uproot

    # open file
    file = uproot.open(Data)

    evt_num = np.arange(num_evts)

    try: 
        evtTree = file['eventTree']
    except uproot.exceptions.KeyInFileError:
        print('Data file is corrupted! Couldnt open by uproot. Gonna try by PyRoot!')
        hasKeyInFileError = True
        trig_num = np.zeros((num_evts))       
        qual_num_tot = np.full((num_evts),0,dtype=int)
        act_evt_num = np.zeros((num_evts))
        trig_chs = np.full((num_Ants, num_evts),0,dtype=int)
        time_stamp = np.zeros((num_evts))
        read_win = np.zeros((num_evts))
        unix_time = np.zeros((2,num_evts))
        del file

        return evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, hasKeyInFileError
    
    hasKeyInFileError = False   
 
    act_evt_num = np.asarray(evtTree['eventNumber'],dtype=int)

    unixTime = np.asarray(evtTree['unixTime'],dtype=int)
    unixTimeUs = np.asarray(evtTree['unixTimeUs'],dtype=int)
    unix_time = np.array([unixTime,unixTimeUs])
    del unixTime, unixTimeUs

    read_win = np.asarray(evtTree['event/numReadoutBlocks'],dtype=int)

    time_stamp = np.asarray(evtTree['event/timeStamp'],dtype=int)
    triggerInfo = np.asarray(evtTree['event/triggerInfo[4]'],dtype=int)

    ch_index_bit = 1<<ch_index
    trig_chs = np.repeat(triggerInfo[:,0][np.newaxis, :], num_Ants, axis=0) & ch_index_bit[:,np.newaxis]
    del ch_index_bit
    trig_chs[trig_chs != 0] = 1
    trig_chs = trig_chs.astype(int)

    cal_index = np.where(np.abs(time_stamp - cal_length(Station)) < 1e4)[0]
    soft_index = np.where(triggerInfo[:,2] == 1)[0]
    del triggerInfo
    trig_num = np.zeros((num_evts),dtype=int)
    trig_num[cal_index] = 1
    trig_num[soft_index] = 2
    del cal_index, soft_index

    # quality cut
    qual_type = 9 #0 five event, 1 block gap, 2 timing, 3 few block, 4 offset, 5 cliff, 6 freq glitch, 7 spare, 8 spikey
    qual_num_tot = np.full((qual_type, num_evts),0,dtype=int)
    
    # first five event
    qual_num_tot[0] = first_five_event_error_uproot(Station, unix_time[0], act_evt_num)    
    
    # block gap
    irsBlockNumber = np.asarray(evtTree['event/blockVec/blockVec.irsBlockNumber'])
    qual_num_tot[1] = block_gap_error_uproot(irsBlockNumber)

    del file, evtTree, irsBlockNumber

    return evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, hasKeyInFileError
"""
def uproot_loader(Data, Station, num_Ants, num_evts, trig_ch):

    # open file
    file = uproot.open(Data)

    entry_num = np.arange(num_evts)

    try:
        evtTree = file['eventTree']
    except uproot.exceptions.KeyInFileError:
        print('Data file is corrupted! Couldnt open by uproot. Gonna try by PyRoot!')
        hasKeyInFileError = True
        trig_type = np.zeros((num_evts))
        evt_num = np.zeros((num_evts))
        trig_ant = np.full((num_Ants, num_evts),0,dtype=int)
        time_stamp = np.zeros((num_evts))
        read_win = np.zeros((num_evts))
        unix_time = np.zeros((2,num_evts))
        del file

        return entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError

    hasKeyInFileError = False

    evt_num = np.asarray(evtTree['eventNumber'],dtype=int)

    unixTime = np.asarray(evtTree['unixTime'],dtype=int)
    unixTimeUs = np.asarray(evtTree['unixTimeUs'],dtype=int)
    unix_time = np.array([unixTime,unixTimeUs])
    del unixTime, unixTimeUs

    read_win = np.asarray(evtTree['event/numReadoutBlocks'],dtype=int)

    time_stamp = np.asarray(evtTree['event/timeStamp'],dtype=int)
    triggerInfo = np.asarray(evtTree['event/triggerInfo[4]'],dtype=int)

    ch_index_bit = 1<<trig_ch
    trig_ant = np.repeat(triggerInfo[:,0][np.newaxis, :], num_Ants, axis=0) & ch_index_bit[:,np.newaxis]
    del ch_index_bit
    trig_ant[trig_ant != 0] = 1
    trig_ant = trig_ant.astype(int)

    cal_index = np.where(np.abs(time_stamp - cal_length(Station)) < 1e4)[0]
    soft_index = np.where(triggerInfo[:,2] == 1)[0]
    del triggerInfo
    trig_type = np.zeros((num_evts),dtype=int)
    trig_type[cal_index] = 1
    trig_type[soft_index] = 2
    del cal_index, soft_index, file, evtTree

    return entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError

"""    
def ara_raw_to_qual_plus_info(ROOT, data, ped, st, ant_ch, yrs):

    # geom. info.
    ch_index, pol_type = AraGeom_loader(ROOT, st, ant_ch, yrs)

    # load data and cal
    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(ROOT, data, ped, st, ant_ch)

    # sub. info.
    evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, hasKeyInFileError = uproot_loader(data, st, ant_ch, num_evts, ch_index)

    if hasKeyInFileError == True:
        q = qual_loader(ROOT)
    else:
        q = None

    return file, evtTree, rawEvt, num_evts, cal, q, ch_index, pol_type, evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, hasKeyInFileError
"""
def ara_raw_to_qual_plus_info(ROOT, data, ped, st, ant_ch, yrs):

    # geom. info.
    trig_ch, pol_type, ele_ch = AraGeom_loader(ROOT, st, ant_ch, yrs)

    # load data and cal
    file, evtTree, rawEvt, num_evts, cal = ara_raw_to_qual(ROOT, data, ped, st, ant_ch)

    # sub. info.
    entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError = uproot_loader(data, st, ant_ch, num_evts, trig_ch)

    return file, evtTree, rawEvt, num_evts, cal, ele_ch, trig_ch, pol_type, entry_num, evt_num, unix_time, trig_type, trig_ant, time_stamp, read_win, hasKeyInFileError

def useful_dda_ch_idx(hist_mask = 0x0f0f0f0f, dda_ch_num = 32):

    useful_ch = []
    for ant in range(dda_ch_num):
        if hist_mask & (1 << ant):
            useful_ch.append(ant)
        else:
            pass
    useful_ch = np.asarray(useful_ch)
    print('Useful DDA Ch.:',useful_ch)

    return useful_ch

def sample_block_identifier(rawEvt, useful_ch, dda_ch_num = 32, chan_per_dda = 8, samp_per_block = 64, block_per_dda = 512, trim = False):

    nsamp = block_per_dda * samp_per_block

    # find chip index
    chip_evt_idx = []
    block_evt_idx = []
    for n in range(dda_ch_num):
        chip_evt_idx.append([])
        block_evt_idx.append([])

    blk_len = rawEvt.blockVec.size()
    for iblk in range(blk_len):
        chan_idx = 0
        nchannels = rawEvt.blockVec[iblk].getNumChannels()
        for ich in range(nchannels):
            if rawEvt.blockVec[iblk].channelMask ==0 and (1 << ich) == 0:
                continue
            chan= chan_per_dda * rawEvt.blockVec[iblk].getDda() + ich
            block_num = rawEvt.blockVec[iblk].getBlock()
            block_evt_idx[chan].append(block_num)
            offset = block_num * samp_per_block
            size = rawEvt.blockVec[iblk].data[chan_idx].size()
            for isamp in range(size):
                i = (offset+isamp) % nsamp
                chip_evt_idx[chan].append(i)
            chan_idx += 1
            del chan, offset, size
        del nchannels, chan_idx
    del blk_len

    chip_evt_idx = np.asarray(chip_evt_idx)
    block_evt_idx = np.asarray(block_evt_idx)
    chip_evt_idx = chip_evt_idx[useful_ch]
    block_evt_idx = block_evt_idx[useful_ch]
      
    if trim == True:
        chip_evt_idx = chip_evt_idx[:, samp_per_block:]
        block_evt_idx = block_evt_idx[:, 1:]
    else:
        pass

    return chip_evt_idx, block_evt_idx

def sample_in_block_loader(Station, ele_ch):

    #cap_name = f'/home/mkim/analysis/MF_filters/data/araAtriStation{Station}SampleTimingNew_CapNum_Only.h5'
    cap_name = f'/home/mkim/analysis/MF_filters/data/araAtriStation{Station}SampleTimingNew.h5'
    cap_file = h5py.File(cap_name, 'r')
    cap_num_arr = cap_file['cap_arr'][:]
    cap_num_arr = cap_num_arr[:,ele_ch]

    idx_arr = cap_file['idx_arr_rm_overlap'][:]

    del cap_file, cap_name
    print('number of samples in even/odd block is loaded!')

    return cap_num_arr, idx_arr 

def block_idx_identifier(rawEvt, trim_1st_blk = False, modulo_2 = False):

    dda_num = 4
    remove_1_blk = int(trim_1st_blk)

    blk_len = rawEvt.blockVec.size()//dda_num - remove_1_blk
    blk_arr = np.full((blk_len),0,dtype=int)
    for iblk in range(blk_len):
        blk_arr[iblk] = rawEvt.blockVec[dda_num*(iblk + remove_1_blk)].getBlock()
    
    if modulo_2 == True:
        blk_arr = blk_arr%2
    else:
        pass
    del blk_len, remove_1_blk, dda_num

    return blk_arr


































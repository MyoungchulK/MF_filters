import numpy as np
import os
import uproot

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

# general data loading procedure by araroot
def ara_raw_to_qual(ROOT, data, ped, st, ant_ch, yrs):

    # open a data file
    file = ROOT.TFile.Open(data)

    # load in the event free for this file
    evtTree = file.Get("eventTree")
    #del file

    # set the tree address to access our raw data type
    rawEvt = ROOT.RawAtriStationEvent()
    evtTree.SetBranchAddress("event",ROOT.AddressOf(rawEvt))

    # get the number of entries in this file
    num_evts = int(evtTree.GetEntries())
    print('total events:', num_evts)

    # create a geomtool
    geomTool = ROOT.AraGeomTool.Instance()

    # trigger channel info
    pol_type = np.zeros((ant_ch), dtype=int)
    trig_chs = np.copy(pol_type)
    for ant in range(ant_ch):
        pol_type[ant] = geomTool.getStationInfo(st, yrs).getAntennaInfo(ant).polType
        trig_chs[ant] = geomTool.getStationInfo(st, yrs).getAntennaInfo(ant).getTrigChan()

    print('trigger channel:',trig_chs)
    print('polarization type:',pol_type)

    # open a pedestal file
    cal = ROOT.AraEventCalibrator.Instance()
    cal.setAtriPedFile(ped, st)

    # open general quilty cut
    q = ROOT.AraQualCuts.Instance()

    return file, evtTree, rawEvt, num_evts, cal, q, trig_chs, pol_type #not sure need to return the 'cal'

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
    high_trig_ch = np.full((num_Ants), np.nan)

    for ant in range(num_Ants):

        # high triggered ch
        if rawEvt.isTriggerChanHigh(int(trig_chs[ant])) == True:
            high_trig_ch[ant] = int(trig_chs[ant])
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
        qual = 1#'Pass'
    elif q.isGoodEvent(usefulEvt) == 0:
        qual = 0#'Cut'
    else:
        pass

    return qual

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

def element_finder(index, act_evt_num = None, read_win = None, time_stamp = None, unix_time = None, trig_chs = None):

    if act_evt_num is not None:
       act_evt_num = act_evt_num[index]
    else:
       pass
    
    if read_win is not None:
       read_win  = read_win[index]
    else:
        pass

    if time_stamp is not None:
       time_stamp  = time_stamp[index]
    else:
        pass

    if unix_time is not None:
       unix_time  = unix_time[:,index]
    else:
        pass

    if trig_chs is not None:
       trig_chs  = trig_chs[:,index]
    else:
        pass

    return act_evt_num, read_win, time_stamp, unix_time, trig_chs

def uproot_loader(Data, Station, event_i, event_f, ch_index,
                    trig_set = 'all',
                    act_evt_info = True,
                    trig_info = True,
                    time_stamp_info = True,
                    read_win_info = True,
                    unix_info = True):

    #custom lib
    from tools.qual import first_five_event_error_uproot
    from tools.qual import block_gap_error_uproot

    # open file
    file = uproot.open(Data)

    evt_num = np.arange(event_i, event_f)

    try: 
        evtTree = file['eventTree']
    except uproot.exceptions.KeyInFileError:
        print('Data file is corrupted! Couldnt open by uproot. Gonna try by PyRoot!')
        hasKeyInFileError = True
        trig_num = []       
        qual_num = np.full((len(evt_num)), -1, dtype=int) 
        qual_num_tot = None

        if act_evt_info == True:
            act_evt_num = []
        else:
            act_evt_num = None
        if trig_info == True:
            trig_chs = []
        else:
            trig_chs = None
        if time_stamp_info == True:
            time_stamp = []
        else:
            time_stamp = None
        if read_win_info == True:
            read_win = []
        else:
            read_win = None
        if unix_info == True:
            unix_time = []
        else:
            unix_time = None

        del file

        return evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, hasKeyInFileError

    hasKeyInFileError = False   
 
    if act_evt_info == True:
        act_evt_num = np.asarray(evtTree['eventNumber'],dtype=int)[event_i:event_f]

    else:
        act_evt_num = None

    if unix_info == True:
        unixTime = np.asarray(evtTree['unixTime'],dtype=int)[event_i:event_f]
        unixTimeUs = np.asarray(evtTree['unixTimeUs'],dtype=int)[event_i:event_f]
        unix_time = np.array([unixTime,unixTimeUs])
        del unixTime, unixTimeUs
    else:
        unix_time = None

    if read_win_info == True:
        read_win = np.asarray(evtTree['event/numReadoutBlocks'],dtype=int)[event_i:event_f]
    else:
        read_win = None

    time_stamp = np.asarray(evtTree['event/timeStamp'],dtype=int)[event_i:event_f]
    triggerInfo = np.asarray(evtTree['event/triggerInfo[4]'],dtype=int)[event_i:event_f]
   
    if trig_info == True: 
        ch_index_bit = 1<<ch_index
        trig_chs = np.repeat(triggerInfo[:,0][np.newaxis, event_i:event_f], len(ch_index), axis=0) & ch_index_bit[:,np.newaxis]
        del ch_index_bit
        trig_chs[trig_chs != 0] = 1
        trig_chs = trig_chs.astype(float)
        trig_chs[trig_chs == 0] = np.nan
        trig_chs *= ch_index[:,np.newaxis]
    else:
        trig_chs = None

    cal_index = np.where(np.abs(time_stamp - cal_length(Station)) < 1e4)[0]
    soft_index = np.where(triggerInfo[:,2] == 1)[0]
    del triggerInfo
    trig_num = np.zeros((event_f - event_i),dtype=int)
    trig_num[cal_index] = 1
    trig_num[soft_index] = 2
    rf_index = np.where(trig_num == 0)[0]

    if time_stamp_info == True:
        pass
    else:
        time_stamp = None
    
    if trig_set == 0:
        act_evt_num, read_win, time_stamp, unix_time, trig_chs = element_finder(rf_index, act_evt_num, read_win, time_stamp, unix_time, trig_chs)
        evt_num = rf_index
        trig_num = np.array([trig_set])
    elif trig_set == 1:
        act_evt_num, read_win, time_stamp, unix_time, trig_chs = element_finder(cal_index, act_evt_num, read_win, time_stamp, unix_time, trig_chs) 
        evt_num = cal_index
        trig_num = np.array([trig_set])
    elif trig_set == 2:
        act_evt_num, read_win, time_stamp, unix_time, trig_chs = element_finder(soft_index, act_evt_num, read_win, time_stamp, unix_time,trig_chs)
        evt_num = soft_index       
        trig_num = np.array([trig_set])        
    elif trig_set == 'all':
        pass

    # quality cut
    qual_type = 5
    qual_num_tot = np.full((qual_type, evt_num),1,dtype=int) #0 five event, 1 block gap, 2 timing, 3 few block, 4 offset
    
    # first five event
    qual_num_tot[0] = first_five_event_error_uproot(Station, unix_time[0], act_evt_num, qual_num_tot[0])    
    
    # block gap
    irsBlockNumber = np.asarray(evtTree['event/blockVec/blockVec.irsBlockNumber'])
    qual_num_tot[1] = block_gap_error_uproot(irsBlockNumber, qual_num[1])

    # quality sum
    qual_num = np.nansum(qual_num_tot, axis = 0)
    qual_num[qual_num < 2] = 0
    qual_num[qual_num >= 2] = 1

    del file, evtTree, rf_index, cal_index, soft_index, irsBlockNumber

    return evt_num, act_evt_num, unix_time, trig_num, trig_chs, time_stamp, read_win, qual_num_tot, qual_num, hasKeyInFileError














































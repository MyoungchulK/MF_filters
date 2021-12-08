import os, sys
import numpy as np
import h5py
from tqdm import tqdm

# custom lib
#from tools.ara_root import ara_root_lib
#from tools.ara_root import ara_raw_to_qual
#from tools.ara_root import useful_evt_maker
#from tools.ara_root import trig_checker
#from tools.ara_root import qual_checker
from tools.wf import station_pad
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.antenna import antenna_info

def wf_collector_dat_v3(Data, Ped, Station,
                trig_set = 'all',
                qual_set = 'all',
                wf_dat = True,
                wf_len = False,
                evt_info = False,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width(),
                chunk_size = 5000,
                chunk_i = None):

    print('Collecting data WF starts!')

    import psutil
    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import trig_checker
    from tools.ara_root import qual_checker

    mem_u = []
    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('start:',mem_mb)
    mem_u.append(mem_mb)

    # import root and ara root lib
    R = ara_root_lib()

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('lib loaded:',mem_mb)
    mem_u.append(mem_mb)

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q = ara_raw_to_qual(R, Data, Ped, Station)
    del Data, Ped

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('data loaded:',mem_mb)
    mem_u.append(mem_mb)

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]

    # output list
    wf_all = []
    wf_len_all = []
    evt_num = []
    trig_num = []
    qual_num = []

    # chunk
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

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('time and evt range set:',mem_mb)
    mem_u.append(mem_mb)

    # loop over the events
    for event in tqdm(range(event_i,event_f)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # trigger filtering
        if trig_set == 'all':
            trig_num.append(trig_checker(rawEvt))
        else:
            if trig_checker(rawEvt) != trig_set:
                continue
            else:
                pass

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after trig, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # quality cut
        if qual_set == 'all':
            qual_num.append(qual_checker(q, usefulEvent))
        else:
            if qual_checker(q, usefulEvent) != qual_set:
                del usefulEvent
                continue
            else:
                pass

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after qual, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # make padded wf and interpolated wf length
        if wf_dat == True:
            if wf_len == True:
                ant_arr, int_time_len = station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)
                wf_all.append(ant_arr)
                wf_len_all.append(int_time_len)
                del ant_arr, int_time_len
            else:
                wf_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[0])
        else:
            if wf_len == True:
                wf_len_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[1])
            else:
                pass

        if evt_info == True:
            evt_num.append(event)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after store to list, {event}:',mem_mb)
        mem_u.append(mem_mb)

        #usefulEvent.Delete()
        del usefulEvent

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after remove useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

    del time_pad_len, time_pad_i, time_pad_f, event_i, event_f
    del R, file, evtTree, rawEvt, num_evts, cal, q

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    if wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    if evt_info == True:
        evt_num = np.asarray(evt_num)
    if trig_set == 'all':
        trig_num = np.asarray(trig_num)
    if qual_set == 'all':
        qual_num = np.asarray(qual_num)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('after convert list to arr:',mem_mb)
    mem_u.append(mem_mb)

    mem_u = np.asarray(mem_u)

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num, trig_num, qual_num, final_chunk, mem_u


def wf_collector_dat(Data, Ped, Station,
                trig_set = 'all', 
                qual_set = 'all', 
                wf_dat = True, 
                wf_len = False, 
                evt_info = False,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width(),
                chunk_size = 5000,
                chunk_i = None): 

    print('Collecting data WF starts!')

    import psutil
    from tools.ara_root import ara_root_lib
    from tools.ara_root import ara_raw_to_qual
    from tools.ara_root import useful_evt_maker
    from tools.ara_root import trig_checker
    from tools.ara_root import qual_checker

    mem_u = []
    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('start:',mem_mb)
    mem_u.append(mem_mb)

    # import root and ara root lib
    R = ara_root_lib()

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('lib loaded:',mem_mb)
    mem_u.append(mem_mb)

    # load raw data and process to general quality cut by araroot
    file, evtTree, rawEvt, num_evts, cal, q = ara_raw_to_qual(R, Data, Ped, Station)
    del Data, Ped

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('data loaded:',mem_mb)
    mem_u.append(mem_mb)

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]

    # output list
    wf_all = []
    wf_len_all = []
    evt_num = []
    trig_num = []
    qual_num = []

    # chunk
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

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('time and evt range set:',mem_mb)
    mem_u.append(mem_mb)

    # loop over the events
    for event in tqdm(range(event_i,event_f)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # trigger filtering
        if trig_set == 'all':
            trig_num.append(trig_checker(rawEvt))
        else:
            if trig_checker(rawEvt) != trig_set:
                continue
            else:
                pass
   
        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after trig, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # quality cut
        if qual_set == 'all':
            qual_num.append(qual_checker(q, usefulEvent))
        else:
            if qual_checker(q, usefulEvent) != qual_set:
                del usefulEvent
                continue
            else:
                pass

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after qual, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # make padded wf and interpolated wf length
        if wf_dat == True:
            if wf_len == True:
                ant_arr, int_time_len = station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)
                wf_all.append(ant_arr)
                wf_len_all.append(int_time_len)
                del ant_arr, int_time_len
            else:
                wf_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[0])
        else:
            if wf_len == True:
                wf_len_all.append(station_pad(usefulEvent, num_Ants, dt_ns, time_pad_len, time_pad_i, time_pad_f)[1])
            else:
                pass 
  
        if evt_info == True:
            evt_num.append(event)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after store to list, {event}:',mem_mb)
        mem_u.append(mem_mb)

        usefulEvent.Delete()
        del usefulEvent

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after remove useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

    del time_pad_len, time_pad_i, time_pad_f, event_i, event_f
    del R, file, evtTree, rawEvt, num_evts, cal, q

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    if wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    if evt_info == True:
        evt_num = np.asarray(evt_num)
    if trig_set == 'all':
        trig_num = np.asarray(trig_num)
    if qual_set == 'all':
        qual_num = np.asarray(qual_num)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('after convert list to arr:',mem_mb)
    mem_u.append(mem_mb)

    mem_u = np.asarray(mem_u)

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num, trig_num, qual_num, final_chunk, mem_u

import ROOT # general cern root lib
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so") # cvmfs ara lib
from tools.array import arr_2d
from tools.array import arr_1d
from scipy.interpolate import Akima1DInterpolator
import psutil

#Akima interpolation from python Akima1DInterpolator library
def akima_interp(raw_t, raw_v, dt):

    # set the initial time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[0] - int(raw_t[0]) > dt:
        int_ti = np.ceil(raw_t[0]) # if value is x.501...~x.999..., ceil to x+1.0
    elif raw_t[0] - int(raw_t[0]) < dt:
        int_ti = int(raw_t[0]) + dt # if value is x.001...~x.499..., ceil to x.5
    else:
        int_ti = raw_t[0] # if value is x.5 exact, leave it

    # set the final time bin to x.0ns or x.5ns at the inside of the original range
    if raw_t[-1] - int(raw_t[-1]) > dt:
        int_tf = int(raw_t[-1]) + dt # if value is x.501...~x.999..., floor to x.5
    elif raw_t[-1] - int(raw_t[-1]) < dt:
        int_tf = np.floor(raw_t[-1]) # # if value is x.001...~x.499..., ceil to x.0
    else:
        int_tf = raw_t[-1] # if value is x.5 exact, leave it

    # set time range by dt
    int_t = np.arange(int_ti, int_tf+dt, dt)

    # akima interpolation!
    akima = Akima1DInterpolator(raw_t, raw_v)
    del raw_t, raw_v

    return int_ti, int_tf, akima(int_t), len(int_t)

def wf_collector_dat_v2(Data, Ped, Station,
                trig_set = 'all',
                qual_set = 'all',
                wf_dat = True,
                wf_len = False,
                evt_info = False,
                num_Ants = antenna_info()[2],
                dt_ns = interpolation_bin_width(),
                chunk_size = 5000,
                chunk_i = None):

    print('Collecting data WF starts!')

    mem_u = []
    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('start:',mem_mb)
    mem_u.append(mem_mb)

    # open a data file
    file = ROOT.TFile.Open(Data)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('file opened:',mem_mb)
    mem_u.append(mem_mb)

    # load in the event free for this file
    evtTree = file.Get("eventTree")

    #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    #print('tree loaed:',mem_mb)
    #mem_u.append(mem_mb)

    # set the tree address to access our raw data type
    rawEvt = ROOT.RawAtriStationEvent()
    evtTree.SetBranchAddress("event",ROOT.AddressOf(rawEvt))

    #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    #print('address set:',mem_mb)
    #mem_u.append(mem_mb)

    # get the number of entries in this file
    num_evts = int(evtTree.GetEntries())
    print('total events:', num_evts)

    #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    #print('total evt checked:',mem_mb)
    #mem_u.append(mem_mb)

    # open a pedestal file
    cal = ROOT.AraEventCalibrator.Instance()
    cal.setAtriPedFile(Ped, Station)

    #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    #print('cal set:',mem_mb)
    #mem_u.append(mem_mb)

    # open general quilty cut
    q = ROOT.AraQualCuts.Instance()
    del Data, Ped

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('qual set:',mem_mb)
    mem_u.append(mem_mb)

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(p_dt = dt_ns)[1:]

    # output list
    wf_all = []
    wf_len_all = []
    evt_num = []
    trig_num = []
    qual_num = []

    # chunk
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

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('time and evt range set:',mem_mb)
    mem_u.append(mem_mb)

    # loop over the events
    for event in tqdm(range(event_i,event_f)):

        # get the event
        evtTree.GetEntry(event)

        #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        #print(f'after evttree, {event}:',mem_mb)
        #mem_u.append(mem_mb)

        # make a useful event
        usefulEvent = ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # trigger filtering
        trig = -1#'Unknown'
        if rawEvt.isSoftwareTrigger() == 1:
            trig = 2#'Soft'
        elif rawEvt.isCalpulserEvent() == 1:
            trig = 1#'Cal'
        elif rawEvt.isSoftwareTrigger() == 0 and rawEvt.isCalpulserEvent() == 0:
            trig = 0#'RF'
        else:
            pass

        if trig_set == 'all':
            trig_num.append(trig)
        else:
            if trig != trig_set:
                continue
            else:
                pass
        del trig

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after trig, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # quality cut
        qual = -1#'Unknown'
        if q.isGoodEvent(usefulEvent) == 1:
            qual = 1#'Pass'
        elif q.isGoodEvent(usefulEvent) == 0:
            qual = 0#'Cut'
        else:
            pass      

        if qual_set == 'all':
            qual_num.append(qual)
        else:
            if qual != qual_set:
                del usefulEvent
                continue
            else:
                pass
        del qual

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after qual, {event}:',mem_mb)
        mem_u.append(mem_mb)

        # make padded wf and interpolated wf length
        if wf_dat == True:
            ant_arr = arr_2d(time_pad_len, num_Ants, 0, float)
        else:
            pass
        if wf_len == True:
                i_time_len = arr_1d(num_Ants, np.nan, int)
        else:
            pass

        #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        #print(f'after arr set, {event}:',mem_mb)
        #mem_u.append(mem_mb)

        # loop over the antennas
        for ant in range(num_Ants):            

            # TGraph
            gr = usefulEvent.getGraphFromRFChan(ant)

            raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
            raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

            # TGraph to interpolated wf
            time_i, time_f, volt_i, i_t_len = akima_interp(raw_t, raw_v, dt_ns)
            if wf_len == True:
                i_time_len[ant] = i_t_len
            else:
                pass

            # put int wf into pad
            if wf_dat == True:
                try:
                    ant_arr[int((time_i - time_pad_i) / dt_ns):-int((time_pad_f - time_f) / dt_ns), ant] = volt_i
                except ValueError:
                    print(f'Too long!, evt:{event}, len:{i_t_len}')
            else:
                pass

            gr.Delete()
            del gr, raw_t, raw_v, time_i, time_f, volt_i, i_t_len

        #mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        #print(f'after store to arr, {event}:',mem_mb)
        #mem_u.append(mem_mb)

        if wf_dat == True:
            wf_all.append(ant_arr/1e3)
        else:
            pass

        if wf_len == True:
            wf_len_all.append(i_time_len)
        else:
            pass

        if evt_info == True:
            evt_num.append(event)
        else:
            pass

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after store to list, {event}:',mem_mb)
        mem_u.append(mem_mb)

        usefulEvent.Delete()
        del usefulEvent

        mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
        print(f'after remove useful, {event}:',mem_mb)
        mem_u.append(mem_mb)

    del time_pad_len, time_pad_i, time_pad_f, event_i, event_f
    del file, evtTree, rawEvt, num_evts, cal, q

    # convert list to numpy array
    if wf_dat == True:
        wf_all = np.transpose(np.asarray(wf_all),(1,2,0))
    if wf_len == True:
        wf_len_all = np.transpose(np.asarray(wf_len_all),(1,0))
    if evt_info == True:
        evt_num = np.asarray(evt_num)
    if trig_set == 'all':
        trig_num = np.asarray(trig_num)
    if qual_set == 'all':
        qual_num = np.asarray(qual_num)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('after convert list to arr:',mem_mb)
    mem_u.append(mem_mb)

    mem_u = np.asarray(mem_u)

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num, trig_num, qual_num, final_chunk, mem_u

def wf_collector_sim(Data,
		wf_len = False, evt_info = False):

    print('Collecting sim WF starts!')

    # load sim wf
    hf = h5py.File(Data, 'r')    

    # load info
    wf_all = hf['volt'][:]

    #wf_all[:400,:8] = 0.
    #wf_all[400+275:,:8] = 0.
    #wf_all[:400,8:] = 0.
    #wf_all[400+235:,8:] = 0.

    wf_all /= 1e3 # mV to V

    if wf_len == True:
        wf_len_all = np.full((wf_all.shape[1],wf_all.shape[2]),np.nan)

        for evt in tqdm(range(wf_all.shape[2])):
            for ant in range(wf_all.shape[1]):
                wf_len_all[ant,evt] = np.count_nonzero(wf_all[:,ant,evt])
    else:
        wf_len_all = []

    if evt_info == True:
        evt_num = hf['inu_thrown'][:].astype(int)
    else:
        evt_num = []

    del hf

    print('WF collecting is done!')

    #output
    return wf_all, wf_len_all, evt_num













import numpy as np
import os
import h5py
import ROOT

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

class ara_geom_loader:

    def __init__(self, st, yrs):

        # create a geomtool
        self.geomTool = ROOT.AraGeomTool.Instance()
        self.rf_ch_idx = np.arange(16) # need to changed
        self.num_rf_chs = len(self.rf_ch_idx)
        self.st = st
        self.yrs = yrs

    def get_ele_ch_idx(self):

        ele_ch_idx = np.full((self.num_rf_chs), 0, dtype = int)
        for ant in range(self.num_rf_chs):
            ele_ch_idx[ant] = self.geomTool.getStationInfo(self.st, self.yrs).getAntennaInfo(ant).daqChanNum
        print('electronic channel:',ele_ch_idx)
        return ele_ch_idx

    def get_pol_ch_idx(self):

        pol_ch_idx = np.full((self.num_rf_chs), 0, dtype = int)
        for ant in range(self.num_rf_chs):
            pol_ch_idx[ant] = self.geomTool.getStationInfo(self.st, self.yrs).getAntennaInfo(ant).polType
        print('polarization type:',pol_ch_idx)
        return pol_ch_idx

    def get_trig_ch_idx(self):

        trig_ch_idx = np.full((self.num_rf_chs), 0, dtype = int)
        for ant in range(self.num_rf_chs):
            trig_ch_idx[ant] = self.geomTool.getStationInfo(self.st, self.yrs).getAntennaInfo(ant).getTrigChan()
        print('trigger channel:',trig_ch_idx)
        return trig_ch_idx

    def get_ant_xyz(self):

        ant_xyz = np.full((self.num_rf_chs), np.nan, dtype = float)
        for ant in range(self.num_rf_chs):
            ant_xyz[ant] = self.geomTool.getStationInfo(self.st, self.yrs).getAntennaInfo(ant).antLocation
        return ant_xyz

class qual_loader:

    def __init__(self):

        # open general quilty cut
        self.q = ROOT.AraQualCuts.Instance()

    def get_cut_result(self, usefulEvt):

        # default qual tag
        qual = -1#'Unknown'

        # qual tag check
        if self.q.isGoodEvent(usefulEvt) == 1:
            qual = 0#'Pass'
        elif self.q.isGoodEvent(usefulEvt) == 0:
            qual = 1#'Cut'
        else:
            pass

        return qual

class ara_root_loader:

    def __init__(self, data, ped, st):

        # open a data file
        self.file = ROOT.TFile.Open(data)

        # load in the event free for this file
        self.evtTree = self.file.Get("eventTree")
            
        # set the tree address to access our raw data type
        self.rawEvt = ROOT.RawAtriStationEvent()
        self.evtTree.SetBranchAddress("event", ROOT.AddressOf(self.rawEvt))

        # get the number of entries in this file
        self.num_evts = int(self.evtTree.GetEntries())
        print('total events:', self.num_evts)

        # open a pedestal file
        self.cal = ROOT.AraEventCalibrator.Instance()
        self.cal.setAtriPedFile(ped, st)

    def get_entry(self, evt):
   
        # get the event
        self.evtTree.GetEntry(evt)

    def get_useful_evt(self):

        self.usefulEvt = ROOT.UsefulAtriStationEvent(self.rawEvt, ROOT.AraCalType.kLatestCalib)

    def get_rf_ch_wf(self, ant):

        gr = self.usefulEvt.getGraphFromRFChan(ant)
        raw_t = np.frombuffer(gr.GetX(),dtype=float,count=-1)
        raw_v = np.frombuffer(gr.GetY(),dtype=float,count=-1)

        return raw_t, raw_v

    def del_TGraph(self):

        self.gr.Delete()
        del self.gr

    def del_usefulEvt(self):

        self.usefulEvt.Delete()
        del self.usefulEvt

    def get_sub_info(self):

        self.evt_num = self.rawEvt.eventNumber
        self.unix_time = self.rawEvt.unixTime
        self.unix_time_us = self.rawEvt.unixTimeUs
        self.read_win = self.rawEvt.numReadoutBlocks
        self.time_stamp = self.rawEvt.timeStamp

    def get_trig_type(self):

        trig_type = -1
        if self.rawEvt.isSoftwareTrigger() == 1:
            trig_type = 2
        elif self.rawEvt.isCalpulserEvent() == 1:
            trig_type = 1
        elif self.rawEvt.isSoftwareTrigger() == 0 and self.rawEvt.isCalpulserEvent() == 0:
            trig_type = 0
        else:
            pass

        return trig_type

    def get_trig_ant(self, trig_ch_idx):

        trig_ant = np.full((len(trig_ch_idx)), 0, dtype = int)
        for ant in range(len(trig_ch_idx)):
            if self.rawEvt.isTriggerChanHigh(int(trig_ch_idx[ant])) == True:
                trig_ant[ant] = 1
            else:
                pass

        return trig_ant

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


































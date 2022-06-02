import numpy as np
import os
import ROOT

# custom lib
from tools.ara_constant import ara_const

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

ara_const = ara_const()
num_useful_chs = ara_const.USEFUL_CHAN_PER_STATION
num_ddas = ara_const.DDA_PER_ATRI

class ara_geom_loader:

    def __init__(self, st, yrs, verbose = False):

        # create a geomtool
        self.geomTool = ROOT.AraGeomTool.Instance()
        self.st_info = self.geomTool.getStationInfo(st, yrs)
        self.verbose = verbose

    def get_ele_ch_idx(self):

        ele_ch_idx = np.full((num_useful_chs), 0, dtype = int)
        for ant in range(num_useful_chs):
            ele_ch_idx[ant] = self.st_info.getAntennaInfo(ant).daqChanNum
        if self.verbose:
            print('electronic channel:',ele_ch_idx)
        return ele_ch_idx

    def get_pol_ch_idx(self):

        pol_ch_idx = np.full((num_useful_chs), 0, dtype = int)
        for ant in range(num_useful_chs):
            pol_ch_idx[ant] = self.st_info.getAntennaInfo(ant).polType
        if self.verbose:
            print('polarization type:',pol_ch_idx)
        return pol_ch_idx

    def get_trig_ch_idx(self):

        trig_ch_idx = np.full((num_useful_chs), 0, dtype = int)
        for ant in range(num_useful_chs):
            trig_ch_idx[ant] = self.st_info.getAntennaInfo(ant).getTrigChan()
        if self.verbose: 
            print('trigger channel:',trig_ch_idx)
        return trig_ch_idx

    def get_ant_xyz(self):

        ant_xyz = np.full((3, num_useful_chs), np.nan, dtype = float)
        for ant in range(num_useful_chs):
            ant_xyz[0, ant] = self.st_info.getAntennaInfo(ant).antLocation[0]
            ant_xyz[1, ant] = self.st_info.getAntennaInfo(ant).antLocation[1]
            ant_xyz[2, ant] = self.st_info.getAntennaInfo(ant).antLocation[2]
        if self.verbose:
            print('antenna location:',ant_xyz)
        return ant_xyz

    def get_cable_delay(self):
        cable_delay = np.full((num_useful_chs), np.nan, dtype = float)
        for ant in range(num_useful_chs):
            cable_delay[ant] = self.st_info.getAntennaInfo(ant).getCableDelay()
        if self.verbose:
            print('cable delay:',cable_delay)
        return cable_delay

class ara_root_loader:

    def __init__(self, data, ped, st, yrs):

        #geom info
        self.ara_geom = ara_geom_loader(st, yrs)

        # open a data file
        self.file = ROOT.TFile.Open(data)

        # load in the event free for this file
        self.evtTree = self.file.Get("eventTree")
            
        # set the tree address to access our raw data type
        self.rawEvt = ROOT.RawAtriStationEvent()
        self.evtTree.SetBranchAddress("event", ROOT.AddressOf(self.rawEvt))

        # get the number of entries in this file
        self.num_evts = int(self.evtTree.GetEntries())
        self.entry_num = np.arange(self.num_evts)
        print('total events:', self.num_evts)

        # open a pedestal file
        self.cal = ROOT.AraEventCalibrator.Instance()
        self.cal.setAtriPedFile(ped, st)

        #calibration mode
        self.cal_type = ROOT.AraCalType

    def get_entry(self, evt):
   
        # get the event
        self.evtTree.GetEntry(evt)

    def get_useful_evt(self, cal_mode):

        self.usefulEvt = ROOT.UsefulAtriStationEvent(self.rawEvt, cal_mode)

    def get_rf_ch_wf(self, ant):

        self.gr = self.usefulEvt.getGraphFromRFChan(ant)
        raw_t = np.frombuffer(self.gr.GetX(),dtype=float,count=-1)
        raw_v = np.frombuffer(self.gr.GetY(),dtype=float,count=-1)

        return raw_t, raw_v

    def get_ele_ch_wf(self, ch):

        self.gr = self.usefulEvt.getGraphFromElecChan(ch)
        raw_t = np.frombuffer(self.gr.GetX(),dtype=float,count=-1)
        raw_v = np.frombuffer(self.gr.GetY(),dtype=float,count=-1)

        return raw_t, raw_v

    def del_TGraph(self):

        self.gr.Delete()
        del self.gr

    def del_usefulEvt(self):

        #self.usefulEvt.Delete()
        del self.usefulEvt

    def get_sub_info(self):

        self.evt_num = self.rawEvt.eventNumber
        self.unix_time = self.rawEvt.unixTime
        #self.unix_time_us = self.rawEvt.unixTimeUs
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

    def get_block_idx(self, trim_1st_blk = False, modulo_2 = False):

        remove_1_blk = int(trim_1st_blk)
        block_vec = self.rawEvt.blockVec

        blk_len = block_vec.size()//num_ddas - remove_1_blk
        blk_idx_arr = np.full((blk_len), 0, dtype=int)
        for iblk in range(blk_len):
            blk_idx_arr[iblk] = block_vec[num_ddas * (iblk + remove_1_blk)].getBlock()

        if modulo_2:
            blk_idx_arr = blk_idx_arr%2
        del blk_len, remove_1_blk, block_vec

        return blk_idx_arr

    def get_qual_cut(self):

        # open general quilty cut
        self.q = ROOT.AraQualCuts.Instance()

    def get_qual_cut_result(self):

        # default qual tag
        qual = -1#'Unknown'

        # qual tag check
        if self.q.isGoodEvent(self.usefulEvt) == 1:
            qual = 0#'Pass'
        elif self.q.isGoodEvent(self.usefulEvt) == 0:
            qual = 1#'Cut'
        else:
            pass

        return qual

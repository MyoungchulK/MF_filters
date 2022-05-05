import numpy as np
import os
import h5py
import uproot
import ROOT
from tqdm import tqdm
from datetime import datetime
import re

# custom lib
from tools.ara_constant import ara_const

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
ROOT.gSystem.Load('/cvmfs/ara.opensciencegrid.org/trunk/centos7/misc_build/lib/libRootFftwWrapper.so')

ara_const = ara_const()
num_useful_chs = ara_const.USEFUL_CHAN_PER_STATION
num_ddas = ara_const.DDA_PER_ATRI
num_samples = ara_const.SAMPLES_PER_BLOCK
num_blocks = ara_const.BLOCKS_PER_DDA
num_buffers = ara_const.SAMPLES_PER_DDA
num_eles = ara_const.CHANNELS_PER_ATRI
num_chs = ara_const.RFCHAN_PER_DDA

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

class ara_uproot_loader:

    def __init__(self, data):

        file = uproot.open(data)

        self.hasKeyInFileError = False
        try:
            self.evtTree = file['eventTree']
            st_arr = np.asarray(self.evtTree['event/RawAraStationEvent/RawAraGenericHeader/stationId'],dtype=int)
            self.station_id = st_arr[0]
            self.num_evts = len(st_arr)
            self.entry_num = np.arange(len(st_arr))
            self.evt_num = np.asarray(self.evtTree['event/eventNumber'],dtype=int)
            run_str = re.findall(r'\d+', data[-11:-5])[0]
            self.run = int(run_str)
            self.year = self.get_year()
            print('total events:', self.num_evts)
            del st_arr, run_str
        except uproot.exceptions.KeyInFileError:
            self.hasKeyInFileError = True
            print('File is currupted!')

    def get_year(self):

        first_unix_time = np.asarray(self.evtTree['event/unixTime'],dtype=int)[0]
        yyyymmdd_str = datetime.fromtimestamp(first_unix_time)
        yyyymmdd = yyyymmdd_str.strftime('%Y%m%d%H%M%S')
        year = int(yyyymmdd[:4])
        del yyyymmdd_str, yyyymmdd, first_unix_time

        return year

    def get_sub_info(self):

        self.unix_time = np.asarray(self.evtTree['event/unixTime'],dtype=int)
        self.read_win = np.asarray(self.evtTree['event/numReadoutBlocks'],dtype=int)
        self.irs_block_number = np.asarray(self.evtTree['event/blockVec/blockVec.irsBlockNumber']) & 0x1ff
        self.channel_mask = np.asarray(self.evtTree['event/blockVec/blockVec.channelMask'])
        self.pps_number = np.asarray(self.evtTree['event/ppsNumber'],dtype=int)
        self.time_stamp = np.asarray(self.evtTree['event/timeStamp'],dtype=int)        

        #self.trigger_blk = np.asarray(self.evtTree['event/triggerBlock[4]'],dtype=int)
        #self.unix_time_us = np.asarray(self.evtTree['event/unixTimeUs'],dtype=int)

    def get_trig_type(self):

        trigger_info = np.asarray(self.evtTree['event/triggerInfo[4]'],dtype=int)

        pulserTime = np.array([254,245,245,400,400], dtype = int)

        trig_type = np.full(len(self.time_stamp), 0, dtype = int)
        if self.station_id == 100:
            print('ARA Staton 1 ID!!!!!!!!!!!')
            self.station_id = 1
        trig_type[np.abs(self.time_stamp - pulserTime[self.station_id - 1]) < 1e4] = 1
        trig_type[trigger_info[:,2] == 1] = 2
        del pulserTime, trigger_info

        return trig_type

    def get_trig_ant(self, trig_ch_idx):

        trigger_info = np.asarray(self.evtTree['event/triggerInfo[4]'],dtype=int)

        trig_ch_idx_bit = 1 << trig_ch_idx
        trig_ant = np.repeat(trigger_info[:,0][np.newaxis, :], len(trig_ch_idx), axis=0) & trig_ch_idx_bit[:,np.newaxis]
        trig_ant[trig_ant != 0] = 1
        trig_ant = trig_ant.astype(int)
        del trig_ch_idx_bit, trigger_info

        return trig_ant

    def get_block_idx(self, evt, trim_1st_blk = False, modulo_2 = False):

        remove_1_blk = int(trim_1st_blk)

        blk_idx_arr = np.asarray(self.irs_block_number[evt][remove_1_blk*num_ddas::num_ddas], dtype = int)
        blk_idx_len = len(blk_idx_arr)

        if modulo_2:
            blk_idx_arr = blk_idx_arr%2
        del remove_1_blk

        return blk_idx_arr, blk_idx_len

    def get_reset_pps_number(self, use_evt_num_sort = False):

        pps_reset = np.copy(self.pps_number)
        if use_evt_num_sort:
            evt_sort_idx = np.argsort(self.evt_num)
            pps_reset = pps_reset[evt_sort_idx]
            del evt_sort_idx

        pps_reset_idx = np.where(np.diff(pps_reset) < -55000)[0]
        if len(pps_reset_idx) > 0:
            pps_limit = 65536
            pps_reset[pps_reset_idx[0]+1:] += pps_limit
            del pps_limit
        del pps_reset_idx

        return pps_reset

    def get_event_rate(self, use_pps = False):

        evt_sort_idx = np.argsort(self.evt_num)
        trig_type = self.get_trig_type()
        trig_sort = trig_type[evt_sort_idx]
        del trig_type

        if use_pps:
            time_sort = self.get_reset_pps_number(use_evt_num_sort = True)
        else:
            time_sort = self.unix_time[evt_sort_idx]
        time_unique = np.sort(np.unique(time_sort))
        del evt_sort_idx

        sec_to_min = 60
        time_bins = np.arange(np.nanmin(time_unique), np.nanmax(time_unique)+1, sec_to_min, dtype = int)
        time_bins = time_bins.astype(float)
        time_bins -= 0.5
        time_bins = np.append(time_bins, np.nanmax(time_unique) + 0.5)
        time_bin_center = (time_bins[1:] + time_bins[:-1]) / 2
        num_secs = np.diff(time_bins).astype(int)
        del sec_to_min, time_unique

        evt_rate = np.histogram(time_sort, bins = time_bins)[0] / num_secs
        rf_evt_rate = np.histogram(time_sort[trig_sort == 0], bins = time_bins)[0] / num_secs
        cal_evt_rate = np.histogram(time_sort[trig_sort == 1], bins = time_bins)[0] / num_secs
        soft_evt_rate = np.histogram(time_sort[trig_sort == 2], bins = time_bins)[0] / num_secs
        del time_sort, trig_sort

        return time_bins, time_bin_center, num_secs, evt_rate, rf_evt_rate, cal_evt_rate, soft_evt_rate

class ara_sensorHk_uproot_loader:

    def __init__(self, data):

        self.empty_file_error = False
        if data is None:
            self.unix_time = np.full((1), np.nan, dtype = float)
            self.empty_file_error = True
            print('There is no sensorHk file!')
            return

        try:
            file = uproot.open(data)
        except ValueError:
            self.unix_time = np.full((1), np.nan, dtype = float)
            self.empty_file_error = True
            print('sensorHk is empty!')
            return

        self.hasKeyInFileError = False
        try:
            self.evtTree = file['sensorHkTree']
            st_arr = np.asarray(self.evtTree['sensorHk/RawAraGenericHeader/stationId'],dtype=int)
            self.station_id = st_arr[0]
            self.num_sensors = len(st_arr)
            self.sensor_entry_num = np.arange(len(st_arr))
            run_str = re.findall(r'\d+', data[-11:-5])[0]
            self.run = int(run_str)
            del st_arr, run_str
        except uproot.exceptions.KeyInFileError:
            self.hasKeyInFileError = True
            print('File is currupted!')

    def get_sub_info(self):

        self.unix_time = np.asarray(self.evtTree['sensorHk/unixTime'],dtype=int)
        self.unix_time_us = np.asarray(self.evtTree['sensorHk/unixTimeUs'],dtype=int)
        self.atri_voltage = np.asarray(self.evtTree['sensorHk/atriVoltage'],dtype=int)
        self.atri_current = np.asarray(self.evtTree['sensorHk/atriVoltage'],dtype=int)
        self.dda_temperature = np.asarray(self.evtTree['sensorHk/ddaTemp[4]'])#,dtype=int)
        self.tda_temperature = np.asarray(self.evtTree['sensorHk/tdaTemp[4]'])#,dtype=int)
        self.dda_volt_curr = np.asarray(self.evtTree['sensorHk/ddaVoltageCurrent[4]'],dtype=int)
        self.tda_volt_curr = np.asarray(self.evtTree['sensorHk/tdaVoltageCurrent[4]'],dtype=int)
        self.verId = np.asarray(self.evtTree['sensorHk/RawAraGenericHeader/verId'],dtype=int)
        self.subVerId = np.asarray(self.evtTree['sensorHk/RawAraGenericHeader/subVerId'],dtype=int)
        self.softVerMajor = np.asarray(self.evtTree['sensorHk/RawAraGenericHeader/softVerMajor'],dtype=int)
        self.softVerMinor = np.asarray(self.evtTree['sensorHk/RawAraGenericHeader/softVerMinor'],dtype=int)

        yyyymmdd_str = datetime.fromtimestamp(self.unix_time[0])
        yyyymmdd = yyyymmdd_str.strftime('%Y%m%d%H%M%S')
        self.year = int(yyyymmdd[:4])
        del yyyymmdd_str, yyyymmdd

    def get_voltage(self, volt_curr):

        VoltageADC = (volt_curr & 0xff) << 4
        VoltageADC = VoltageADC | (volt_curr & 0xf00000) >> 20

        volt = (6.65 / 4096) * VoltageADC
        del VoltageADC

        return volt

    def get_current(self, volt_curr, diff = 0.1):
        
        CurrentADC = (volt_curr & 0x00ff00) >> 4
        CurrentADC = CurrentADC | (volt_curr & 0x0f0000) >> 16

        curr = CurrentADC * (0.10584 / 4096) / diff
        del CurrentADC

        return curr

    def get_temperature(self, temperature):

        msb = temperature << 4
        lsb = temperature >> 12
        tempADC = msb | lsb

        temp = np.full(temperature.shape,np.nan,dtype = float)

        for poi in range(self.num_sensors):
            for dda in range(num_ddas):

                isNegative = False
                if msb[poi, dda] & 0x0800:
                    isNegative=True

                if isNegative:
                    tempADC[poi, dda] = ~tempADC[poi, dda] + 1
                    tempADC[poi, dda] = tempADC[poi, dda] & 0x0fff

                temp[poi, dda] = tempADC[poi, dda] * 0.0625
                if isNegative:
                    temp[poi, dda] *= -1
                del isNegative
        del msb, lsb, tempADC

        return temp

    def get_atri_voltage_current(self, volt_var = 0.0605, curr_var = 0.0755):

        atri_volt = np.full(self.atri_voltage.shape, np.nan, dtype = float)       
        atri_curr = np.copy(atri_volt)

        for poi in range(self.num_sensors):
            if (self.verId[poi] > 4 or (self.verId[poi] == 4 and self.subVerId[poi] > 1)) or (self.softVerMajor[poi] > 3 or (self.softVerMajor[poi] == 3 and self.softVerMinor[poi] > 11)):
                atri_volt[poi] = self.atri_voltage[poi] * volt_var
                atri_curr[poi] = self.atri_current[poi] * curr_var
            else:
                atri_volt[poi] = self.atri_current[poi] * volt_var 
                atri_curr[poi] = self.atri_voltage[poi] * curr_var

        return atri_volt, atri_curr

    def get_daq_sensor_info(self):

        if self.empty_file_error:
            atri_volt = np.full((1), np.nan, dtype = float)
            atri_curr = np.copy(atri_volt) 
            dda_volt = np.full((1, num_ddas), np.nan, dtype = float)
            dda_curr = np.copy(dda_volt)
            dda_temp = np.copy(dda_volt)
            tda_volt = np.copy(dda_volt)
            tda_curr = np.copy(dda_volt)
            tda_temp = np.copy(dda_volt)
    
        else:
            self.get_sub_info()
            atri_volt, atri_curr = self.get_atri_voltage_current()
            dda_volt = self.get_voltage(self.dda_volt_curr)
            dda_curr = self.get_current(self.dda_volt_curr)
            dda_temp = self.get_temperature(self.dda_temperature)
            tda_volt = self.get_voltage(self.tda_volt_curr)
            tda_curr = self.get_current(self.tda_volt_curr, diff = 0.2)
            tda_temp = self.get_temperature(self.tda_temperature)

        return atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp

    def get_sensor_hist(self, bin_width = 100):

        atri_volt_hist = np.full((bin_width), 0, dtype = int)
        atri_curr_hist = np.copy(atri_volt_hist)
        dda_volt_hist = np.full((bin_width, num_ddas), 0, dtype = int)
        dda_curr_hist = np.copy(dda_volt_hist)
        dda_temp_hist = np.copy(dda_volt_hist)
        tda_volt_hist = np.copy(dda_volt_hist)
        tda_curr_hist = np.copy(dda_volt_hist)
        tda_temp_hist = np.copy(dda_volt_hist)

        if self.empty_file_error:
            return atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist
        else:
            atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = self.get_daq_sensor_info()

            if self.station_id == 2:
                dda_volt_bins = np.linspace(3, 3.5, bin_width+1)
                dda_curr_bins = np.linspace(0, 0.5, bin_width+1)
                tda_volt_bins = np.linspace(3, 3.5, bin_width+1)
                tda_curr_bins = np.linspace(0.16, 0.21, bin_width+1)
                temp_bins = np.linspace(-25, 25, bin_width+1)
                atri_bins = np.linspace(0, 4.5, bin_width+1)
            else:
                dda_volt_bins = np.linspace(0, 10, bin_width+1)
                dda_curr_bins = np.linspace(0, 1, bin_width+1)
                tda_volt_bins = np.linspace(0, 5, bin_width+1)
                tda_curr_bins = np.linspace(0, 0.3, bin_width+1)
                temp_bins = np.linspace(-25, 25, bin_width+1)
                atri_bins = np.linspace(0, 15, bin_width+1)

            atri_volt_hist = np.histogram(atri_volt, bins = atri_bins)[0].astype(int)
            atri_curr_hist = np.histogram(atri_curr, bins = atri_bins)[0].astype(int)
            for d in range(num_ddas):
                dda_volt_hist[:,d] = np.histogram(dda_volt[:,d], bins = dda_volt_bins)[0].astype(int)
                dda_curr_hist[:,d] = np.histogram(dda_curr[:,d], bins = dda_curr_bins)[0].astype(int)
                dda_temp_hist[:,d] = np.histogram(dda_temp[:,d], bins = temp_bins)[0].astype(int)
                tda_volt_hist[:,d] = np.histogram(tda_volt[:,d], bins = tda_volt_bins)[0].astype(int)
                tda_curr_hist[:,d] = np.histogram(tda_curr[:,d], bins = tda_curr_bins)[0].astype(int)
                tda_temp_hist[:,d] = np.histogram(tda_temp[:,d], bins = temp_bins)[0].astype(int)

            return atri_volt_hist, atri_curr_hist, dda_volt_hist, dda_curr_hist, dda_temp_hist, tda_volt_hist, tda_curr_hist, tda_temp_hist

class ara_eventHk_uproot_loader:

    def __init__(self, data):

        self.empty_file_error = False
        if data is None:
            self.unix_time = np.full((1), np.nan, dtype = float)
            self.pps_counter = np.copy(self.unix_time)
            self.empty_file_error = True
            print('There is no eventHk file!')
            return

        try:
            file = uproot.open(data)
        except ValueError:
            self.unix_time = np.full((1), np.nan, dtype = float)
            self.pps_counter = np.copy(self.unix_time)
            self.empty_file_error = True
            print('eventHk is empty!')
            return

        self.hasKeyInFileError = False
        try:
            self.evtTree = file['eventHkTree']
            st_arr = np.asarray(self.evtTree['eventHk/RawAraGenericHeader/stationId'],dtype=int)
            self.station_id = st_arr[0]
            self.num_events = len(st_arr)
            self.events_entry_num = np.arange(len(st_arr))
            run_str = re.findall(r'\d+', data[-11:-5])[0]
            self.run = int(run_str)
            del st_arr, run_str
        except uproot.exceptions.KeyInFileError:
            self.unix_time = np.full((1), np.nan, dtype = float)
            self.pps_counter = np.copy(self.unix_time)
            self.empty_file_error = True
            self.hasKeyInFileError = True
            print('File is currupted!')

    def get_sub_info(self):

        self.unix_time = np.asarray(self.evtTree['eventHk/unixTime'],dtype=int)
        self.unix_time_us = np.asarray(self.evtTree['eventHk/unixTimeUs'],dtype=int)
        self.pps_counter = np.asarray(self.evtTree['eventHk/ppsCounter'],dtype=int)
        self.l1_scaler = np.asarray(self.evtTree['eventHk/l1Scaler[32]'],dtype=int)
        self.l1_threshold = np.asarray(self.evtTree['eventHk/thresholdDac[32]'],dtype=int)

        yyyymmdd_str = datetime.fromtimestamp(self.unix_time[0])
        yyyymmdd = yyyymmdd_str.strftime('%Y%m%d%H%M%S')
        self.year = int(yyyymmdd[:4])
        del yyyymmdd_str, yyyymmdd

    def get_l1_info(self):

        if self.empty_file_error:
            l1_thres = np.full((1, num_eles), np.nan, dtype = float)
            l1_rate = np.copy(l1_thres)
        else:
            pre_scale = 32
            self.get_sub_info()
            l1_rate = self.l1_scaler * pre_scale
            l1_thres = self.l1_threshold

        return l1_rate, l1_thres

class sin_subtract_loader:

    def __init__(self, max_fail_atts = 3, min_power_reduc = 0.05, min_freq = 0.2, max_freq = 0.3, dt = 0.5):

        self.dt = dt

        self.sin_sub = ROOT.FFTtools.SineSubtract(max_fail_atts, min_power_reduc, False) # no store
        self.sin_sub.setVerbose(False)
        self.sin_sub.setFreqLimits(min_freq, max_freq)
        #self.sin_sub.unsetFreqLimits()
        
    def get_sin_subtract_wf(self, int_v, int_num):

        int_v = int_v.astype(np.double)

        cw_v = np.full((int_num), 0, dtype = np.double)
        self.sin_sub.subtractCW(int_num, int_v, self.dt, cw_v)
        cw_v = cw_v.astype(float)

        self.num_sols = self.sin_sub.getNSines()
        self.num_freqs = np.frombuffer(self.sin_sub.getFreqs(), dtype = float, count = self.num_sols)
        self.num_amps = np.frombuffer(self.sin_sub.getAmps(0), dtype = float, count = self.num_sols)
    
        return cw_v

class analog_buffer_info_loader:

    def __init__(self, st, run, yrs, incl_cable_delay = False):

        if st == 3 and run > 12865:
            cap_name = os.getcwd() + f'/../data/analog_buffer/araAtriStation{st}SampleTimingNew_2019DataSet.h5'
        else:
            cap_name = os.getcwd() + f'/../data/analog_buffer/araAtriStation{st}SampleTimingNew.h5'
        cap_file = h5py.File(cap_name, 'r')
        self.num_idxs = cap_file['cap_arr'][:]
        self.idx_num_arr = cap_file['idx_arr_rm_overlap'][:]
        self.time_num_arr = cap_file['time_arr_rm_overlap'][:]
        del cap_file, cap_name

        ara_geom = ara_geom_loader(st, yrs)
        ele_ch = ara_geom.get_ele_ch_idx()
        cable_delay = ara_geom.get_cable_delay()
        del ara_geom

        self.num_idxs = self.num_idxs[:,ele_ch]
        self.idx_num_arr = self.idx_num_arr[:,:,ele_ch]
        self.time_num_arr = self.time_num_arr[:,:,ele_ch]
        if incl_cable_delay:
            self.time_num_arr -= cable_delay[np.newaxis, np.newaxis, :]
        del ele_ch, cable_delay

        self.blk_time = 20.

        print('number of samples in even/odd block are loaded!')

    def get_int_time_info(self, dt = 0.5):

        from tools.ara_wf_analyzer import wf_analyzer
        wf_int = wf_analyzer(dt = dt)

        self.num_int_idxs = np.full((2, num_useful_chs), 0, dtype = int)
        self.num_int_idxs_f = np.copy(self.num_int_idxs)

        self.int_time_num_arr = np.full((num_samples, 2, num_useful_chs), np.nan, dtype = float)
        self.int_time_f_num_arr = np.copy(self.int_time_num_arr)

        offset_arr = np.array([self.blk_time * 2, 0.])
        time_wo_nan = ~np.isnan(self.time_num_arr)

        for ant in range(num_useful_chs):
            for cap in range(2):

                # cap int
                int_ti = self.time_num_arr[:,cap,ant][time_wo_nan[:,cap,ant]][0]
                int_tf = self.time_num_arr[:,cap,ant][time_wo_nan[:,cap,ant]][-1]

                # another cap int
                an_cap = 1 - cap
                an_int_ti = self.time_num_arr[:,an_cap,ant][time_wo_nan[:,an_cap,ant]][0] + offset_arr[an_cap]
    
                # 1st & middle blk
                int_t_mid = wf_int.get_int_time(int_ti, an_int_ti)
                int_t_mid_len = len(int_t_mid)
                self.num_int_idxs[cap, ant] = int_t_mid_len
                self.int_time_num_arr[:int_t_mid_len, cap, ant] = int_t_mid

                # last blk
                int_t_last = wf_int.get_int_time(int_ti, int_tf)
                int_t_last_len = len(int_t_last)
                self.num_int_idxs_f[cap, ant] = int_t_last_len
                self.int_time_f_num_arr[:int_t_last_len, cap, ant] = int_t_last

                del an_cap, int_ti, int_tf, an_int_ti, int_t_mid, int_t_mid_len, int_t_last, int_t_last_len
        del offset_arr, time_wo_nan, wf_int

    def get_num_samp_in_blk(self, blk_idx_arr, use_int_dat = False):

        cap_idx_arr = blk_idx_arr%2
        if use_int_dat:
            self.int_samp_in_blk = np.full((len(cap_idx_arr), num_useful_chs), 0, dtype = int)
            self.int_samp_in_blk[:-1] = self.num_int_idxs[cap_idx_arr[:-1]]
            self.int_samp_in_blk[-1] = self.num_int_idxs_f[cap_idx_arr[-1]]
        else:
            self.samp_in_blk = self.num_idxs[cap_idx_arr%2]
        del cap_idx_arr

    def get_num_samps(self, use_int_dat = False):

        if use_int_dat:
            num_samp_in_blks = self.int_samp_in_blk
        else:
            num_samp_in_blks = self.samp_in_blk
        num_samps = np.nansum(num_samp_in_blks, axis = 0)
        del num_samp_in_blks

        return num_samps

    def get_mean_blk(self, ant, raw_v, use_int_dat = False):

        if use_int_dat:
            samp_in_blk_ant = self.int_samp_in_blk[:, ant]
        else:
            samp_in_blk_ant = self.samp_in_blk[:, ant]

        cs_samp_in_blk = np.nancumsum(samp_in_blk_ant)
        cs_samp_in_blk = np.concatenate((0, cs_samp_in_blk), axis=None)

        cs_raw_v = np.nancumsum(raw_v)
        cs_raw_v = np.concatenate((0.,cs_raw_v), axis=None)

        mean_blks = cs_raw_v[cs_samp_in_blk[1:]] - cs_raw_v[cs_samp_in_blk[:-1]]
        mean_blks /= samp_in_blk_ant
        del samp_in_blk_ant, cs_samp_in_blk, cs_raw_v

        return mean_blks

    def get_samp_idx(self, blk_idx_arr, ch_shape = False):

        samp_idx = self.idx_num_arr[:, blk_idx_arr%2]
        samp_idx += (blk_idx_arr * num_samples)[np.newaxis, :, np.newaxis]
        if ch_shape:
            samp_idx = np.reshape(samp_idx, (num_samples * len(blk_idx_arr), num_useful_chs), order='F')

        return samp_idx

    def get_time_arr(self, blk_idx_arr, trim_1st_blk = False, ch_shape = False, use_int_dat = False, return_min_max = False):

        remove_1_blk = int(trim_1st_blk)
        cap_idx_arr = blk_idx_arr%2

        time_offset = self.blk_time * (np.arange(len(cap_idx_arr)) + remove_1_blk) - self.blk_time * (cap_idx_arr)
        if use_int_dat:        
            time_arr = np.full((num_samples, len(cap_idx_arr), num_useful_chs), np.nan, dtype = float)
            time_arr[:, :-1] = self.int_time_num_arr[:, cap_idx_arr[:-1]]
            time_arr[:, -1] = self.int_time_f_num_arr[:, cap_idx_arr[-1]]
        else:
            time_arr = self.time_num_arr[:, cap_idx_arr]
        time_arr += time_offset[np.newaxis, : , np.newaxis]
        if ch_shape:
            time_arr = np.reshape(time_arr, (num_samples * len(cap_idx_arr), num_useful_chs), order='F')
        del remove_1_blk, cap_idx_arr, time_offset 

        if return_min_max:
            time_arr = np.asarray([np.nanmin(time_arr), np.nanmax(time_arr)], dtype = float)

        return time_arr

class repeder_loader:
       
    def __init__(self, ara_uproot, trim_1st_blk = False):

        self.sample_range = np.arange(num_samples, dtype = int)
        self.ara_uproot = ara_uproot
        self.trim_1st_blk = trim_1st_blk
       
        dda_range = np.arange(num_ddas, dtype = int)
        ch_range = np.arange(num_chs, dtype = int)
        blk_range = np.arange(num_blocks, dtype = int)
        ped_len = num_ddas * num_chs * num_blocks
        self.ped_range = np.arange(ped_len, dtype = int)

        self.ped_arr = np.full((ped_len, num_samples + 3), 0, dtype = int)
        self.ped_arr[:, 0] = np.tile(dda_range, (num_chs, num_blocks)).flatten('F')
        self.ped_arr[:, 1] = np.repeat(blk_range[:, np.newaxis], num_eles, axis = 1).flatten()
        self.ped_arr[:, 2] = np.tile(ch_range, ped_len // num_chs)
        del dda_range, ch_range, blk_range, ped_len 

    def get_samp_idx(self, evt):

        blk_idx_arr, blk_len = self.ara_uproot.get_block_idx(evt, trim_1st_blk = self.trim_1st_blk)
        samp_idx = np.repeat(self.sample_range[:, np.newaxis], blk_len, axis = 1)
        samp_idx += (blk_idx_arr * num_samples)[np.newaxis, :]
        samp_idx = np.reshape(samp_idx, (num_samples * blk_len), order = 'F')
        del blk_idx_arr, blk_len

        return samp_idx

    def get_pedestal_foramt(self, samp_medi_int, ele_ch):

        ped_ch_range = self.ped_range[ele_ch::num_eles]
        for row in ped_ch_range:
            samp_idxs = self.ped_arr[int(row), 1] * num_samples
            self.ped_arr[int(row), 3:] = samp_medi_int[samp_idxs:samp_idxs + num_samples]
            del samp_idxs
        del ped_ch_range

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
































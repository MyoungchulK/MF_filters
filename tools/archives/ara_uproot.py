import numpy as np
import uproot
import h5py

class ara_uproot_loader:

    def __init__(self, data):

        file = uproot.open(data)

        self.hasKeyInFileError = False
        try:
            self.evtTree = file['eventTree']
            st_arr = np.asarray(self.evtTree['event/RawAraStationEvent/RawAraGenericHeader/stationId'],dtype=int)
            self.station_id = st_arr[0]
            self.num_evts = len(st_arr)
            del st_arr
        except uproot.exceptions.KeyInFileError:
            self.hasKeyInFileError = True
            print('File is currupted!')

    def get_sub_info(self):
                
        self.evt_num = np.asarray(self.evtTree['event/eventNumber'],dtype=int)
        self.unix_time = np.asarray(self.evtTree['event/unixTime'],dtype=int)
        self.unix_time_us = np.asarray(self.evtTree['event/unixTimeUs'],dtype=int)
        self.read_win = np.asarray(self.evtTree['event/numReadoutBlocks'],dtype=int)
        self.time_stamp = np.asarray(self.evtTree['event/timeStamp'],dtype=int)
        self.trigger_info = np.asarray(self.evtTree['event/triggerInfo[4]'],dtype=int)

    def get_trig_type(self):

        pulserTime = np.array([254,245,245,400,400])

        trig_type = np.full(len(self.time_stamp), 0, dtype = int)
        trig_type[np.abs(self.time_stamp - pulserTime[self.station_id - 1]) < 1e4] = 1
        trig_type[self.trigger_info[:,2] == 1] = 2
        del pulserTime

        return trig_type
                
    def get_trig_ant(self, trig_ch_idx):

        trig_ch_idx_bit = 1 << trig_ch_idx
        trig_ant = np.repeat(self.trigger_info[:,0][np.newaxis, :], len(trig_ch_idx), axis=0) & trig_ch_idx_bit[:,np.newaxis]
        trig_ant[trig_ant != 0] = 1
        trig_ant = trig_ant.astype(int)
        del trig_ch_idx_bit

        return trig_ant

    def get_irs_block_number(self):

        irs_block_number = np.asarray(self.evtTree['event/blockVec/blockVec.irsBlockNumber'])

        return irs_block_number


































##
# @file tools.py
#
# @section Created on 06/29/2022, mkim@icecube.wisc.edu
#
# @brief loads data by AraRoot, make FFTs, and perform Rayleigh fitting for given run

import numpy as np
import os
import h5py
import ROOT
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import butter, filtfilt
from scipy.stats import rayleigh

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

class ara_root_loader:
    """! Class for loading data by using AraRoot"""

    def __init__(self, data, ped, st, yrs):
        """! Class initializer
        Imported from Bootcamp repo: https://github.com/ara-software/tutorials/blob/master/bootcamps/bootcamp_2020/solutions/exercise1_solution.py

        @param data  String. data path
        @param ped  String. pedestal path
        @param st  Station ID
        @param yrs  Year
        """

        ## geom info
        self.geomTool = ROOT.AraGeomTool.Instance()
        self.st_info = self.geomTool.getStationInfo(st, yrs)

        ## open a data file
        self.file = ROOT.TFile.Open(data)

        ## load in the event free for this file
        self.evtTree = self.file.Get("eventTree")
            
        ## set the tree address to access our raw data type
        self.rawEvt = ROOT.RawAtriStationEvent()
        self.evtTree.SetBranchAddress("event", ROOT.AddressOf(self.rawEvt))

        ## get the number of entries in this file
        self.num_evts = int(self.evtTree.GetEntries())
        self.entry_num = np.arange(self.num_evts)
        print('total events:', self.num_evts)

        ## open a pedestal file
        self.cal = ROOT.AraEventCalibrator.Instance()
        self.cal.setAtriPedFile(ped, st)

        ## calibration mode
        self.cal_type = ROOT.AraCalType

    def get_entry(self, evt):
        """! Get entry of the event

        @param evt  Entry/Event(blined data) number
        """   

        # get the event
        self.evtTree.GetEntry(evt)

    def get_useful_evt(self, cal_mode):
        """! Apply general ARA calibration

        @param cal_mode  Enum. calibration type
        """

        self.usefulEvt = ROOT.UsefulAtriStationEvent(self.rawEvt, cal_mode)

    def get_rf_ch_wf(self, ant):
        """! Get time/amplitude of single channel WF 
    
        @param ant  Channel index
        """        

        self.gr = self.usefulEvt.getGraphFromRFChan(ant)
        raw_t = np.frombuffer(self.gr.GetX(),dtype=float,count=-1)
        raw_v = np.frombuffer(self.gr.GetY(),dtype=float,count=-1)

        return raw_t, raw_v

    def del_TGraph(self):
        """! Delete TGraph"""

        self.gr.Delete()
        del self.gr

    def del_usefulEvt(self):
        """! Delete ARA calibration pointer. Probably doesnt needed..."""

        #self.usefulEvt.Delete()
        del self.usefulEvt

    def get_trig_type(self):
        """! Get trigger type of event 
        0: RF trigger
        1: Calpuler trigger
        2: Software (random/forced) trigger

        @return trig_type  Integer. trigger type index
        """

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

class wf_analyzer:
    """! Class that takes care of zero padding, interpolation, and fft"""

    def __init__(self, dt = 0.5, num_chs = 16, use_time_pad = False, use_freq_pad = False, use_band_pass = False, use_rfft = False):
        """! Class initializer

        @param dt time  Binwidth
        @param num_chs  Number of RF channels
        @param use_time_pad  Boolean. turn on/off zero padding
        @param use_freq_pad  Boolean. turn on/off frequency spectrum pad
        @param use_band_pass  Boolean. turn on/off band pass filter
        @param use_rfft  Boolean. turn on/off real fft
        """

        self.dt = dt
        self.num_chs = num_chs
        if use_time_pad:
            self.get_time_pad()
        if use_freq_pad:
            self.get_freq_pad(use_rfft = use_rfft)
        if use_band_pass:
            self.get_band_pass_filter()

    def get_band_pass_filter(self, low_freq_cut = 0.13, high_freq_cut = 0.85, order = 10, pass_type = 'band'):
        """! Band pass filter config       

        @param low_freq_cut  Frequency of highpass filter
        @param high_freq_cut  Frequency of lowpass filter
        @param order  The order of the filter.
        @param pass_type  The type of filter. Default is bandpass
        """

        self.nu, self.de = butter(order, [low_freq_cut, high_freq_cut], btype = pass_type, fs = 1 / self.dt)
        self.de_pad = 3*len(self.nu)

    def get_band_passed_wf(self, volt):
        """! Apply band pass filter to WF with scipy filtfilt and numerator/denominator from scipy butter
        @param volt  Amplitude of single channel WF in time-domain

        @return bp_wf Amplitude of band-passed single channel WF in time-domain
        """

        if len(volt) < self.de_pad: # if number of bins are smaller than 3*len(self.nu), use own number of bins for filtfilt process
            bp_wf = filtfilt(self.nu, self.de, volt, padlen = len(volt) - 1)
        else:
            bp_wf = filtfilt(self.nu, self.de, volt)

        return bp_wf

    def get_time_pad(self):
        """! Zero pad for time-domain WF
        It will create very long zeros array to cover WFs that have different length and cable delay      
        It will also create nan arrays for storing times/volts from all 16 channels
        """

        ## max/min time from 2013 ~2019 a2/3. You can just pad in n=1024 
        pad_i = -186.5
        pad_f = 953

        self.pad_zero_t = np.arange(pad_i, pad_f+self.dt/2, self.dt, dtype = float)
        self.pad_len = len(self.pad_zero_t)
        self.pad_t = np.full((self.pad_len, self.num_chs), np.nan, dtype = float)
        self.pad_v = np.copy(self.pad_t)
        self.pad_num = np.full((self.num_chs), 0, dtype = int)
        print(f'time pad length: {self.pad_len * self.dt} ns')

    def get_freq_pad(self, use_rfft = False):
        """! Frequency range by zero pad length. 
        It will changes if user choose real fft

        @param use_rfft  Boolean. turn on/off real fft
        """

        if use_rfft:
            self.pad_zero_freq = np.fft.rfftfreq(self.pad_len, self.dt)
        else:
            self.pad_zero_freq = np.fft.fftfreq(self.pad_len, self.dt)
        self.pad_fft_len = len(self.pad_zero_freq)
        self.df = 1 / (self.pad_len *  self.dt) # df for pad
        self.sqrt_dt = np.sqrt(self.dt)

    def get_int_wf(self, raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = False):
        """! Interpolation module for single channel WF
        User can turn on/off zero padding and band pass filtering
        
        @param raw_t  Numpy array. time of WF
        @param raw_v  Numpy array. amplitude of WF
        @param ant  RF channel index
        @param use_zero_pad  Boolean. turn on/off zero padding
        @param use_band_pass  Boolean. turn on/off band pass filter
        """

        ## akima interpolation
        akima = Akima1DInterpolator(raw_t, raw_v)
        int_v = akima(self.pad_zero_t)
        int_idx = ~np.isnan(int_v)
        int_num = np.count_nonzero(int_idx)
        int_v = int_v[int_idx]

        if use_band_pass:
            int_v = self.get_band_passed_wf(int_v)

        if use_zero_pad:
            self.pad_v[:, ant] = 0 # remaining element would be 0
            self.pad_v[int_idx, ant] = int_v
        else:
            self.pad_t[:, ant] = np.nan # clear previous info. by np.nan
            self.pad_v[:, ant] = np.nan
            self.pad_t[:int_num, ant] = self.pad_zero_t[int_idx]
            self.pad_v[:int_num, ant] = int_v

        ## store number of bins
        self.pad_num[ant] = 0
        self.pad_num[ant] = int_num
        del akima, int_idx, int_v, int_num

    def get_fft_wf(self, use_zero_pad = False, use_rfft = False, use_abs = False, use_norm = False):
        """! FFT module for multiple WFs
        User can turn on/off zero padding, real fft, absolute and normalization

        @param use_zero_pad  Boolean. turn on/off frequency spectrum pad
        @param use_rfft  Boolean. turn on/off real fft
        @param use_abs  Boolean. turn on/off absolute
        @param use_norm  Boolean. turn on/off normalization
        """

        if use_zero_pad:
            if use_rfft:
                #self.pad_fft = 2 * np.fft.rfft(self.pad_v, axis = 0)
                self.pad_fft = np.fft.rfft(self.pad_v, axis = 0)
            else:
                self.pad_fft = np.fft.fft(self.pad_v, axis = 0)
        else:
            self.pad_freq = np.full((self.pad_fft_len, self.num_chs), np.nan, dtype = float)
            self.pad_fft = np.full(self.pad_freq.shape, np.nan, dtype = complex)
            if use_rfft:
                rfft_len = self.pad_num//2 + 1
                for ant in range(self.num_chs):
                    self.pad_freq[:rfft_len[ant], ant] = np.fft.rfftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:rfft_len[ant], ant] = np.fft.rfft(self.pad_v[:self.pad_num[ant], ant])
                del rfft_len
                #self.pad_fft *= 2
            else:
                for ant in range(self.num_chs):
                    self.pad_freq[:self.pad_num[ant], ant] = np.fft.fftfreq(self.pad_num[ant], self.dt)
                    self.pad_fft[:self.pad_num[ant], ant] = np.fft.fft(self.pad_v[:self.pad_num[ant], ant])

        if use_norm:
            ## H to H/(N*sqrt(df))
            self.pad_fft /= np.sqrt(self.pad_num)[np.newaxis, :]
            self.pad_fft *= self.sqrt_dt

        if use_abs:
            self.pad_fft = np.abs(self.pad_fft)

def get_rayl_distribution(dat, binning = 1000):
    """! Rayl. fitting in each frequency

    @param dat  Numpy array. rfft of all events. dimension of array should be (number of events, rfft length, number of channels)
    @param binning  Integer. binning for 2d rfft distribution. default is 1000

    @return rayl_params  Numpy array. fit parameters from rayl. dimension of array should be (loc ans scale, rfft length, number of channels)
    @return rfft_2d  Numpy array. rfft distribution in each frequency
    @return dat_bin_edges  Numpy array. bin edges for each frequency
    """

    fft_len = dat.shape[1] # rfft length
    num_ants = dat.shape[2]
    rfft_2d = np.full((fft_len, binning, num_ants), 0, dtype = int) # rfft distribution in each frequency
    rayl_params = np.full((2, fft_len, num_ants), np.nan, dtype = float) # bin edges for each frequency

    ## if input arrray is empty, return nan arrays
    if dat.shape[0] == 0:
        rfft_2d = np.full((fft_len, binning, num_ants), np.nan, dtype =float)
        dat_bin_edges = np.full((2, fft_len, num_ants), np.nan, dtype = float)
        return rayl_params, rfft_2d, dat_bin_edges

    dat_bin_edges = np.array([np.nanmin(dat, axis = 0), np.nanmax(dat, axis = 0)], dtype = float) # collect max/min mV in each frequency
    dat_bins = np.linspace(dat_bin_edges[0], dat_bin_edges[1], binning + 1, axis = 0) # use max/min mV for creating bin space in each frequency
    dat_half_bin_width = np.abs(dat_bins[1] - dat_bins[0]) / 2 # half bins width

    ## loop over frequency and channels
    for freq in tqdm(range(fft_len)):
        for ant in range(num_ants):

            ## hitogram and save into array
            fft_hist = np.histogram(dat[:, freq, ant], bins = dat_bins[:, freq, ant])[0].astype(int)
            rfft_2d[freq, :, ant] = fft_hist

            ## calculate initial guess of scale
            mu_init_idx = np.nanargmax(fft_hist)
            if np.isnan(mu_init_idx):
                continue
            mu_init = dat_bins[mu_init_idx, freq, ant] + dat_half_bin_width[freq, ant] # it should be bin center
            del fft_hist, mu_init_idx

            try:
                ## scipy unbin rayl. fitting. It gives you scale and location. Since location is almost negligible, user can use scale as a sigma or loc+scale as a sigma
                rayl_params[:, freq, ant] = rayleigh.fit(dat[:, freq, ant], loc = dat_bin_edges[0, freq, ant], scale = mu_init)
            except RuntimeError:
                print(f'Runtime Issue in Freq. {freq} index!')
                pass
            del mu_init
    del dat_bins, dat_half_bin_width, fft_len

    return rayl_params, rfft_2d, dat_bin_edges

def get_path_info(dat_path, front_key, end_key):
    """! scrap info from string

    @param dat_path  String
    @param front_key  String
    @param end_key  String
    @return val  Value between keys
    """

    mask_idx = dat_path.find(front_key)
    if mask_idx == -1:
        print('Cannot scrap the info from path!')
        sys.exit(1)
    mask_len = len(front_key)
    end_idx = dat_path.find(end_key, mask_idx + mask_len)
    val = dat_path[mask_idx + mask_len:end_idx]
    del mask_idx, mask_len, end_idx

    return val

def get_config_info(dat_path):
    """! Get basic data path and config 
    
    @param dat_path  String
    @return Ped  String pedestal path
    @return st  Station ID
    @return run  Run number
    @return year  Year
    @return blind_type  Boolean
    """

    st = int(get_path_info(dat_path, '/ARA0', '/'))
    run = int(get_path_info(dat_path, '/run', '/'))
    year = int(get_path_info(dat_path, 'ARA/', '/'))
    Ped = f'/data/user/mkim/OMF_filter/ARA0{st}/ped_full/ped_full_values_A{st}_R{run}.dat'
    print(f'Station ID: {st}')
    print(f'Run: {run}')
    print(f'Year: {year}')

    return Ped, st, run, year

def get_qual_cut(st, run):
    """! Get quality cut result

    @param st  Station ID
    @param run  Run number
    @return bad_evt  Numpy array. 0: pass, 1: cut
    """

    path = f'/data/user/mkim/OMF_filter/ARA0{st}/qual_cut/qual_cut_A{st}_R{run}.h5'
    qual_file = h5py.File(path, 'r') 
    total_cut = qual_file['total_qual_cut_sum'][:]
    rp_ant = np.nansum(qual_file['rp_ants'][:], axis = 0)

    bad_evt = np.logical_or(total_cut != 0, rp_ant != 0).astype(int)
    print(f'Quality Cut: {path}')

    return bad_evt










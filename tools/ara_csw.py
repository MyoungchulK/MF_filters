import os
import numpy as np
import h5py
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.signal import hilbert, argrelextrema
from scipy.stats import linregress

# custom lib
from tools.ara_constant import ara_const
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_pols = ara_const.POLARIZATION

class ara_csw:

    def __init__(self, st, run, dt, pad_zero_t, analyze_blind_dat = False, get_sub_file = False, use_debug = False, verbose = False, use_all_chs = False, sim_reco_path = None, sim_psd_path = None):

        self.st = st
        self.run = run
        self.dt = dt
        self.pad_zero_t = pad_zero_t
        self.pad_len = len(self.pad_zero_t)            
        self.use_debug = use_debug
        self.verbose = verbose
        self.sim_psd_path = sim_psd_path
        self.sim_reco_path = sim_reco_path
        self.analyze_blind_dat = analyze_blind_dat
        if get_sub_file:
            if use_all_chs:
                self.good_chs = np.arange(num_ants, dtype = int)
            else:
                known_issue = known_issue_loader(self.st)
                self.good_chs = known_issue.get_bad_antenna(self.run, good_ant_true = True, print_ant_idx = True)
                del known_issue
            self.good_ch_len = len(self.good_chs)
            self.num_half_ants = int(num_ants // 2)
            self.good_pols = self.good_chs // self.num_half_ants
            self.get_detector_response()
            self.get_arrival_time_delay()
            self.get_zero_pad()

    def get_arrival_time_delay(self):

        table_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{self.st}/arr_time_table/arr_time_table_A{self.st}_Y2015.h5'
        if self.verbose:
            print('arrival time table:', table_path)
        table_hf = h5py.File(table_path, 'r')
        self.num_sols = len(table_hf['num_ray_sol'][:])
        self.arr_delay = table_hf['arr_time_table'][:] # theta, phi, rad, ant, sol
        if self.use_debug:
            self.arr_table = np.copy(self.arr_delay)
        self.arr_delay -= np.nanmean(self.arr_delay, axis = 3)[:, :, :, np.newaxis, :]
        self.arr_delay = np.transpose(self.arr_delay, (0, 1, 2, 4, 3)) # theta, phi, rad, sol, ant
        del table_path, table_hf

        if self.sim_reco_path is None:
            run_info = run_info_loader(self.st, self.run, analyze_blind_dat = self.analyze_blind_dat)
            reco_dat = run_info.get_result_path(file_type = 'reco', verbose = self.verbose)
        else:
            reco_dat = self.sim_reco_path
            if self.verbose:
                print('reco path:', self.sim_reco_path)
        reco_hf = h5py.File(reco_dat, 'r')
        coef = reco_hf['coef'][:] # pol, rad, sol, evt
        coord = reco_hf['coord'][:] # pol, tp, rad, sol, evt
        self.pol_range = np.arange(len(coef[:, 0, 0, 0]), dtype = int)
        self.sol_range = np.arange(len(coef[0, 0, :, 0]), dtype = int)
        evt_range = np.arange(len(coef[0, 0, 0, :]), dtype = int)
        tp_range = np.arange(len(coord[0, :, 0, 0, 0]), dtype = int)       

        self.coef_r_max_idx = np.argmax(coef, axis = 1) # pol, sol, evt
        self.coord_r_max_idx = coord[self.pol_range[:, np.newaxis, np.newaxis, np.newaxis], tp_range[np.newaxis, :, np.newaxis, np.newaxis], self.coef_r_max_idx, self.sol_range[np.newaxis, np.newaxis, :, np.newaxis], evt_range[np.newaxis, np.newaxis, np.newaxis, :]]
        self.coord_r_max_idx -= 0.5 
        self.coord_r_max_idx[:, 0] -= 90
        self.coord_r_max_idx[:, 0] *= -1
        self.coord_r_max_idx[:, 1] += 180
        self.coord_r_max_idx = self.coord_r_max_idx.astype(int)

        if self.use_debug:
            self.corf_r_max = coef[self.pol_range[:, np.newaxis, np.newaxis], self.coef_r_max_idx, self.sol_range[np.newaxis, :, np.newaxis], evt_range[np.newaxis, np.newaxis, :]]
        del reco_dat, reco_hf, coef, coord, evt_range, tp_range

        if self.verbose:
            print('arrival time delay is on!')

    def get_detector_response(self):
        
        from tools.ara_matched_filter import get_psd
        if self.sim_psd_path is None:
            sc_freq_amp, sc_amp = get_psd(st = self.st, run = self.run, verbose = self.verbose, analyze_blind_dat = True)[1:]
        else:
            sc_freq_amp, sc_amp = get_psd(dat_type = 'baseline', sim_path = self.sim_psd_path)[1:]
        self.sc_rms = 2 * np.nansum(sc_amp**2, axis = 0) / self.pad_len**2

        phase_path = '/home/mkim/analysis/MF_filters/data/sc_info/SC_Phase_from_sim.txt'
        if self.verbose:
            print('phase path:', phase_path)
        sc_p = np.loadtxt(phase_path)
        sc_freq_phase = sc_p[:, 0] / 1e3 # MHz to GHz
        sc_phase = sc_p[:, 1]
        self.sc_phases = interp1d(sc_freq_phase, sc_phase, fill_value = 'extrapolate')
        del phase_path, sc_p
        
        if self.use_debug:
            self.sc_freq_amp = np.copy(sc_freq_amp)
            self.sc_amp = np.copy(sc_amp)
            self.sc_freq_phase = np.copy(sc_freq_phase)
            self.sc_phase = np.copy(sc_phase)
        del sc_freq_amp, sc_amp, sc_freq_phase, sc_phase

        if self.verbose:    
            print('detector response is on!')

    def get_zero_pad(self):

        self.double_pad_len = self.pad_len * 2
        self.bool_pad = np.full((self.double_pad_len, num_pols, self.num_sols), False, dtype = bool)
        self.norm_pad = np.full((self.double_pad_len, num_pols, self.num_sols), 0, dtype = float)
        self.zero_pad = np.full((self.double_pad_len, num_pols, self.num_sols), 0, dtype = float)
        self.time_pad = np.arange(self.double_pad_len, dtype = float) * self.dt + self.pad_zero_t[0] - float(self.pad_len // 2) * self.dt
        self.range_pad = np.arange(self.double_pad_len, dtype = int)
        self.param_shape = (num_pols, self.num_sols)

        if self.verbose:
            print(f'CSW WF time! {self.time_pad[0]} ~ {self.time_pad[-1]} ns' )
            print('pad is on!')

    def get_de_dispersed_wf(self, ant):

        pad_num_ant = self.pad_num[ant]

        wf_freq = np.fft.rfftfreq(pad_num_ant, self.dt)
        
        int_sc_phase = self.sc_phases(wf_freq)
        if self.use_debug:
            self.int_sc_freq = np.copy(wf_freq)
            self.int_sc_phase = np.copy(int_sc_phase)
        int_sc_com = np.exp((0 + 1j) * int_sc_phase) 
        del wf_freq, int_sc_phase

        wf_fft = np.fft.rfft(self.pad_v[:pad_num_ant, ant])
        wf_fft /= int_sc_com 
        del int_sc_com   
 
        dd_wf_v = np.fft.irfft(wf_fft, n = pad_num_ant)
        if self.use_debug:
            self.dd_fft_v = np.copy(wf_fft)
            self.dd_wf_v = np.copy(dd_wf_v)
        del wf_fft, pad_num_ant

        return dd_wf_v

    def get_csw_wf(self):

        self.nan_flag = np.full(self.param_shape, 0, dtype = int)
        if self.use_debug:
            self.int_sc_phases = np.full((self.pad_len, num_ants), np.nan, dtype = float)
            self.int_sc_freqs = np.full((self.pad_len, num_ants), np.nan, dtype = float)
            self.dd_fft_vs = np.copy(self.int_sc_phases)
            self.dd_wf_ts = np.copy(self.int_sc_phases)
            self.dd_wf_vs = np.copy(self.int_sc_phases)
            self.shift_time = np.full((self.double_pad_len, num_ants, self.num_sols), np.nan, dtype = float)
            self.shift_dd_wf = np.copy(self.shift_time)
        for ant in range(self.good_ch_len):
            pols = self.good_pols[ant]
            dd_wf_v = self.get_de_dispersed_wf(ant)
            if self.use_debug:
                fft_len = len(self.int_sc_phase)
                wf_len = len(self.dd_wf_v)
                self.int_sc_freqs[:fft_len, self.good_chs[ant]] = self.int_sc_freq
                self.int_sc_phases[:fft_len, self.good_chs[ant]] = self.int_sc_phase
                self.dd_fft_vs[:fft_len, self.good_chs[ant]] = np.abs(self.dd_fft_v)
                self.dd_wf_ts[:, self.good_chs[ant]] = self.pad_t[:, ant]
                self.dd_wf_vs[:wf_len, self.good_chs[ant]] = self.dd_wf_v
                del fft_len, wf_len
 
            for sol in range(self.num_sols):
                arr_del = self.arr_delay[self.coord_r_max_idx[pols, 0, sol, self.evt], self.coord_r_max_idx[pols, 1, sol, self.evt], self.coef_r_max_idx[pols, sol, self.evt], sol, self.good_chs[ant]]
                if np.isnan(arr_del):
                    self.nan_flag[pols, sol] = 1
                    continue
                shift_t = self.pad_t[:self.pad_num[ant], ant] - arr_del
                dd_f = Akima1DInterpolator(shift_t, dd_wf_v)   
                int_v = dd_f(self.time_pad)
                int_idx = ~np.isnan(int_v)
                self.bool_pad[int_idx, pols, sol] = True
                self.norm_pad[int_idx, pols, sol] += self.sc_rms[self.good_chs[ant]]
                self.zero_pad[int_idx, pols, sol] += int_v[int_idx]
                if self.use_debug:
                    int_num = np.nansum(int_idx)
                    self.shift_time[:int_num, self.good_chs[ant], sol] = self.time_pad[int_idx]
                    self.shift_dd_wf[:int_num, self.good_chs[ant], sol] = int_v[int_idx]
                del shift_t, dd_f, int_v, int_idx, arr_del
            del pols, dd_wf_v

        self.csw_wf = self.zero_pad / np.sqrt(self.norm_pad)
        self.csw_wf[np.isnan(self.csw_wf) | np.isinf(self.csw_wf)] = 0

    def get_impulsivity(self):

        csw_hill = np.abs(hilbert(self.csw_wf, axis = 0)) # (# of bins, # of pols, # of rays)

        ## hillbert max
        self.hill_max_idx = np.nanargmax(csw_hill, axis = 0) # known as max spot 
        self.hill_max = csw_hill[self.hill_max_idx, self.pol_range[:, np.newaxis], self.sol_range[np.newaxis, :]] # (# of pols, # of rays)

        ## csw snr
        self.snr_csw = self.get_p2p_multiple_array() # rms is already devided it. So, p2p is already snr
        self.snr_csw /= 2 # (# of pols, # of rays) 

        ## sorting, cdf
        self.cdf_avg = np.full(self.param_shape, np.nan, dtype = float)
        self.slope = np.copy(self.cdf_avg)
        self.intercept = np.copy(self.cdf_avg)
        self.r_value = np.copy(self.cdf_avg)
        self.p_value = np.copy(self.cdf_avg)
        self.std_err = np.copy(self.cdf_avg)
        self.ks = np.copy(self.cdf_avg)
        for p in range(num_pols):
            for s in range(self.num_sols):
                if self.nan_flag[p, s]:
                    continue
                csw_trim = self.csw_wf[self.bool_pad[:, p, s], p, s]        
                range_trim = self.range_pad[self.bool_pad[:, p, s]]
                closeness = range_trim - self.hill_max_idx[p, s]
                clo_sort_idx = np.argsort(closeness)
                csw_sort = csw_trim[clo_sort_idx]
                del csw_trim, closeness, clo_sort_idx

                ## cdf
                cdf = np.nancumsum(csw_sort)
                cdf /= cdf[-1]       
                self.cdf_avg[p, s] = np.nanmean(cdf) * 2 - 1  # why?????
                del csw_sort

                range_norm = range_trim.astype(float)
                range_norm -= range_norm[0]
                range_norm /= range_norm[-1]
                self.slope[p, s], self.intercept[p, s], self.r_value[p, s], self.p_value[p, s], self.std_err[p, s] = linregress(range_norm, cdf)
                del range_trim

                self.ks[p, s] = np.nanmax(np.abs(range_norm * self.slope[p, s] + self.intercept[p, s] - cdf)) # max diff between fit and data
                del range_norm, cdf

        cdf_avg_n = self.cdf_avg < 0
        self.cdf_avg[cdf_avg_n] = 0
        del csw_hill, cdf_avg_n

    def get_p2p_multiple_array(self):

        p2p = np.full(self.param_shape, np.nan, dtype = float)    

        for p in range(num_pols):
            for s in range(self.num_sols):
                csw_each = self.csw_wf[:, p, s]
                upper_peak_idx = argrelextrema(csw_each, np.greater_equal, order=1)[0]
                lower_peak_idx = argrelextrema(csw_each, np.less_equal, order=1)[0]
                peak_idx = np.unique(np.concatenate((upper_peak_idx, lower_peak_idx)))
                peak = csw_each[peak_idx]

                p2p[p, s] = np.nanmax(np.abs(np.diff(peak)))
                del upper_peak_idx, lower_peak_idx, peak_idx, peak, csw_each

        return p2p

    def get_csw_params(self, pad_t, pad_v, pad_num, evt):

        ## clear pad
        self.bool_pad[:] = False
        self.norm_pad[:] = 0
        self.zero_pad[:] = 0

        ## self~ self~
        self.pad_t = pad_t[:, self.good_chs]
        self.pad_v = pad_v[:, self.good_chs]
        self.pad_num = pad_num[self.good_chs]
        self.evt = evt

        ## csw wf making
        self.get_csw_wf()

        ## impulsivity value
        self.get_impulsivity()
        del self.csw_wf, self.pad_t, self.pad_v, self.pad_num, self.evt









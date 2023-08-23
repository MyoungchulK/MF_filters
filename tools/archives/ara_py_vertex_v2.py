import os
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from itertools import combinations
import ROOT

# custom lib
from tools.ara_constant import ara_const
from tools.ara_known_issue import known_issue_loader
from tools.ara_data_load import ara_geom_loader

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraVertex.so")

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION
num_pols = ara_const.POLARIZATION
num_pols_com = int(num_pols + 1)

class py_reco_handler:

    def __init__(self, st, run, dt, hit_thres, sum_win = 25, num_ants_cut = 3, use_debug = False, use_input_hit = False):

        self.dt = dt
        self.sum_win = sum_win
        self.sum_win_idx = int(np.round(self.sum_win / self.dt))
        self.hit_thres = hit_thres
        self.num_ants_cut = num_ants_cut
        self.use_debug = use_debug
        self.use_input_hit = use_input_hit

        ## ch info
        self.ant_range = np.arange(num_ants, dtype = int)
        self.polarizations = (self.ant_range > int(num_ants // 2 - 1)).astype(int)
        known_issue = known_issue_loader(st)
        self.can_use_ch = known_issue.get_bad_antenna(run, good_ant_true = True) # good chs in bool
        ara_geom = ara_geom_loader(st, int(2015))
        self.ant_pos = ara_geom.get_ant_xyz() # (xyz, chs)
        del known_issue, ara_geom

    def get_max_info(self):

        ## max bin, volt, and time of al chs
        self.max_bin = np.nanargmax(self.pad_v, axis = 0)
        self.max_val = self.pad_v[self.max_bin, self.ant_range]
        self.max_time = self.pad_t[self.max_bin, self.ant_range]

    def get_sqrt_volt_sum_wf(self):

        ## rolling mean of all chs by uniform_filter1d()
        self.pad_v = np.sqrt(uniform_filter1d(self.pad_v, size = self.sum_win_idx, mode = 'constant', axis = 0))

    def get_mean_sigma_in_no_max(self):

        ## half signal window
        bin_8 = self.pad_num // 8
       
        ## remove signal windo info 
        pad_no_max = np.full(self.pad_v.shape, np.nan, dtype = float)
        for ant in range(num_ants):
            pad_no_max[:self.pad_num[ant], ant] = self.pad_v[:self.pad_num[ant], ant]
            front_idx = int(self.max_bin[ant] - bin_8[ant])
            if front_idx < 0:
                front_idx = 0
            pad_no_max[front_idx:self.max_bin[ant] + bin_8[ant] + 1, ant] = np.nan
            del front_idx
        if self.use_debug:
            self.pad_no_max = pad_no_max
            self.bin_8 = bin_8

        ## mean sigma
        self.pad_mean = np.nanmean(pad_no_max, axis = 0)
        self.pad_sigma = np.nanstd(pad_no_max, axis = 0)
        del bin_8, pad_no_max
           
    def get_ch_sliding_v2_snr_uw(self): 

        ## to power and rolling mean
        self.get_sqrt_volt_sum_wf()

        ## max val, time and bin
        self.get_max_info()

        ## mean sigma
        self.get_mean_sigma_in_no_max()

        ## snr and hit time
        self.snr_arr = (self.max_val - self.pad_mean) / self.pad_sigma        
        self.hit_time_arr = self.max_time
        nega_sigma_idx = self.pad_sigma <= 0
        if np.count_nonzero(nega_sigma_idx) > 0:
            self.snr_arr[nega_sigma_idx] = 0
            self.hit_time_arr[nega_sigma_idx] = -9999
        if self.use_debug:
            self.nega_sigma_idx = nega_sigma_idx
        del nega_sigma_idx

    def get_id_hits_prep_to_vertex(self, pad_v = None, pad_t = None, pad_num = None, snr = None, hit = None):

        if self.use_input_hit:
            self.snr_arr = snr
            self.hit_time_arr = hit
        else:
            ## input waveform
            self.pad_v = pad_v ** 2
            self.pad_v[np.isnan(self.pad_v)] = 0
            self.pad_t = pad_t
            self.pad_num = pad_num

            ## hit time and snr
            self.get_ch_sliding_v2_snr_uw()
            if self.use_debug == False:
                del self.pad_v, self.pad_t, self.pad_num, self.max_bin, self.max_val, self.max_time, self.pad_mean, self.pad_sigma

        ## ch selection. tag only uesful ch, ch that has bigger than snr, same pol, and numher of hit that bigger than cut
        ch_for_reco = np.full((num_ants, num_pols_com), False, dtype = bool)
        ch_for_reco[:, 2] = np.logical_and(self.can_use_ch, self.snr_arr > self.hit_thres)
        ch_for_reco[:, 0] = np.all((self.polarizations == 0, ch_for_reco[:, 2]), axis = 0)
        ch_for_reco[:, 1] = np.all((self.polarizations == 1, ch_for_reco[:, 2]), axis = 0)
        n_hits = np.count_nonzero(ch_for_reco, axis = 0) > self.num_ants_cut
        ch_for_reco[:, n_hits == False] = False
        if self.use_debug:
            self.ch_for_reco = ch_for_reco
            self.n_hits = n_hits
        del n_hits

        ## paring info
        self.pair_info = []
        self.useful_num_ants = np.full((num_pols_com), 0, dtype = int)
        for pol in range(num_pols_com):
            useful_ants = self.ant_range[ch_for_reco[:, pol]]
            self.useful_num_ants[pol] = len(useful_ants)
            if self.useful_num_ants[pol] == 0:
                pair_pol = np.full((7, 0), np.nan, dtype = float)
                self.pair_info.append(pair_pol)
                del useful_ants
                continue
            useful_pair = np.asarray(list(combinations(useful_ants, 2))).astype(int) 
            pair_pol = np.full((7, len(useful_pair[:, 0])), np.nan, dtype = float) # dt, x1, y1, z1, x2, y2, z2
            pair_pol[0] = self.hit_time_arr[useful_pair[:, 0]] - self.hit_time_arr[useful_pair[:, 1]] 
            pair_pol[1:4] = self.ant_pos[:, useful_pair[:, 0]] 
            pair_pol[4:] = self.ant_pos[:, useful_pair[:, 1]]
            self.pair_info.append(pair_pol)
            del useful_ants, useful_pair
        del ch_for_reco
         
class py_ara_vertex:

    def __init__(self, st):

        ara_geom = ara_geom_loader(st, int(2015))
        self.ant_pos_mean = ara_geom.get_ant_xyz(use_mean = True) # (xyz)
        del ara_geom

        self.reco = ROOT.AraVertex()
        self.reco.SetCOG(self.ant_pos_mean[0], self.ant_pos_mean[1], self.ant_pos_mean[2])

    def get_add_pair(self, pair_indi, pair_len):

        self.reco.clear()
        for pair in range(pair_len):
            self.reco.addPair(pair_indi[0, pair], pair_indi[1, pair], pair_indi[2, pair], pair_indi[3, pair], pair_indi[4, pair], pair_indi[5, pair], pair_indi[6, pair])

    def get_print_pair(self):

        self.reco.printPairs()

    def get_pair_fit_spherical(self, pair_info, useful_num_ants):

        self.theta = np.full((num_pols_com), np.nan, dtype = float) 
        self.phi = np.copy(self.theta)
        for pol in range(num_pols_com):
            if useful_num_ants[pol] == 0:
                continue
            if pol == 2:
                pass_flag = True
                if useful_num_ants[pol] == useful_num_ants[0]:
                    pol_copy = 0
                elif useful_num_ants[pol] == useful_num_ants[1]:
                    pol_copy = 1
                else:
                    pass_flag = False
                if pass_flag:    
                    self.theta[pol] = self.theta[pol_copy]
                    self.phi[pol] = self.phi[pol_copy]
                    del pol_copy
                    continue
                else:
                    pass

            pair_indi = pair_info[pol]
            pair_len = len(pair_indi[0])
            
            self.get_add_pair(pair_indi, pair_len)
            #self.get_print_pair()

            fit_result = self.reco.doPairFitSpherical()
            self.theta[pol] = fit_result.theta
            self.phi[pol] = fit_result.phi
            del pair_indi, pair_len, fit_result

        self.theta = 90 - np.degrees(self.theta)
        self.phi = np.degrees(self.phi)












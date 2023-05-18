import numpy as np
import os
import ROOT
import uproot
import ctypes
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/../AraSim/libAra.so")
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/../AraSim/libSim.so")
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

def get_sim_rf_ch_map():
    # im TIRED pulling out info from SIM!!

    ch_map = np.full((num_ants, 3), 0, dtype = int)
    ch_map[:, 0] = np.arange(num_ants, dtype = int) # rf ch order
    ch_map[:, 1] = np.array([3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype = int) # sim st index
    ch_map[:, 2] = np.array([2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1], dtype = int) # sim ant index

    return ch_map

def get_sim_final_posant_rf(st):
    # shitty sim...

    posnu_rf = np.full((6, num_ants), np.nan, dtype = float) # x,y,z,r,t,p
    if st == 2:
        posnu_rf[:, 0] = np.array([10010.18372738906, 10001.939659758575, 6359462.114436145, 6359477.858088968, 0.0022251388159165593, 0.7849862097529178], dtype = float)
        posnu_rf[:, 1] = np.array([10004.448086540997, 9989.198631557718, 6359462.043487443, 6359477.758088969, 0.0022230848931226303, 0.7846354485090016], dtype = float)
        posnu_rf[:, 2] = np.array([9997.013473727071, 10008.972119082271, 6359460.78208568, 6359476.516088969, 0.002224457034326554, 0.7859959166985997], dtype = float)
        posnu_rf[:, 3] = np.array([9991.748020720568, 9995.530765170863, 6359457.023492144, 6359472.728088968, 0.0022223779955075707, 0.785587420997027], dtype = float)
        posnu_rf[:, 4] = np.array([10010.153418912172, 10001.90937624278, 6359442.859483813, 6359458.603088968, 0.0022251388159165593, 0.7849862097529178], dtype = float)
        posnu_rf[:, 5] = np.array([10004.418113207235, 9989.168703911335, 6359442.9905345235, 6359458.705088968, 0.0022230848931226303, 0.7846354485090016], dtype = float)
        posnu_rf[:, 6] = np.array([9996.982579469453, 10008.94118786827, 6359441.129134304, 6359456.863088969, 0.002224457034326554, 0.7859959166985997], dtype = float)
        posnu_rf[:, 7] = np.array([9991.718343082748, 9995.501076297478, 6359438.13453879, 6359453.839088969, 0.0022223779955075707, 0.785587420997027], dtype = float)
        posnu_rf[:, 8] = np.array([10010.188063917434, 10001.943992715522, 6359464.869429325, 6359480.613088969, 0.0022251388159165593, 0.7849862097529178], dtype = float)
        posnu_rf[:, 9] = np.array([10004.452678582213, 9989.203216599433, 6359464.962480229, 6359480.677088968, 0.0022230848931226303, 0.7846354485090016], dtype = float)
        posnu_rf[:, 10] = np.array([9997.018379898038, 10008.977031122107, 6359463.903077958, 6359479.637088968, 0.002224457034326554, 0.7859959166985997], dtype = float)
        posnu_rf[:, 11] = np.array([9991.752666640154, 9995.53541284933, 6359459.980484842, 6359475.685088969, 0.0022223779955075707, 0.785587420997027], dtype = float)
        posnu_rf[:, 12] = np.array([10010.158071826101, 10001.914025324719, 6359445.815476495, 6359461.559088969, 0.0022251388159165593, 0.7849862097529178], dtype = float)
        posnu_rf[:, 13] = np.array([10004.4232810234, 9989.173863850365, 6359446.275526406, 6359461.990088969, 0.0022230848931226303, 0.7846354485090016], dtype = float)
        posnu_rf[:, 14] = np.array([9996.988427260545, 10008.947042654618, 6359444.8491251, 6359460.583088969, 0.002224457034326554, 0.7859959166985997], dtype = float)
        posnu_rf[:, 15] = np.array([9991.723504342754, 9995.50623951147, 6359441.419530679, 6359457.124088969, 0.0022223779955075707, 0.785587420997027], dtype = float)

    elif st == 3:
        posnu_rf[:, 0] = np.array([10004.41617157611, 9990.175300978386, 6359459.781995977, 6359475.498087071, 0.0022231906474390853, 0.7846859273805618], dtype = float)
        posnu_rf[:, 1] = np.array([10010.229739645067, 10002.995598081141, 6359456.150686229, 6359471.896087071, 0.0022252633842382136, 0.7850366953618478], dtype = float)
        posnu_rf[:, 2] = np.array([9997.580674733237, 10009.002507324123, 6359458.265138069, 6359474.00008707, 0.002224524325320767, 0.7859690670468141], dtype = float)
        posnu_rf[:, 3] = np.array([9991.447372859913, 9995.660741698302, 6359460.186766124, 6359475.891087071, 0.0022223579228420297, 0.7856089677195057], dtype = float)
        posnu_rf[:, 4] = np.array([10004.386198327313, 9990.145370395263, 6359440.729043062, 6359456.445087071, 0.0022231906474390853, 0.7846859273805618], dtype = float)
        posnu_rf[:, 5] = np.array([10010.200266829603, 10002.96614656494, 6359437.426732587, 6359453.1720870705, 0.0022252633842382136, 0.7850366953618478], dtype = float)
        posnu_rf[:, 6] = np.array([9997.551495418265, 10008.973294672962, 6359439.704183994, 6359455.4390870705, 0.002224524325320767, 0.7859690670468141], dtype = float)
        posnu_rf[:, 7] = np.array([9991.417438467512, 9995.63079468264, 6359441.1338131735, 6359456.838087071, 0.0022223579228420297, 0.7856089677195057], dtype = float)
        posnu_rf[:, 8] = np.array([10004.421081380839, 9990.180203794213, 6359462.902988264, 6359478.619087071, 0.0022231906474390853, 0.7846859273805618], dtype = float)
        posnu_rf[:, 9] = np.array([10010.23439415881, 10003.000249231185, 6359459.107678907, 6359474.85308707, 0.0022252633842382136, 0.7850366953618478], dtype = float)
        posnu_rf[:, 10] = np.array([9997.586158136644, 10009.007996992097, 6359461.75312944, 6359477.488087071, 0.002224524325320767, 0.7859690670468141], dtype = float)
        posnu_rf[:, 11] = np.array([9991.452018637392, 9995.665389434895, 6359463.143758821, 6359478.8480870705, 0.0022223579228420297, 0.7856089677195057], dtype = float)
        posnu_rf[:, 12] = np.array([10004.391314214841, 9990.15047900054, 6359443.981035026, 6359459.697087071, 0.0022231906474390853, 0.7846859273805618], dtype = float)
        posnu_rf[:, 13] = np.array([10010.20491976928, 10002.970796142055, 6359440.382725269, 6359456.128087071, 0.0022252633842382136, 0.7850366953618478], dtype = float)
        posnu_rf[:, 14] = np.array([9997.556144048964, 10008.977948614534, 6359442.661176679, 6359458.396087071, 0.002224524325320767, 0.7859690670468141], dtype = float)
        posnu_rf[:, 15] = np.array([9991.42234190732, 9995.635700190218, 6359444.254805467, 6359459.959087071, 0.0022223579228420297, 0.7856089677195057], dtype = float)
 
    return posnu_rf

class ara_root_loader:

    def __init__(self, data, st, yrs):

        #geom info
        from tools.ara_data_load import ara_geom_loader
        self.st = st
        self.ara_geom = ara_geom_loader(self.st, yrs) # maybe no need this....

        # open a data file
        self.file = ROOT.TFile.Open(data)

        # load in the event free for this file
        self.evtTree = self.file.Get("eventTree")

        # set the tree address to access our raw data type
        self.realEvt = ROOT.UsefulAtriStationEvent()
        self.evtTree.SetBranchAddress("UsefulAtriStationEvent", ROOT.AddressOf(self.realEvt))

        # get the number of entries in this file
        self.num_evts = int(self.evtTree.GetEntries())
        self.entry_num = np.arange(self.num_evts, dtype = int)
        print('total events:', self.num_evts)

    def get_sub_info(self, data, get_angle_info = False, get_temp_info = False):

        # tired of dealing with PyROOT.....
        file_uproot = uproot.open(data)

        ara_tree = file_uproot['AraTree']
        settings = ara_tree['settings']
        self.time_step = np.asarray(settings['TIMESTEP'], dtype = float) * 1e9
        self.waveform_length = np.asarray(settings['WAVEFORM_LENGTH'], dtype = int)[0]
        self.wf_time = np.arange(self.waveform_length) * self.time_step - self.waveform_length // 2 * self.time_step
        self.posnu_radius = np.asarray(settings['POSNU_RADIUS'], dtype = int)
        self.nnu_tot = np.asarray(settings['NNU'], dtype = int)
        self.exponent_range = np.full((2), 0, dtype = int)
        try:
            self.exponent_range[0] = np.asarray(settings['EXPONENT_MIN'], dtype = int)[0]
            self.exponent_range[1] = np.asarray(settings['EXPONENT_MAX'], dtype = int)[0]
        except uproot.exceptions.KeyInFileError:
            print('OLD SIM!!! SHAME ON YOU!!!')
            pass
        
        ara_tree_2 = file_uproot['AraTree2']
        event = ara_tree_2['event']
        self.pnu = np.asarray(event['pnu'], dtype = float)
        self.nuflavorint = np.asarray(event['nuflavorint'], dtype = int)
        self.nu_nubar = np.asarray(event['nu_nubar'], dtype = int)
        self.inu_thrown = np.asarray(event['inu_thrown'], dtype = int)
        self.weight = np.asarray(event['Nu_Interaction/Nu_Interaction.weight'], dtype = float)
        self.probability = np.asarray(event['Nu_Interaction/Nu_Interaction.probability'], dtype = float)
        self.currentint = np.asarray(event['Nu_Interaction/Nu_Interaction.currentint'], dtype = int)
        self.elast_y = np.asarray(event['Nu_Interaction/Nu_Interaction.elast_y'], dtype = float)
        self.posnu = np.full((6, self.num_evts), np.nan, dtype = float)
        self.posnu[0] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.x'], dtype = float)
        self.posnu[1] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.y'], dtype = float)
        self.posnu[2] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.z'], dtype = float)
        self.posnu[3] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.theta'], dtype = float) # from earth center. If user want to get theta,phi,and r from antenna center, user have to get from xyz...
        self.posnu[4] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.phi'], dtype = float) # from earth center
        self.posnu[5] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.r'], dtype = float) # radius from earth center
        self.nnu = np.full((6, self.num_evts), np.nan, dtype = float)
        self.nnu[0] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.x'], dtype = float)
        self.nnu[1] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.y'], dtype = float)
        self.nnu[2] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.z'], dtype = float)
        self.nnu[3] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.theta'], dtype = float)
        self.nnu[4] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.phi'], dtype = float)
        self.nnu[5] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.r'], dtype = float)
        del file_uproot, ara_tree, settings, ara_tree_2, event

        self.sim_rf_ch_map = get_sim_rf_ch_map()
        self.posant_rf = get_sim_final_posant_rf(self.st)
        self.posant_center = np.nanmean(self.posant_rf[:3], axis = 1)
        self.posnu_antcen_tpr = self.get_posnu_ant_cen_theta_phi_r()

        self.rec_ang = np.full((2, num_ants, self.num_evts), np.nan, dtype = float)
        self.view_ang = np.copy(self.rec_ang)
        self.launch_ang = np.copy(self.rec_ang)
        self.arrival_time = np.copy(self.rec_ang)
        self.signal_bin = np.copy(self.rec_ang)
        if get_angle_info:
            print('angle info in on! prepare for the looong for loop...')
            sim_st_index = self.sim_rf_ch_map[:, 1]
            sim_ant_index = self.sim_rf_ch_map[:, 2]
 
            ROOT.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/Report.h"')
            AraTree2 = self.file.AraTree2
            for evt in tqdm(range(self.num_evts)):
                AraTree2.GetEntry(evt)
                for ant in range(num_ants):
                    sim_st = int(sim_st_index[ant])
                    sim_ant = int(sim_ant_index[ant])
                    rec = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].rec_ang[:]))
                    view = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].view_ang[:]))
                    launch = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].launch_ang[:]))
                    arrival = np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].arrival_time[:]) * 1e9        
                    sig_bin = np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].SignalBinTime[:])

                    self.rec_ang[:len(rec), ant, evt] = rec
                    self.view_ang[:len(view), ant, evt] = view
                    self.launch_ang[:len(launch), ant, evt] = launch
                    self.arrival_time[:len(arrival), ant, evt] = arrival
                    self.signal_bin[:len(sig_bin), ant, evt] = sig_bin
                    del rec, view, arrival, launch, sim_st, sim_ant, sig_bin
            del AraTree2
        if get_temp_info:
            print('template info is on!')
            sim_st_index = self.sim_rf_ch_map[:, 1]
            sim_ant_index = self.sim_rf_ch_map[:, 2]

            ROOT.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/Report.h"')
            AraTree2 = self.file.AraTree2
            for evt in tqdm(range(self.num_evts)):
                AraTree2.GetEntry(evt)
                for ant in range(num_ants):
                    sim_st = int(sim_st_index[ant])
                    sim_ant = int(sim_ant_index[ant])
                    sig_bin = np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].SignalBinTime[:])
                    rec = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st].antennas[sim_ant].rec_ang[:]))
                    self.signal_bin[:len(sig_bin), ant, evt] = sig_bin
                    self.rec_ang[:len(rec), ant, evt] = rec
                    del sim_st, sim_ant, sig_bin, rec
            del AraTree2

    def get_posnu_ant_cen_theta_phi_r(self, use_radian = False):

        # unit vector
        theta_unit_vec = np.array([0, 0, 1])
        phi_unit_vec = np.array([1, 0, 0])
        
        # posnu vector
        posnu_vec = self.posnu[:3] - self.posant_center[:, np.newaxis] # (xyz, num_evts)

        # theta phi r
        tpr = np.full((3, self.num_evts), np.nan, dtype = float)

        # zenith calculation to center of antenna
        AB = np.nansum(posnu_vec * theta_unit_vec[:, np.newaxis]) # A.B
        ABabs = np.sqrt(np.nansum(posnu_vec ** 2, axis = 0)) * np.sqrt(np.nansum(theta_unit_vec ** 2)) # |AB|
        zen_ang = np.arccos(AB / ABabs)
        tpr[0] = np.degrees(zen_ang)
        if use_radian:
            tpr[0] = np.nan # just in case
            tpr[0] = zen_ang
        del AB, ABabs, theta_unit_vec, zen_ang

        # phi calculation to center of antenna
        AD = np.nansum(posnu_vec[:2] * phi_unit_vec[:2][:, np.newaxis]) # A.D
        ADabs = np.sqrt(np.nansum(posnu_vec[:2] ** 2, axis = 0)) * np.sqrt(np.nansum(phi_unit_vec[:2] ** 2)) # |AD|
        phi_ang = np.arccos(AD / ADabs)
        minus_index = self.posnu[1] < self.posant_center[1]
        phi_ang[minus_index] *= -1
        phi_ang[minus_index] += np.radians(360)
        tpr[1] = np.degrees(phi_ang)
        if use_radian:
            tpr[1] = np.nan # just in case
            tpr[1] = phi_ang
        del AD, ADabs, phi_unit_vec, phi_ang, minus_index

        # R calculation
        tpr[2] = np.sqrt(np.nansum(posnu_vec ** 2, axis = 0))
        del posnu_vec

        return tpr

    def get_entry(self, evt):

        # get the event
        self.evtTree.GetEntry(evt)

    def get_rf_ch_wf(self, ant):

        self.gr = self.realEvt.getGraphFromRFChan(ant)
        raw_t = np.frombuffer(self.gr.GetX(),dtype=float,count=-1)
        raw_v = np.frombuffer(self.gr.GetY(),dtype=float,count=-1)

        return raw_t, raw_v

    def del_TGraph(self):

        self.gr.Delete()
        del self.gr

    def get_rf_wfs(self, evt):

        wf_v = np.full((self.waveform_length, num_ants), 0, dtype = float)

        self.get_entry(evt)
        for ant in range(num_ants):
            wf_v[:, ant] = self.get_rf_ch_wf(ant)[1]
            self.del_TGraph()

        return wf_v

# Got the idea from Kaeli Hughes.
# kahughes@uchicago.edu
class ara_raytrace_loader:

    def __init__(self, n0 = 1.35, nf = 1.78, l = 0.0132, n_bulk = 1.5, use_bulk_ice = False, verbose = False):

        self.use_bulk_ice = use_bulk_ice
        self.verbose = verbose
        self.ara_sim = ROOT
        if self.use_bulk_ice:
            self.ice_model = np.array([n_bulk], dtype = float)
        else:
            self.ice_model = np.array([n0, nf, l], dtype = float)
        if self.verbose:
            print('ice model:', self.ice_model)

        # header
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/RayTrace.h"')
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/RayTrace_IceModels.h"')
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/Vector.h"')

        # attenuation & exponential model
        if self.use_bulk_ice:
            self.ara_sim.gInterpreter.ProcessLine(f'auto refr_model = boost::shared_ptr<constantRefractiveIndex>(new constantRefractiveIndex({n_bulk}));')
        else:
            self.ara_sim.gInterpreter.ProcessLine(f'auto refr_model = boost::shared_ptr<exponentialRefractiveIndex>(new exponentialRefractiveIndex({n0},{nf},{l}));')
        self.ara_sim.gInterpreter.ProcessLine('auto atten_model = boost::shared_ptr<basicAttenuationModel>( new basicAttenuationModel );')

        # link ray tracing model
        self.ara_sim.gInterpreter.ProcessLine('RayTrace::TraceFinder tf(refr_model, atten_model);')

        # link ray trace output format
        self.ara_sim.gInterpreter.ProcessLine('Vector src; Vector rec; std::vector<RayTrace::TraceRecord> paths;')

    def get_src_trg_position(self, st, yrs, theta_bin = np.linspace(0.5, 179.5, 179 + 1), phi_bin = np.linspace(-179.5, 179.5, 359 + 1), radius_bin = np.array([41,300]), num_ray_sol = np.array([1, 2], dtype = int), debug = False):

        from tools.ara_data_load import ara_geom_loader
        ara_geom = ara_geom_loader(st, yrs)
        self.ant_pos = ara_geom.get_ant_xyz()
        del ara_geom

        self.num_ray_sol = num_ray_sol
        self.theta_bin = theta_bin
        self.phi_bin = phi_bin
        self.radius_bin = radius_bin
        if self.verbose:
            print(f'antenna position (A{st} Y{yrs}):', self.ant_pos.T)
            print('theta range:', self.theta_bin)
            print('phi range:', self.phi_bin)
            print('radius range:', self.radius_bin)
            print('number of ray solution:', self.num_ray_sol)
        theta_radian = np.radians(self.theta_bin)
        phi_radian = np.radians(self.phi_bin)

        r_4d = np.tile(self.radius_bin[np.newaxis, np.newaxis, :, np.newaxis], (len(self.theta_bin), len(self.phi_bin), len(self.num_ray_sol), num_ants))
        x_4d = r_4d * np.sin(theta_radian)[:, np.newaxis, np.newaxis, np.newaxis] * np.cos(phi_radian)[np.newaxis, :, np.newaxis, np.newaxis]
        y_4d = r_4d * np.sin(theta_radian)[:, np.newaxis, np.newaxis, np.newaxis] * np.sin(phi_radian)[np.newaxis, :, np.newaxis, np.newaxis]
        z_4d = r_4d * np.cos(theta_radian)[:, np.newaxis, np.newaxis, np.newaxis]
        del theta_radian, phi_radian

        self.trg_r = np.sqrt((x_4d - self.ant_pos[np.newaxis, np.newaxis, np.newaxis, 0, :])**2 + (y_4d - self.ant_pos[np.newaxis, np.newaxis, np.newaxis, 1, :])**2)
        self.trg_z = np.copy(self.ant_pos[2])

        self.src_r = np.full((1), 0, dtype = float)
        self.src_z = z_4d + np.nanmean(self.ant_pos[2])
        del r_4d, x_4d, y_4d, z_4d

        if debug and self.verbose:
            print('src_r:',self.src_r)
            print('trg_z:',self.trg_z)
            print('trg_r:',self.trg_r.shape)
            print('trg_r[:,0,0,0]:',self.trg_r[:,0,0,0])
            print('src_z:',self.src_z.shape)
            print('src_z[:,0,0,0]:',self.src_z[:,0,0,0])

    def get_ray_solution(self, trg_r, src_z, trg_z = -200, debug = False):

        #setting vector
        self.ara_sim.src.SetXYZ(trg_r, 0, src_z);
        self.ara_sim.rec.SetXYZ(0, 0, trg_z);

        # number of solution and error
        sol_cnt = ctypes.c_int()
        sol_err = ctypes.c_int()

        #raytracing
        self.ara_sim.paths = self.ara_sim.tf.findPaths(self.ara_sim.rec, self.ara_sim.src, 0.3, self.ara_sim.TMath.Pi()/2, sol_cnt, sol_err, 1, 0.2)
        del sol_cnt, sol_err

        path_len = np.full((len(self.num_ray_sol)), np.nan, dtype = float)
        path_time = np.copy(path_len)
        launch_ang = np.copy(path_len)
        receipt_ang = np.copy(path_len)
        reflection_ang = np.copy(path_len)
        miss = np.copy(path_len)
        attenuation = np.copy(path_len)

        sol_num = 0
        for sol in self.ara_sim.paths:
            path_len[sol_num] = sol.pathLen
            path_time[sol_num] = sol.pathTime*1e9
            launch_ang[sol_num] = sol.launchAngle
            receipt_ang[sol_num] = sol.receiptAngle
            reflection_ang[sol_num] = sol.reflectionAngle
            miss[sol_num] = sol.miss
            attenuation[sol_num] = sol.attenuation
            if debug and self.verbose:
                print(f'Path time: {sol.pathTime*1e9} ns, Path length: {sol.pathLen} m, Attenuation: {sol.attenuation} m, Miss: {sol.miss} m')
            if sol_num == 0 and self.use_bulk_ice == True:
                break
            sol_num += 1
            
        return path_len, path_time, launch_ang, receipt_ang, reflection_ang, miss, attenuation

    def get_arrival_time_table(self, debug = False):

        path_len = np.full((len(self.theta_bin), len(self.phi_bin), len(self.radius_bin), num_ants, len(self.num_ray_sol)), np.nan, dtype = float)
        path_time = np.copy(path_len)
        launch_ang = np.copy(path_len)
        receipt_ang = np.copy(path_len)
        reflection_ang = np.copy(path_len)
        miss = np.copy(path_len)
        attenuation = np.copy(path_len)

        if self.verbose:
            print('arrival time table size:', path_time.shape)

        for t in tqdm(range(len(self.theta_bin)), ascii = False):
            for p in range(len(self.phi_bin)):
                for r in range (len(self.radius_bin)):
                    for a in range(num_ants):

                        path_len[t, p, r, a], path_time[t, p, r, a], launch_ang[t, p, r, a], receipt_ang[t, p, r, a], reflection_ang[t, p, r, a], miss[t, p, r, a], attenuation[t, p, r, a] = self.get_ray_solution(self.trg_r[t, p, r, a], self.src_z[t, p, r, a], self.trg_z[a], debug = debug)

        return path_len, path_time, launch_ang, receipt_ang, reflection_ang, miss, attenuation



















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

class ara_root_loader:

    def __init__(self, data, st, yrs):

        #geom info
        from tools.ara_data_load import ara_geom_loader
        self.st = st
        self.ara_geom = ara_geom_loader(self.st, yrs)

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

    def get_sub_info(self, data, get_angle_info = False):

        # tired of dealing with PyROOT.....
        file_uproot = uproot.open(data)

        ara_tree = file_uproot['AraTree']
        settings = ara_tree['settings']
        self.time_step = np.asarray(settings['TIMESTEP'], dtype = float) * 1e9
        self.waveform_length = np.asarray(settings['WAVEFORM_LENGTH'], dtype = int)[0]
        self.wf_time = np.arange(self.waveform_length) * self.time_step - self.waveform_length // 2 * self.time_step
        self.posnu_radius = np.asarray(settings['POSNU_RADIUS'], dtype = int)
 
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
        self.posnu[3] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.theta'], dtype = float)
        self.posnu[4] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.phi'], dtype = float)
        self.posnu[5] = np.asarray(event['Nu_Interaction/Nu_Interaction.posnu.r'], dtype = float)
        self.nnu = np.full((6, self.num_evts), np.nan, dtype = float)
        self.nnu[0] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.x'], dtype = float)
        self.nnu[1] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.y'], dtype = float)
        self.nnu[2] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.z'], dtype = float)
        self.nnu[3] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.theta'], dtype = float)
        self.nnu[4] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.phi'], dtype = float)
        self.nnu[5] = np.asarray(event['Nu_Interaction/Nu_Interaction.nnu.r'], dtype = float)
        del file_uproot, ara_tree, settings, ara_tree_2, event
       
        if get_angle_info:
            if self.st == 2:
                sim_st_index = [3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2]
                sim_ant_index = [2,2,2,2,0,0,0,0,3,3,3,3,1,1,1,1]
            self.rec_ang = np.full((2, num_ants, self.num_evts), np.nan, dtype = float)
            self.view_ang = np.copy(self.rec_ang)
            self.arrival_time = np.copy(self.rec_ang)
            
            ROOT.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../AraSim/Report.h"')
            AraTree2 = self.file.AraTree2
            for evt in tqdm(range(self.num_evts)):
                AraTree2.GetEntry(evt)
                for ant in range(num_ants):
                    rec = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st_index[ant]].antennas[sim_ant_index[ant]].rec_ang[:]))
                    view = np.degrees(np.asarray(AraTree2.report.stations[0].strings[sim_st_index[ant]].antennas[sim_ant_index[ant]].view_ang[:]))
                    arrival = np.asarray(AraTree2.report.stations[0].strings[sim_st_index[ant]].antennas[sim_ant_index[ant]].arrival_time[:]) * 1e9        
    
                    self.rec_ang[:len(rec), ant, evt] = rec
                    self.view_ang[:len(view), ant, evt] = view
                    self.arrival_time[:len(arrival), ant, evt] = arrival
                    del rec, view, arrival
            del AraTree2

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

    def __init__(self, n0, nf, l, verbose = False):

        self.verbose = verbose
        self.ara_sim = ROOT
        self.ice_model = np.array([n0, nf, l], dtype = float)
        if self.verbose:
            print('ice model:', self.ice_model)

        # header
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../source/AraSim/RayTrace.h"')
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../source/AraSim/RayTrace_IceModels.h"')
        self.ara_sim.gInterpreter.ProcessLine('#include "'+os.environ.get('ARA_UTIL_INSTALL_DIR')+'/../source/AraSim/Vector.h"')

        # attenuation model
        self.ara_sim.gInterpreter.ProcessLine('auto atten_model = boost::shared_ptr<basicAttenuationModel>( new basicAttenuationModel );')

        #exponential model
        self.ara_sim.gInterpreter.ProcessLine(f'auto refr_model = boost::shared_ptr<exponentialRefractiveIndex>(new exponentialRefractiveIndex({n0},{nf},{l}));') 
        #1.249,1.774,0.0163 #(1.353,1.78,0.0160)

        # link ray tracing model
        self.ara_sim.gInterpreter.ProcessLine('RayTrace::TraceFinder tf(refr_model, atten_model);')

        # link ray trace output format
        self.ara_sim.gInterpreter.ProcessLine('Vector src; Vector rec; std::vector<RayTrace::TraceRecord> paths;')

    def get_src_trg_position(self, st, yrs, theta_bin = np.linspace(0.5, 179.5, 179 + 1), phi_bin = np.linspace(-179.5, 179.5, 359 + 1), radius_bin = np.array([41,300]), debug = False):

        from tools.ara_data_load import ara_geom_loader
        ara_geom = ara_geom_loader(st, yrs)
        self.ant_pos = ara_geom.get_ant_xyz()
        del ara_geom

        self.theta_bin = theta_bin
        self.phi_bin = phi_bin
        self.radius_bin = radius_bin
        if self.verbose:
            print(f'antenna position (A{st} Y{yrs}):', self.ant_pos.T)
            print('theta range:', self.theta_bin)
            print('phi range:', self.phi_bin)
            print('radius range:', self.radius_bin)
        theta_radian = np.radians(self.theta_bin)
        phi_radian = np.radians(self.phi_bin)

        r_4d = np.tile(self.radius_bin[np.newaxis, np.newaxis, :, np.newaxis], (len(self.theta_bin), len(self.phi_bin), 1, num_ants))
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

        arr_time = np.full((2), np.nan, dtype = float)

        sol_num = 0
        for sol in self.ara_sim.paths:
            arr_time[sol_num] = sol.pathTime*1e9
            if debug and self.verbose:
                print(f'Path time: {sol.pathTime*1e9} ns, Path length: {sol.pathLen} m, Attenuation: {sol.attenuation} m, Miss: {sol.miss} m')
            sol_num += 1
            
        return arr_time

    def get_arrival_time_table(self):

        self.num_ray_sol = 2
        arr_time_table = np.full((len(self.theta_bin), len(self.phi_bin), len(self.radius_bin), num_ants, self.num_ray_sol), np.nan, dtype = float)
        if self.verbose:
            print('arrival time table size:', arr_time_table.shape)

        for t in tqdm(range(len(self.theta_bin)), ascii = False):
            for p in range(len(self.phi_bin)):
                for r in range (len(self.radius_bin)):
                    for a in range(num_ants):

                        arr_time_table[t, p, r, a] = self.get_ray_solution(self.trg_r[t, p, r, a], self.src_z[t, p, r, a], self.trg_z[a])

        return arr_time_table



















# Got the idea from Kaeli Hughes.
# kahughes@uchicago.edu

import numpy as np
import os
import ROOT
import ctypes
from tqdm import tqdm

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAra.so")

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_raytrace_loader:

    def __init__(self, n0, nf, l):

        self.ara_sim = ROOT
        self.ice_model = np.array([n0, nf, l], dtype = float)
        #print('ice model:', self.ice_model)

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

    def get_src_trg_position(self, st, yrs, theta_bin = np.linspace(-89.5, 89.5, 179 + 1), phi_bin = np.linspace(-179.5, 179.5, 359 + 1), radius_bin = np.array([41,300])):

        from tools.ara_data_load import ara_geom_loader
        ara_geom = ara_geom_loader(st, yrs)
        self.ant_pos = ara_geom.get_ant_xyz()
        del ara_geom

        self.theta_bin = theta_bin
        self.phi_bin = phi_bin
        self.radius_bin = radius_bin
        #print('theta range:', self.theta_bin)
        #print('phi range:', self.phi_bin)
        #print('radius range:', self.radius_bin)

        r_4d = np.tile(self.radius_bin[np.newaxis, np.newaxis, :, np.newaxis], (len(self.theta_bin), len(self.phi_bin), 1, num_ants))
        x_4d = r_4d * np.sin(self.theta_bin)[:, np.newaxis, np.newaxis, np.newaxis] * np.cos(self.phi_bin)[np.newaxis, :, np.newaxis, np.newaxis]
        y_4d = r_4d * np.sin(self.theta_bin)[:, np.newaxis, np.newaxis, np.newaxis] * np.sin(self.phi_bin)[np.newaxis, :, np.newaxis, np.newaxis]
        z_4d = r_4d * np.cos(self.theta_bin)[:, np.newaxis, np.newaxis, np.newaxis]

        self.trg_r = np.sqrt((x_4d - self.ant_pos[np.newaxis, np.newaxis, np.newaxis, 0, :])**2 + (y_4d - self.ant_pos[np.newaxis, np.newaxis, np.newaxis, 1, :])**2)
        self.trg_z = np.copy(self.ant_pos[2])

        self.src_r = np.full((1), 0, dtype = float)
        self.src_z = z_4d + np.nanmean(self.ant_pos[2])
        del r_4d, x_4d, y_4d, z_4d

    def get_ray_solution(self, trg_r, src_z, trg_z = -200):

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
            sol_num += 1
            #{'tof':path.pathTime*1e9,'dist': path.pathLen, 'atten': path.attenuation,'miss': path.miss}            

        return arr_time

    def get_arrival_time_table(self):

        self.num_ray_sol = 2
        arr_time_table = np.full((len(self.theta_bin), len(self.phi_bin), len(self.radius_bin), num_ants, self.num_ray_sol), np.nan, dtype = float)
        #print('arrival time table size:', arr_time_table.shape)

        for t in tqdm(range(len(self.theta_bin)), ascii = True):
            for p in range(len(self.phi_bin)):
                for r in range (len(self.radius_bin)):
                    for a in range(num_ants):

                        arr_time_table[t, p, r, a] = self.get_ray_solution(self.trg_r[t, p, r, a], self.src_z[t, p, r, a], self.trg_z[a])

        return arr_time_table



















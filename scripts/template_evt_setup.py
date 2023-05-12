import os, sys
import numpy as np
from tqdm import tqdm
import ROOT

print('Lib Loading Complete!')

st = int(sys.argv[1])
config = int(sys.argv[2])
if config > 5:
    year = int(2018)
else:
    year = int(2015)

## ARA station coordinate
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

geomTool = ROOT.AraGeomTool.Instance()
st_info = geomTool.getStationInfo(st, year)
ant_pos = np.full((16, 3), np.nan, dtype = float)
for ant in range(16):
    ant_pos[ant, 0] = st_info.getAntennaInfo(ant).antLocation[0] + 10000
    ant_pos[ant, 1] = st_info.getAntennaInfo(ant).antLocation[1] + 10000
    ant_pos[ant, 2] = st_info.getAntennaInfo(ant).antLocation[2]
print('antenna location:',ant_pos)

#nu flavor
nu_bar = ['Nu', 'Nu']
nu_fla = ['NuE', 'NuMu']
nu_curr = ['CC', 'NC']
shower = ['EM', 'HAD']
elst = np.array([0.1, 0.9])

# energy
en_scale = 400
en_range = np.array([18.5])
energy = (en_range * 10 + en_scale).astype(int)

# angles
cheren_ang = 56.03
view_ang = np.arange(0, 4.1, 2)
rec_ang = np.arange(0, -61, -20)
radius = np.array([300])

#antenna center x,y,z
ant_pos_mean = np.nanmean(ant_pos, axis = 0)

# unit vector
theta_unit_vec = np.array([0, 0, 1])
phi_unit_vec = np.array([1, 0, 0])

#antenna pos path
output_path = f'/home/mkim/analysis/MF_filters/sim/ARA0{st}/sim_temp_setup_full/temp_A{st}_R{config}_setup.txt'
param_path = f'/home/mkim/analysis/MF_filters/sim/ARA0{st}/sim_temp_setup_full/temp_A{st}_R{config}_setup_parameter.txt'
print(output_path)
print(param_path)

output = open(output_path, 'w')
param = open(param_path, 'w')
lines = f'Event_id Flavor CC_NC Elst Ant_Res Off_cone Useful_Ch'+'\n'
param.write(lines)

counts = 0
for i in tqdm(range(16)):
    for b in range(len(rec_ang)):
    
        #vertex position x,y,z
        ver_pos = np.copy(ant_pos[i])
        ver_pos[0] += radius * np.cos(np.radians(-1 * rec_ang[b]))
        ver_pos[2] -= radius * np.sin(np.radians(-1 * rec_ang[b]))

        # antenna center and vertex vector
        ant_cen_ver_vec = ver_pos - ant_pos_mean

        #elevation calculation to center of antenna
        AB = np.nansum(theta_unit_vec * ant_cen_ver_vec) # A.B
        ABabs = np.sqrt(np.nansum(theta_unit_vec ** 2)) * np.sqrt(np.nansum(ant_cen_ver_vec ** 2)) # |AB|
        zen_ang = np.degrees(np.arccos(AB / ABabs))
        zen_rad = np.radians(zen_ang)

        #phi calculation to center of antenna
        AD = np.nansum(phi_unit_vec[:2] * ant_cen_ver_vec[:2]) # A.D
        ADabs = np.sqrt(np.nansum(phi_unit_vec[:2] ** 2)) * np.sqrt(np.nansum(ant_cen_ver_vec[:2] ** 2)) # |AD|
        if ver_pos[1] >= ant_pos_mean[1]:
            phi_ang = np.degrees(np.arccos(AD / ADabs))
        elif ver_pos[1] < ant_pos_mean[1]:
            phi_ang = 360-np.degrees(np.arccos(AD / ADabs))
        phi_rad = np.radians(phi_ang)

        #R calculation
        R = np.sqrt(np.nansum(ant_cen_ver_vec ** 2))

        for c in range(len(view_ang)):
        
            if i < 8: # vpol
                #Neutrino theta angle
                nu_theta_ang = cheren_ang + 90 + rec_ang[b]
                nu_theta_rad = np.radians(nu_theta_ang)

                #Offcone theta angle
                cone_theta_ang = nu_theta_ang + view_ang[c]
                cone_theta_rad = np.radians(cone_theta_ang)

                nu_dir_theta = np.copy(cone_theta_rad)
                nu_dir_phi = np.pi
                nu_dir_theta_ang = np.copy(cone_theta_ang)
                nu_dir_phi_ang = np.degrees(np.pi)

            if i > 7: # hpol
                # vertex to path point
                r_ver_path = radius * np.tan(np.radians(cheren_ang + view_ang[c]))                

                #Neutrino path point
                nu_phi_path = np.copy(ant_pos_mean)
                nu_phi_path[1] -= r_ver_path

                #phi_vector
                phi_vec = nu_phi_path - ver_pos

                #cosine angle calculation for elevation angle
                AE = np.nansum(theta_unit_vec * phi_vec) # A.E
                AEabs = np.sqrt(np.nansum(theta_unit_vec ** 2)) * np.sqrt(np.nansum(phi_vec ** 2)) # |AE|
                hpol_ele_ang = np.degrees(np.arccos(AE / AEabs))
                hpol_ele_rad = np.radians(hpol_ele_ang)	

                #cosine angle calculation for phi angle
                AF = np.nansum(phi_unit_vec[:2] * phi_vec[:2]) # A.F
                AFabs = np.sqrt(np.nansum(phi_unit_vec[:2] ** 2)) * np.sqrt(np.nansum(phi_vec[:2] ** 2)) # |AF|
                hpol_phi_ang = 360 - np.degrees(np.arccos(AF / AFabs))
                hpol_phi_rad = np.radians(hpol_phi_ang)

                nu_dir_theta = np.copy(hpol_ele_rad)
                nu_dir_phi = np.copy(hpol_phi_rad)
                nu_dir_theta_ang = np.copy(hpol_ele_ang)
                nu_dir_phi_ang = np.copy(hpol_phi_ang)

            # write
            for s in range(2):
                if s == 0:
                    curr = 1
                else:
                    curr = 0
                
                #          entry,    flavor,    bar,   energy,     cc/nc, radius,pos zen,  pos phi,  nu zen,        nu phi,      elst
                lines = f'{counts} {int(s + 1)} {0} {energy[0]} {int(curr)} {R} {zen_rad} {phi_rad} {nu_dir_theta} {nu_dir_phi} {elst[s]}'+'\n'
                output.write(lines) 
         
                # Reference
                #lines = f'Event_id Flavor CC_NC Elst Ant_Res Off_cone Useful_Ch'+'\n'           
                lines = f'{counts} {nu_fla[s]} {nu_curr[s]} {elst[s]} {rec_ang[b]} {view_ang[c]} {i}'+'\n'
                param.write(lines)            
                    
                counts += 1


output.close()
param.close()
print(counts) 
print('Done!')

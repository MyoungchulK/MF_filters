import os, sys
import numpy as np
from tqdm import tqdm

print('Lib Loading Complete!')

if len (sys.argv) !=2:
        Usage='Usage = python3 %s <station ex)2>'%(sys.argv[0])
        print(Usage)
        sys.exit(1)

st=int(sys.argv[1])

if st == 2:
    ant_pos = np.full((16, 3), np.nan, dtype = float)
    ant_pos[6-1] = np.array([10004.9, 9989.6, -189.4], dtype = float)
    ant_pos[14-1] = np.array([10004.9, 9989.6, -186.115], dtype = float)
    ant_pos[2-1] = np.array([10004.9, 9989.6, -170.347], dtype = float)
    ant_pos[10-1] = np.array([10004.9, 9989.6, -167.428], dtype = float)
    ant_pos[7-1] = np.array([9997.42, 10009.4, -191.242], dtype = float)
    ant_pos[15-1] = np.array([9997.42, 10009.4, -187.522], dtype = float)
    ant_pos[3-1] = np.array([9997.42, 10009.4, -171.589], dtype = float)
    ant_pos[11-1] = np.array([9997.42, 10009.4, -168.468], dtype = float)
    ant_pos[8-1] = np.array([9992.16, 9995.94, -194.266], dtype = float)
    ant_pos[16-1] = np.array([9992.16, 9995.94, -190.981], dtype = float)
    ant_pos[4-1] = np.array([9992.16, 9995.94, -175.377], dtype = float)
    ant_pos[12-1] = np.array([9992.16, 9995.94, -172.42], dtype = float)
    ant_pos[5-1] = np.array([10010.6, 10002.3, -189.502], dtype = float)
    ant_pos[13-1] = np.array([10010.6, 10002.3, -186.546], dtype = float)
    ant_pos[1-1] = np.array([10010.6, 10002.3, -170.247], dtype = float)
    ant_pos[9-1] = np.array([10010.6, 10002.3, -167.492], dtype = float)
    print(ant_pos)

#nu flavor
nu_bar = ['Nu', 'Nu']
nu_fla = ['NuE', 'NuMu']
nu_curr = ['CC', 'NC']
shower = ['EM', 'HAD']

#inelsity
elst = np.array([0.1, 0.9])
print(elst)

# energy
Energy_setting=400
#Energy_origin=np.arange(16, 22, 1, dtype = int)
Energy_origin = np.array([18], dtype = int)
print(Energy_origin)
Energy=Energy_origin * 10 + Energy_setting
Energy=Energy.astype(int)
print(Energy)

# antenna response
#Gain_angle=np.arange(0,-91,-1, dtype = int)
Gain_angle=np.arange(0,-61,-10, dtype = int)
print(Gain_angle)

# distance
Distance=200 # 200 m
print(Distance)

# veiw angle
Cherenkov_angle=56.03
#Offcone_angle=np.arange(0,11,1, dtype = int)
Offcone_angle=np.arange(0,4.1,0.5)
print(Offcone_angle)


# calculation!
#antenna center x,y,z
ant_pos_mean = np.nanmean(ant_pos, axis = 0)
ant_avgx=ant_pos_mean[0]
ant_avgy=ant_pos_mean[1]
ant_avgz=ant_pos_mean[2]
#print('Center of antenna is',ant_avgx,ant_avgy,ant_avgz)

# z unit vector
theta_unit_x=0
theta_unit_y=0
theta_unit_z=1

# x unit vector
phi_unit_x=1
phi_unit_y=0
#phi_unit_z=0

#antenna pos path
output_path = os.getcwd()+f'/../../sim/sim_temp/temp_A{st}_setup.txt'
param_path = os.getcwd()+f'/../../sim/sim_tmep/temp_A{st}_setup_parameter.txt'
print(output_path)
print(param_path)

output = open(output_path, 'w')
param = open(param_path, 'w')

lines = f'Event_id Flavor CC_NC Elst Ant_Res Off_cone Useful_Ch'+'\n'
param.write(lines)

counts = 0
#for a in range(len(Energy)):
for i in tqdm(range(16)):

    #selected antenna x,y,z
    ant_indix=ant_pos[i,0]
    ant_indiy=ant_pos[i,1]
    ant_indiz=ant_pos[i,2]
    #print('Antenna',Antenna[i],'is',ant_indix,ant_indiy,ant_indiz) 

    for b in range(len(Gain_angle)):

        #vertex position x,y,z
        ver_x=ant_indix + Distance*np.cos(np.radians(-1*Gain_angle[b]))
        ver_y=ant_indiy
        ver_z=ant_indiz - Distance*np.sin(np.radians(-1*Gain_angle[b]))
        #print('Vertex',ver_x,ver_y,ver_z)

        # antenna center and vertex vector
        antcen_ver_x=ver_x - ant_avgx
        antcen_ver_y=ver_y - ant_avgy
        antcen_ver_z=ver_z - ant_avgz

        #elevation calculation to center of antenna
        #cosine angle calculation for elevation angle
        AB = theta_unit_x*antcen_ver_x + theta_unit_y*antcen_ver_y + theta_unit_z*antcen_ver_z # A.B
        ABabs = np.sqrt(theta_unit_x**2 + theta_unit_y**2 + theta_unit_z**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2 + antcen_ver_z**2) # |AB|
        elevation_ang=90-np.degrees(np.arccos(AB/ABabs))
        elevation_rad=np.radians(elevation_ang)
        #print('Elevation angle for vertex is',elevation_ang,elevation_rad)

        #phi calculation to center of antenna
        #cosine angle calculation for phi angle
        AD = phi_unit_x*antcen_ver_x + phi_unit_y*antcen_ver_y# + phi_unit_z*antcen_ver_z # A.D
        #ADabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2 + phi_unit_z**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2 + antcen_ver_z**2) # |AD|
        ADabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2) # |AB|
        if ver_y >= ant_avgy:
            phi_ang=np.degrees(np.arccos(AD/ADabs))
        elif ver_y < ant_avgy:
            phi_ang=360-np.degrees(np.arccos(AD/ADabs))
        phi_rad=np.radians(phi_ang)
        #print('Phi angle for vertex is',phi_ang,phi_rad)

        #R calculation
        R=np.sqrt((ver_x-ant_avgx)**2+(ver_y-ant_avgy)**2+(ver_z-ant_avgz)**2)
        #print('R distance for vertex is',R)

        for c in range(len(Offcone_angle)):
            
            if i < 8:
                #Neutrino theta angle
                Nu_theta_ang=Cherenkov_angle+90+Gain_angle[b]
                Nu_theta_rad=np.radians(Nu_theta_ang)
                #print('Vpol Neutrino Dir is',Nu_theta_ang,Nu_theta_rad)

                #Offcone theta angle
                Offcone_theta_ang=Nu_theta_ang+Offcone_angle[c]
                Offcone_theta_rad=np.radians(Offcone_theta_ang)
                #print('Offcone Neutrino Dir is',Offcone_theta_ang,Offcone_theta_rad)

                Nu_dir_theta = np.copy(Offcone_theta_rad)
                Nu_dir_phi = np.pi
                Nu_dir_theta_ang = np.copy(Offcone_theta_ang)
                Nu_dir_phi_ang = np.degrees(np.pi)

            if i > 7:
                # vertex to path point
                R_ver_path = Distance*np.tan(np.radians(Cherenkov_angle+Offcone_angle[c]))                

                #Neutrino path point
                Nu_phi_path_x = ant_avgx
                Nu_phi_path_y = ant_avgy - R_ver_path
                Nu_phi_path_z = ant_avgz

                #phi_vector
                phi_Vec_x = Nu_phi_path_x - ver_x
                phi_Vec_y = Nu_phi_path_y - ver_y
                phi_Vec_z = Nu_phi_path_z - ver_z

                #cosine angle calculation for elevation angle
                AE = theta_unit_x*phi_Vec_x + theta_unit_y*phi_Vec_y + theta_unit_z*phi_Vec_z # A.B
                AEabs = np.sqrt(theta_unit_x**2 + theta_unit_y**2 + theta_unit_z**2)*np.sqrt(phi_Vec_x**2 + phi_Vec_y**2 + phi_Vec_z**2) # |AB|
                Hpol_elevation_ang=np.degrees(np.arccos(AE/AEabs))
                Hpol_elevation_rad=np.radians(Hpol_elevation_ang)	
                #print('Hpol theta angle is',Hpol_elevation_ang)

                #cosine angle calculation for phi angle
                AF = phi_unit_x*phi_Vec_x + phi_unit_y*phi_Vec_y
                AFabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2)*np.sqrt(phi_Vec_x**2 + phi_Vec_y**2) # |AB|
                Hpol_phi_ang=360-np.degrees(np.arccos(AF/AFabs))
                Hpol_phi_rad=np.radians(Hpol_phi_ang)
                #print('Hpol phi angle is',Hpol_phi_ang)

                Nu_dir_theta = np.copy(Hpol_elevation_rad)
                Nu_dir_phi = np.copy(Hpol_phi_rad)
                Nu_dir_theta_ang = np.copy(Hpol_elevation_ang)
                Nu_dir_phi_ang = np.copy(Hpol_phi_ang)

            # write
            for s in range(2):
                if s == 0:
                    curr = 1
                else:
                    curr = 0

                lines = f'{counts} {s+1} {0} {Energy[0]} {curr} {R} {elevation_rad} {phi_rad} {Nu_dir_theta} {Nu_dir_phi} {elst[s]}'+'\n'
                output.write(lines) 
         
                # Reference
                #lines = f'Event_id Flavor CC_NC Elst Ant_Res Off_cone Useful_Ch'+'\n'           
                lines = f'{counts} {nu_fla[s]} {nu_curr[s]} {elst[s]} {Gain_angle[b]} {Offcone_angle[c]} {i}'+'\n'
                param.write(lines)            
                    
                counts += 1


output.close()
param.close()
print(counts) 
print('Def Loading Complete!')

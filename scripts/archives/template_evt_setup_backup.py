import os, sys
import numpy as np
from tqdm import tqdm

print('Lib Loading Complete!')

if len (sys.argv) !=2:
        Usage='Usage = python3 %s <station ex)2>'%(sys.argv[0])
        print(Usage)
        sys.exit(1)

st=str(sys.argv[1])

#antenna pos path
output_path= os.getcwd()+'/../sim/temp_A2_setup.txt'
print(output_path)

if st == 2:
    ant_pos = np.full(16, 3), np.nan, dtype = float)
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
nu_bar = ['Nu']
if sho_argv == 'EM':
    nu_fla = ['NuE']
    nu_curr = ['CC']
    shower = ['EM']
if sho_argv == 'HAD':
    nu_fla = ['NuMu']
    nu_curr = ['NC']
    shower = ['HAD']

#inelsity
if sho_argv == 'HAD':
    elst = np.arange(0.1, 1 + 0001, 0.1)
if sho_argv == 'EM':
    elst = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
elst = np.round(elst,1)
print(elst)

#input variable
Energy_setting=400
Energy_origin=np.arange(16, 22, 1, dtype = int)
print(Energy_origin)
Energy=Energy_origin * 10 + Energy_setting
Energy=Energy.astype(int)
print(Energy)

# antenna response
Gain_angle=np.arange(0,-91,-1, dtype = int)
print(Gain_angle)

# distance
Distance=500 # 5km
print(Distance)

# veiw angle
Cherenkov_angle=56.03
Offcone_angle=np.arange(0,11,1 dtype = int)
print(Offcone_angle)


# calculation!
#antenna center x,y,z
ant_pos_mean = np.nanmean(ant_pos, axis = 0)
ant_avgx=ant_pos_mean[0]
ant_avgy=ant_pos_mean[1]
ant_avgz=anr_pos_mean[2]
#print('Center of antenna is',ant_avgx,ant_avgy,ant_avgz)

for i in tqdm(range(16)):

    

  for a in range(len(Energy)):
    for b in range(len(Gain_angle)):
        for c in range(len(Offcone_angle)):
            
            #antenna center x,y,z
            ant_pos_mean = np.nanmean(ant_pos, axis = 0)
            ant_avgx=ant_pos_mean[0]
            ant_avgy=ant_pos_mean[1]
            ant_avgz=anr_pos_mean[2]
            #print('Center of antenna is',ant_avgx,ant_avgy,ant_avgz)

            #selected antenna x,y,z
            ant_indix=ant_pos[i,0]
            ant_indiy=ant_pos[i,1]
            ant_indiz=ant_pos[i,2]
            #print('Antenna',Antenna[i],'is',ant_indix,ant_indiy,ant_indiz)

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
            # z unit vector
            theta_unit_x=0
            theta_unit_y=0
            theta_unit_z=1
            #cosine angle calculation for elevation angle
            AB = theta_unit_x*antcen_ver_x + theta_unit_y*antcen_ver_y + theta_unit_z*antcen_ver_z # A.B
            ABabs = np.sqrt(theta_unit_x**2 + theta_unit_y**2 + theta_unit_z**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2 + antcen_ver_z**2) # |AB|
            elevation_ang=90-np.degrees(np.arccos(AB/ABabs))
            elevation_rad=np.radians(elevation_ang)
            #print('###')
            #print('Elevation angle is',elevation_ang,elevation_rad)

            #phi calculation to center of antenna 
            phi_unit_x=1
            phi_unit_y=0
            #phi_unit_z=0
            #cosine angle calculation for phi angle
            AD = phi_unit_x*antcen_ver_x + phi_unit_y*antcen_ver_y# + phi_unit_z*antcen_ver_z # A.D
            #ADabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2 + phi_unit_z**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2 + antcen_ver_z**2) # |AD|
            ADabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2)*np.sqrt(antcen_ver_x**2 + antcen_ver_y**2) # |AB|
            
            if ver_y >= ant_avgy:
                phi_ang=np.degrees(np.arccos(AD/ADabs))
            elif ver_y < ant_avgy:
                phi_ang=360-np.degrees(np.arccos(AD/ADabs))

            phi_rad=np.radians(phi_ang)
            #print('Phi angle is',phi_ang,phi_rad)

            #R calculation
            R=np.sqrt((ver_x-ant_avgx)**2+(ver_y-ant_avgy)**2+(ver_z-ant_avgz)**2)
            #print('R distance is',R)

            if Antenna_num[i][-1] == 'V':
                #Neutrino theta angle
                Nu_theta_ang=Cherenkov_angle+90+Gain_angle[b]
                Nu_theta_rad=np.radians(Nu_theta_ang)
                #print('Vpol Neutrino Dir is',Nu_theta_ang,Nu_theta_rad)

                #Offcone theta angle
                Offcone_theta_ang=Nu_theta_ang+Offcone_angle[c]
                Offcone_theta_rad=np.radians(Offcone_theta_ang)
                #print('Offcone Neutrino Dir is',Offcone_theta_ang,Offcone_theta_rad)

            if Antenna_num[i][-1] == 'H':
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

            #energy
            #print('Energy is',Energy[a])

            #print('###')
            os.chdir(mainpath)
            with open(simtxt_name, 'r') as file:
                simtxt = file.readlines()
            
                simtxt[14]=simtxt[14][0:9]+str(Energy[a])+'\n'
                simtxt[37]=simtxt[37][0:8]+str(R)+'\n'
                simtxt[38]=simtxt[38][0:12]+str(elevation_rad)+'\n'
                simtxt[39]=simtxt[39][0:10]+str(phi_rad)+'\n'
                if Antenna_num[i][-1] == 'V':
                    simtxt[43]=simtxt[43][0:10]+str(Offcone_theta_rad)+'\n'
                if Antenna_num[i][-1] == 'H':
                    simtxt[43]=simtxt[43][0:10]+str(Hpol_elevation_rad)+'\n'
                    simtxt[46]=simtxt[46][0:8]+str(Hpol_phi_rad)+'\n'

                for d in range(len(nu_fla)):
                    if sho_argv == 'EM':
                        simtxt[18]=simtxt[18][0:14]+str(0)+'\n'
                    if sho_argv == 'HAD': 
                        simtxt[18]=simtxt[18][0:14]+str(1)+'\n'

                    for e in range(len(nu_bar)):
                        simtxt[17]=simtxt[17][0:21]+str(0)+'\n'

                        for f in range(len(nu_curr)):
                            if sho_argv == 'EM':
                                simtxt[19]=simtxt[19][0:15]+str(1)+'\n'
                            if sho_argv == 'HAD':
                                simtxt[19]=simtxt[19][0:15]+str(0)+'\n'

                            for g in range(len(shower)):
                                if sho_argv == 'EM':
                                    simtxt[22]=simtxt[22][0:12]+str(0)+'\n'
                                if sho_argv == 'HAD':
                                    simtxt[22]=simtxt[22][0:12]+str(1)+'\n'

                                for h in range(len(elst)):
                                    #if sho_argv == 'EM':
                                    #    simtxt[94]=simtxt[94][0:8]+str(1 - elst[h])+'\n'
                                    #if sho_argv == 'HAD':
                                    simtxt[32]=simtxt[32][0:8]+str(elst[h])+'\n'
                
                                    if not os.path.exists(simtxt_resultpath):
                                        os.makedirs(simtxt_resultpath)
                                    os.chdir(simtxt_resultpath)
           
                                    new_simtxt_name='setup.template.S.N0.E%s.D%s.A%s.O%s.%s.%s.%s.%s.%s.El%s.txt'%(Energy_origin[a],Distance,Gain_angle[b],Offcone_angle[c],Antenna_num[i],nu_fla[0],nu_bar[0],nu_curr[0],shower[0],elst_name[h])
                                    #print(new_simtxt_name)
                                    with open(new_simtxt_name, 'w') as file:
                                        file.writelines( simtxt )
                
print('Loop is done !')

print('Def Loading Complete!')

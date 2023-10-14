##
# @file check_ch_mapping.py
#
# @section Created on 09/14/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to check channle mapping

import os, sys
import numpy as np
import h5py
import click # 'pip3 install click' will make you very happy
import ROOT

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
num_rf_chs = 16

class ara_geom_loader:

    def __init__(self, st, yrs, verbose = False):

        # create a geomtool
        self.geomTool = ROOT.AraGeomTool.Instance()
        self.st_info = self.geomTool.getStationInfo(st, yrs)
        self.verbose = verbose

    def get_ele_ch_idx(self):

        ele_ch_idx = np.full((num_rf_chs), 0, dtype = int)
        for ant in range(num_rf_chs):
            ele_ch_idx[ant] = self.st_info.getAntennaInfo(ant).daqChanNum
        if self.verbose:
            print('electronic channel:',ele_ch_idx)
        return ele_ch_idx

    def get_pol_ch_idx(self):

        pol_ch_idx = np.full((num_rf_chs), 0, dtype = int)
        for ant in range(num_rf_chs):
            pol_ch_idx[ant] = self.st_info.getAntennaInfo(ant).polType
        if self.verbose:
            print('polarization type:',pol_ch_idx)
        return pol_ch_idx

    def get_trig_ch_idx(self):

        trig_ch_idx = np.full((num_rf_chs), 0, dtype = int)
        for ant in range(num_rf_chs):
            trig_ch_idx[ant] = self.st_info.getAntennaInfo(ant).getTrigChan()
        if self.verbose:
            print('trigger channel:',trig_ch_idx)
        return trig_ch_idx

    def get_ant_xyz(self):

        ant_xyz = np.full((3, num_rf_chs), np.nan, dtype = float)
        for ant in range(num_rf_chs):
            ant_xyz[0, ant] = self.st_info.getAntennaInfo(ant).antLocation[0]
            ant_xyz[1, ant] = self.st_info.getAntennaInfo(ant).antLocation[1]
            ant_xyz[2, ant] = self.st_info.getAntennaInfo(ant).antLocation[2]
        if self.verbose:
            print('antenna location:',ant_xyz)
        return ant_xyz

    def get_cable_delay(self):
        cable_delay = np.full((num_rf_chs), np.nan, dtype = float)
        for ant in range(num_rf_chs):
            cable_delay[ant] = self.st_info.getAntennaInfo(ant).getCableDelay()
        if self.verbose:
            print('cable delay:',cable_delay)
        return cable_delay


@click.command()
@click.option('-s', '--st', type = int, help = 'ex) 2')
@click.option('-y', '--yrs', type = int, help = 'ex) 2013')
@click.option('-o', '--output_path', type = str, help = 'ex) /home/mkim/')
def main(st, yrs, output_path):
    """! This is designed to check channle mapping

    @param st  integer. station id
    @param yrs  integer. year
    """
    
    rf_ch = np.arange(num_rf_chs, dtype = int)
    print('rf channel:', rf_ch)
    ara_geom = ara_geom_loader(st, yrs, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    pol_ch = ara_geom.get_pol_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    ant_xyz = ara_geom.get_ant_xyz()
    cable_delay = ara_geom.get_cable_delay()
    
    ## create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(f'{output_path}A{st}_Y{yrs}_ch_mapping.h5', 'w')
    hf.create_dataset('rf_ch', data=rf_ch, compression="gzip", compression_opts=9)
    hf.create_dataset('ele_ch', data=ele_ch, compression="gzip", compression_opts=9)
    hf.create_dataset('pol_ch', data=pol_ch, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_ch', data=trig_ch, compression="gzip", compression_opts=9)
    hf.create_dataset('ant_xyz', data=ant_xyz, compression="gzip", compression_opts=9)
    hf.create_dataset('cable_delay', data=cable_delay, compression="gzip", compression_opts=9)
    hf.close()
    print('Done!')

if __name__ == "__main__":

    main()
















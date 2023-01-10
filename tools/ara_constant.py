##
# @file ara_constant.py
#
# @section Created on 01/10/2023, mkim@icecube.wisc.edu
#
# @brief all the constant values we are often use. In the future, we hsould call them from AraRoot

import numpy as np

class ara_const:

    def __init__(self):

        self.ANTS_PER_ATRI = 20
        self.CAL_ANTS_PER_ATRI = 4
        self.DDA_PER_ATRI = 4
        self.BLOCKS_PER_DDA = 512
        self.RFCHAN_PER_DDA = 8
        self.TDA_PER_ATRI = 4
        self.ANTS_PER_TDA = 4
        self.L2_PER_TDA = 4 #Wrong?!
        self.THRESHOLDS_PER_ATRI = 16 # TDA_PER_ATRI*ANTS_PER_TDA
        self.SAMPLES_PER_BLOCK = 64
        self.MAX_TRIG_BLOCKS = 4
        self.CHANNELS_PER_ATRI = 32 # DDA_PER_ATRI*RFCHAN_PER_DDA

        # implementation
        self.SAMPLES_PER_DDA = 32768 # SAMPLES_PER_BLOCK*BLOCKS_PER_DDA
        self.USEFUL_CHAN_PER_STATION = 16
        self.BUFFER_BIT_RANGE = 4096 # 12bits
        self.POLARIZATION = 2
        self.TRIGGER_TYPE = 3
        

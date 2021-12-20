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

        # calibration mode
        kNoCalib                        = 0 #The 260 samples straight from raw data
        kJustUnwrap                     = 1 #The X good samples from raw data (260-hitbus)
        kJustPed                        = 2 #Just subtract peds
        kADC                            = 3 #Same as kNoCalib -- i.e. useless
        kVoltageTime                    = 4 #Using 1 and 2.6
        kFirstCalib                     = 5 #First attempt at a calibration by RJN
        kFirstCalibPlusCables           = 6 #Same as First Calib but also doing the cable delays
        kSecondCalib                    = 7 #Same as first calib but also doing the clock alignment
        kSecondCalibPlusCables          = 8 #Same as second calib but also doing the clock alignment
        kSecondCalibPlusCablesUnDiplexed = 9 #Same as secondCalibPlusCableDelays but with the undiplexing of diplexed channels in ARA_STATION1
        kLatestCalib                    = 9 #Currenly this is kSecondCalibPlusCables
        kLatestCalib14to20_Bug          = 10 #new calibration type: everything except voltage calibration. Will reproduce "kLatestCalib" bug present from between ~2014 to September 2020. Use with caution!
        kLatestCalibWithOutZeroMean     = 11 #Performs every calibration except the ADC and Voltage zero meaning
        kOnlyGoodPed                    = 12 #Get the only pedestal values for the corresponding raw WF without bad samples
        kOnlyGoodADC                    = 13 #Get the only raw ADC WF without bad samples and pedestal subtraction

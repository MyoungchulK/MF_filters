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
        
        """
        # calibration mode
        self.kNoCalib                         = ROOT.AraCalType.kNoCalib                         # The 260 samples straight from raw data
        self.kJustUnwrap                      = ROOT.AraCalType.kJustUnwrap                      # The X good samples from raw data (260-hitbus)
        self.kJustPed                         = ROOT.AraCalType.kJustPed                         # Just subtract peds
        self.kADC                             = ROOT.AraCalType.kADC                             # Same as kNoCalib -- i.e. useless
        self.kVoltageTime                     = ROOT.AraCalType.kVoltageTime                     # Using 1 and 2.6
        self.kFirstCalib                      = ROOT.AraCalType.kFirstCalib                      # First attempt at a calibration by RJN
        self.kFirstCalibPlusCables            = ROOT.AraCalType.kFirstCalibPlusCables            # Same as First Calib but also doing the cable delays
        self.kSecondCalib                     = ROOT.AraCalType.kSecondCalib                     # Same as first calib but also doing the clock alignment
        self.kSecondCalibPlusCables           = ROOT.AraCalType.kSecondCalibPlusCables           # Same as second calib but also doing the clock alignment
        self.kSecondCalibPlusCablesUnDiplexed = ROOT.AraCalType.kSecondCalibPlusCablesUnDiplexed # Same as secondCalibPlusCableDelays but with the undiplexing of diplexed channels in ARA_STATION1
        self.kLatestCalib                     = ROOT.AraCalType.kLatestCalib                     # Currenly this is kSecondCalibPlusCables
        self.kLatestCalib14to20_Bug           = ROOT.AraCalType.kLatestCalib14to20_Bug           # new calibration type: everything except voltage calibration. Will reproduce "kLatestCalib" bug present from between ~2014 to September 2020. Use with caution!
        self.kLatestCalibWithOutZeroMean      = ROOT.AraCalType.kLatestCalibWithOutZeroMean      # Performs every calibration except the ADC and Voltage zero meaning
        self.kOnlyPed                         = ROOT.AraCalType.kOnlyPed                         # Get the pedestal values for the corresponding raw WF
        self.kOnlyGoodPed                     = ROOT.AraCalType.kOnlyGoodPed                     # Get the pedestal values for the corresponding raw WF without bad samples
        self.kOnlyGoodADC                     = ROOT.AraCalType.kOnlyGoodADC                     # Get the raw ADC WF without bad samples and pedestal subtraction
        """

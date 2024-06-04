import numpy as np
import argparse

class ofdm_config:

    EXE_MODE = None
    QAM_MODE = 4

    N_FRAMES = 1
    N = None
    CP_LEN = None

    SAMPLE_RATE = 512e6

# FRAME HEAD
    LLTF_GAIN = 30000
    LTF_SEQ_LEN = 64
    LTFseq = np.array([0,0,0,0,0,0,1,1,-1,\
                  -1,1,1,-1,1,-1,1,1,1,\
                  1,1,1,-1,-1,1,1,-1,1,\
                  -1,1,1,1,1,0,1,-1,-1,\
                  1,1,-1,1,-1,1,-1,-1,\
                  -1,-1,-1,1,1,-1,-1,1,\
                  -1,1,-1,1,1,1,1,0,0,0,\
                  0,0])
    LTFseq_inverted = np.array([0,0,\
                            0,0,0,1,1,1,1,-1,1,-1,\
                            1,-1,-1,1,1,-1,-1,-1,\
                            -1,-1,1,-1,1,-1,1,1\
                            -1,-1,1,0,1,1,1,1,-1,\
                            1,-1,1,1,1,-1,-1,1,1,1,\
                            1,1,1,-1,1,-1,1,1,-1,\
                            -1,1,1,0,0,0,0,0,0
                            ])
    
    TRAINseq = np.array([1,1,-1,-1,1,-1,1,1,-1,\
                  -1,1,1,-1,1,-1,1,1,1,\
                  1,1,1,-1,-1,1,1,-1,1,\
                  -1,1,1,1,1,1,1,-1,-1,\
                  1,1,-1,1,-1,1,-1,-1,\
                  -1,-1,-1,1,1,-1,-1,1,\
                  -1,1,-1,1,1,1,1,-1,-1,1,\
                  -1,1])

    LTFcp = LTFseq[32:64]
    # LTF matched filter
    LTF_matched_filter = LTFseq_inverted * LLTF_GAIN
    # Concatenate to form L-LTF
    LLTF = np.concatenate((LTFcp, LTFseq, LTFseq)) * LLTF_GAIN
    LTF_DATA_LEN = 64
    LTF_CP_LEN = 32
    LTF_WITH_CP_LEN = 160

# TRAINING SEQ
    N_TRAINING_SEQ = 2
    TRAINING_SEQ_LEN = None
    train_seq = None
    train_seq_ifft = None
    train_seq_cp = None
    train_symb = None
    train_symbx2 = None
    

# OFDM FRAME
    PILOT_MODE = "ZEROS"
    N_OFDM = None
    N_DATA_SC = None
    N_PILOT_SC = None
    N_GUARD_SC = None
    N_SOLID_SC = None
    N_DATA_QAM_SYMB = None
    PAYLOAD_LEN = None
    FRAME_LEN = None
    OFDM_DATA_INDEX = None
    OFDM_PILOT_INDEX = None
    OFDM_GUARD_INDEX = None
    OFDM_SOLID_INDEX = None

    pilot_seq = None

#GAIN
    QAM_GAIN = 32767
    LLTF_GAIN = 30000
    TRAINING_SEQ_GAIN = 65536
    PILOT_GAIN = 32767

#QAM MAP
    QAM_MAP = None
    
#SIMU CONFIG
    PEAK = 800000000000000
    SNR = 30


    TX_INPUT_QAM_FILE = "./dats/simu/tx_input_qam.txt"
    TX_OUTPUT_QAM_NO_PILOT_FILE = "./dats/rtl_input/tx_output_qam_no_pilot.txt"
    TX_OUTPUT_QAM_PILOT_FILE = "./dats/rtl_input/tx_output_qam_pilot.txt"
    TX_OUTPUT_SINGLE_FRAME_FILE = "./dats/single_frame.txt"
    TX_OUTPUT_FILE = "./dats/rtl_test_input.txt"
    CHANNEL_OUTPUT_FILE = "./dats/simu/channel_output.txt"
    RTL_OUTPUT_I_FILE = "./dats/rtl_output/rtl_output_I.txt"
    RTL_OUTPUT_Q_FILE = "./dats/rtl_output/rtl_output_Q.txt"
    DATS_DIR = "./dats/rx_output/"
    RTL_BITFILE = "/home/xilinx/work/test_AXI/bitfile/design_1_wrapper.bit"
    





    def __init__(self, args):
        # EXE_MODE
        if args.EXE_MODE is not None:
            self.EXE_MODE = args.EXE_MODE
        else:
            print("EXE_MODE not detected, please read usage")
            self.init_ok = -1
            return
        # N

        if args.N is not None:
            if(args.N == 64 or args.N == 2048 or args.N == 4096):
                self.N = args.N
                self.CP_LEN = self.N // 4
                self.N_TRAINING_SEQ = 2
            else:
                print("incorrect N, please read classes.md")
                self.init_ok = -1
                return
        else:
            print("N not detected, please read usage")
            self.init_ok = -1
            return
        if(self.N == 64):
            # N 64 training seq
            self.N_TRAINING_SEQ =2 
            self.TRAINING_SEQ_LEN = 2*(self.CP_LEN + self.N)
            #self.train_seq = self.LTFseq
            self.train_seq = self.TRAINseq * self.TRAINING_SEQ_GAIN
            self.train_seq_ifft = np.fft.ifft((self.train_seq ),self.N)
            self.train_seq_cp = self.train_seq_ifft[self.N-self.CP_LEN:self.N]
            self.train_symb = np.concatenate((self.train_seq_cp,self.train_seq_ifft))
            self.train_symbx2 = np.concatenate((self.train_symb,self.train_symb))

        # QAM_MODE
        if args.QAM_MODE is not None:
            self.QAM_MODE = args.QAM_MODE
        else:
            self.QAM_MODE = 4  # default value
        
        if(self.QAM_MODE == 4):
            self.QAM_MAP = {
                0: (1 + 1j) *  self.QAM_GAIN,
                1: (-1 + 1j) * self.QAM_GAIN,
                2: (1 - 1j) *  self.QAM_GAIN, 
                3: (-1 - 1j) * self.QAM_GAIN
            }
        elif(self.QAM_MODE == 16):
            self.QAM_MAP= {
                0: ( 3 - 3j) * self.QAM_GAIN,
                1: ( 3 - 1j) * self.QAM_GAIN,
                2: ( 3 + 3j) * self.QAM_GAIN, 
                3: ( 3 + 1j) * self.QAM_GAIN,
                4: ( 1 - 3j) * self.QAM_GAIN, 
                5: ( 1 - 1j) * self.QAM_GAIN, 
                6: ( 1 + 3j) * self.QAM_GAIN, 
                7: ( 1 + 1j) * self.QAM_GAIN,
                8: (-3 - 3j) * self.QAM_GAIN, 
                9: (-3 - 1j) * self.QAM_GAIN, 
                10:(-3 + 3j) * self.QAM_GAIN, 
                11:(-3 + 1j) * self.QAM_GAIN,
                12:(-1 - 3j) * self.QAM_GAIN, 
                13:(-1 - 1j) * self.QAM_GAIN, 
                14:(-1 + 3j) * self.QAM_GAIN, 
                15:(-1 + 1j) * self.QAM_GAIN,
            }
        else:
            print("incorrect qam map, plz read classes.md")
            return -1



        # N_OFDM
        if args.N_OFDM is not None:
            self.N_OFDM = args.N_OFDM
        else:
            print("N_OFDM not detected, please read usage")
            self.init_ok = -1
            return

        # PILOT_MODE
        if args.PILOT_MODE is not None:
            self.PILOT_MODE = args.PILOT_MODE
        else:
            self.PILOT_MODE = "ZEROS"  # default value

        if(self.PILOT_MODE == "NO_PILOT"): 
            # no pilot, all sc are data sc
            self.N_DATA_SC = self.N
            self.N_PILOT_SC = 0
            self.N_GUARD_SC = 0

            self.OFDM_SOLID_INDEX = np.arange(self.N)
            self.OFDM_DATA_INDEX = np.arange(self.N)
            self.OFDM_PILOT_INDEX = None
        elif(self.PILOT_MODE == "ZEROS"):
            # only guard interval
            if(self.N == 64):
                self.N_DATA_SC = 52
                self.N_PILOT_SC = 0
                self.N_GUARD_SC = 12

                self.OFDM_GUARD_INDEX = np.array([0,1,2,3,4,5,32,59,60,61,62,63])
                self.OFDM_DATA_INDEX = np.concatenate((np.arange(start=6, stop=32),np.arange(start=33, stop=59)))
                self.OFDM_SOLID_INDEX = np.concatenate((np.arange(start=6, stop=32),np.arange(start=33, stop=59)))
                self.OFDM_PILOT_INDEX = None
            elif(self.N == 2048):
                # for 2048 and 4096, 25% guard interval
                self.N_DATA_SC = 1023
                self.N_GUARD_SC = 1025
                self.N_PILOT_SC = 0

                self.OFDM_GUARD_INDEX = np.concatenate((np.array(start = 0, stop = 512),np.array(1024),np.arange(start=1536,stop=2048)))
                self.OFDM_DATA_INDEX = np.concatenate((np.arange(start=512, stop=1024),np.arange(start=1025, stop=1536)))
                self.OFDM_SOLID_INDEX = np.concatenate((np.arange(start=512, stop=1024),np.arange(start=1025, stop=1536)))
                self.OFDM_PILOT_INDEX = None
            elif(self.N == 4096):
                # for 2048 and 4096, 25% guard interval
                self.N_DATA_SC = 2047
                self.N_GUARD_SC = 2049
                self.N_PILOT_SC = 0

                self.OFDM_GUARD_INDEX = np.concatenate((np.array(start = 0, stop = 1024),np.array(2048),np.arange(start=3072,stop=4096)))
                self.OFDM_DATA_INDEX = np.concatenate((np.arange(start=1024, stop=2048),np.arange(start=2049, stop=3072)))
                self.OFDM_SOLID_INDEX = np.concatenate((np.arange(start=1024, stop=2048),np.arange(start=2049, stop=3072)))
                self.OFDM_PILOT_INDEX = None


        elif(self.PILOT_MODE == "PILOT_MODE_1"):
            # MODE 1 for N = 64:
            # not used yet
            if(self.N == 64):
                print("not used for now")
                self.init_ok = -1
                return
                self.N_DATA_SC = 52
                self.N_PILOT_SC = 12
                self.N_OFDM_PAYLOAD_INDEX = np.concatenate((np.arange(start=6, stop=32),np.arange(start=33, stop=59)))
                self.OFDM_PILOT_INDEX = np.array([0,1,2,3,4,5,32,59,60,61,62,63])
                self.OFDM_DATA_INDEX = np.setdiff1d(self.OFDM_SOLID_INDEX, self.OFDM_PILOT_INDEX)
            else:
                print("not used for now")
                self.init_ok = -1
                return
            self.N_PAYLOAD = self.N_DATA_SC + self.N_PILOT_SC
        else:
            print("incorrect pilot mode, plz read classes.md")
            self.init_ok = -1
            return
        self.N_SOLID_SC = self.N_DATA_SC + self.N_PILOT_SC
        self.pilot_seq = np.ones(self.N_PILOT_SC) * self.PILOT_GAIN
        self.N_DATA_QAM_SYMB = self.N_OFDM * self.N_DATA_SC
        self.PAYLOAD_LEN = self.N_OFDM * (self.CP_LEN + self.N)
        self.FRAME_LEN = (self.CP_LEN + self.N ) * (self.N_OFDM + self.N_TRAINING_SEQ) + self.LTF_WITH_CP_LEN

        # N_FRAMES
        if args.N_FRAMES is not None:
            self.N_FRAMES = args.N_FRAMES
        else:
            self.N_FRAMES = 1  # default value

        # PEAK
        if args.PEAK is not None:
            self.PEAK = args.PEAK
        else:
            self.PEAK = 0.9e19  # default value

        # SNR
        if args.SNR is not None:
            self.SNR = args.SNR
        else:
            self.SNR = 30  # default value

        # TX_INPUT_QAM_FILE
        if args.TX_INPUT_QAM_FILE is not None:
            self.TX_INPUT_QAM_FILE = args.TX_INPUT_QAM_FILE
        else:
            self.TX_INPUT_QAM_FILE = "./dats/simu/tx_input_qam.txt"  # default value

        # TX_OUTPUT_QAM_NO_PILOT_FILE
        if args.TX_OUTPUT_QAM_NO_PILOT_FILE is not None:
            self.TX_OUTPUT_QAM_NO_PILOT_FILE = args.TX_OUTPUT_QAM_NO_PILOT_FILE
        else:
            self.TX_OUTPUT_QAM_NO_PILOT_FILE = "./dats/rtl_input/tx_output_qam_no_pilot.txt"  # default value

        # TX_OUTPUT_QAM_PILOT_FILE
        if args.TX_OUTPUT_QAM_PILOT_FILE is not None:
            self.TX_OUTPUT_QAM_PILOT_FILE = args.TX_OUTPUT_QAM_PILOT_FILE
        else:
            self.TX_OUTPUT_QAM_PILOT_FILE = "./dats/rtl_input/tx_output_qam_pilot.txt"  # default value

        # TX_OUTPUT_SINGLE_FRAME_FILE
        if args.TX_OUTPUT_SINGLE_FRAME_FILE is not None:
            self.TX_OUTPUT_SINGLE_FRAME_FILE = args.TX_OUTPUT_SINGLE_FRAME_FILE
        else:
            self.TX_OUTPUT_QAM_PILOT_FILE = "./dats/rtl_input/single_frame.txt"

        # TX_OUTPUT_FILE
        if args.TX_OUTPUT_FILE is not None:
            self.TX_OUTPUT_FILE = args.TX_OUTPUT_FILE
        else:
            self.TX_OUTPUT_FILE = "./dats/rtl_input/rtl_test_input.txt"  # default value

        # CHANNEL_OUTPUT_FILE
        if args.CHANNEL_OUTPUT_FILE is not None:
            self.CHANNEL_OUTPUT_FILE = args.CHANNEL_OUTPUT_FILE
        else:
            self.CHANNEL_OUTPUT_FILE = "./dats/simu/channel_output.txt"  # default value

        # RTL_OUTPUT_I_FILE
        if args.RTL_OUTPUT_I_FILE is not None:
            self.RTL_OUTPUT_I_FILE = args.RTL_OUTPUT_I_FILE
        else:
            self.RTL_OUTPUT_I_FILE = "./dats/rtl_output/rtl_output_I.txt"  # default value

        # RTL_OUTPUT_Q_FILE
        if args.RTL_OUTPUT_Q_FILE is not None:
            self.RTL_OUTPUT_Q_FILE = args.RTL_OUTPUT_Q_FILE
        else:
            self.RTL_OUTPUT_Q_FILE = "./dats/rtl_output/rtl_output_Q.txt"  # default value
            
        if args.DATS_DIR is not None:
            self.DATS_DIR = args.DATS_DIR
        else:
            self.DATS_DIR = "./dats/rx_output/"

        # RTL_BITFILE
        if args.RTL_BITFILE is not None:
            self.RTL_BITFILE = args.RTL_BITFILE
        else:
            self.RTL_BITFILE = "/home/xilinx/work/test_AXI/bitfile/design_1_wrapper.bit"



        self.init_ok = 1
        return

    def display(self):
        print("EXE_MODE:", self.EXE_MODE)
        print("QAM_MODE:", self.QAM_MODE)
        print("N_FRAMES:", self.N_FRAMES)
        print("N:", self.N)
        print("CP_LEN:", self.CP_LEN)
        print("LLTF_GAIN:", self.LLTF_GAIN)
        print("LTF_SEQ_LEN:", self.LTF_SEQ_LEN)
        print("LTFcp:", self.LTFcp)
        print("LTF_matched_filter:", self.LTF_matched_filter)
        print("LLTF:", self.LLTF)
        print("LTF_DATA_LEN:", self.LTF_DATA_LEN)
        print("LTF_CP_LEN:", self.LTF_CP_LEN)
        print("LTF_WITH_CP_LEN:", self.LTF_WITH_CP_LEN)
        print("N_TRAINING_SEQ:", self.N_TRAINING_SEQ)
        print("TRAINING_SEQ_LEN:", self.TRAINING_SEQ_LEN)
        print("N_DATA_SC:", self.N_DATA_SC)
        print("N_PILOT_SC:", self.N_PILOT_SC)
        print("N_GUARD_SC:", self.N_GUARD_SC)
        print("N_SOLID_SC:", self.N_SOLID_SC)
        print("N_DATA_QAM_SYMB:", self.N_DATA_QAM_SYMB)
        print("PAYLOAD_LEN:", self.PAYLOAD_LEN)
        print("FRAME_LEN:", self.FRAME_LEN)
        print("OFDM_DATA_INDEX:", self.OFDM_DATA_INDEX)
        print("OFDM_PILOT_INDEX:", self.OFDM_PILOT_INDEX)
        print("OFDM_GUARD_INDEX:", self.OFDM_GUARD_INDEX)
        print("OFDM_SOLID_INDEX:", self.OFDM_SOLID_INDEX)
        print("N_TRAINING_SEQ:", self.N_TRAINING_SEQ)
        print("TRAINING_SEQ_LEN:", self.TRAINING_SEQ_LEN)
        print("QAM_GAIN:", self.QAM_GAIN)
        print("LLTF_GAIN:", self.LLTF_GAIN)
        print("TRAINING_SEQ_GAIN:", self.TRAINING_SEQ_GAIN)
        print("PILOT_GAIN:", self.PILOT_GAIN)
        print("QAM_MAP:", self.QAM_MAP)
        print("PEAK:", self.PEAK)
        print("SNR:", self.SNR)
        print("TX_INPUT_QAM_FILE:", self.TX_INPUT_QAM_FILE)
        print("TX_OUTPUT_QAM_NO_PILOT_FILE:", self.TX_OUTPUT_QAM_NO_PILOT_FILE)
        print("TX_OUTPUT_QAM_PILOT_FILE:", self.TX_OUTPUT_QAM_PILOT_FILE)
        print("TX_OUTPUT_FILE:", self.TX_OUTPUT_FILE)
        print("CHANNEL_OUTPUT_FILE:", self.CHANNEL_OUTPUT_FILE)
        print("RTL_OUTPUT_I_FILE:", self.RTL_OUTPUT_I_FILE)
        print("RTL_OUTPUT_Q_FILE:", self.RTL_OUTPUT_Q_FILE)
        return
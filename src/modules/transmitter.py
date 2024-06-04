import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from scipy import signal

from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import data_func as df
import plt_funcs as pf
import configs

def qam_mapping(qam_map,value):
    return qam_map[value]

def add_cp(ofdm_symb,N,cp_len):
    
    #Extract CP
    cp = ofdm_symb[N-cp_len:N:1]
    
    # Concatenate CP and symbol 
    ofdm_symb_cp = np.concatenate((cp,ofdm_symb))
    
    return ofdm_symb_cp

# qam_seq must be N aligned
def ofdm_mod(qam_seq,n_ofdm,N,CP_LEN):
    n_ofdm_symb_index = None
    
    ofdm_res = np.zeros((N+CP_LEN) * n_ofdm,dtype = np.complex64)
    ofdm_res_index = None
    for i in range (n_ofdm):
        n_ofdm_symb_index = np.arange(start = i*N, stop = (i+1)*N)
        ofdm_res_index = np.arange(start = i*(N+CP_LEN),stop = (i+1)*(N+CP_LEN))

        ifft_res = np.fft.ifft(qam_seq[n_ofdm_symb_index])
        ofdm_res[ofdm_res_index] = add_cp(ifft_res,N,CP_LEN)

    return ofdm_res

class tx:
    ofdm_cfg:configs.ofdm_config = None
    #gen_rand_qam_seq
    RAND_SEED = 123
    rand_qam_seq_len = None
    rand_qam_seq_file = None
    rand_qam_seq = None

    #TX_qam_seq
    guard_seq = None
    pilot_seq = None
    tx_qam_seq = None
    tx_qam_seq_len = None
    tx_qam_seq_file = None
    
    #ofdm_seq
    ofdm_symb_seq = None

    #tx_frame_seq
    tx_frame_seq = None
    tx_frame_seq_len = None
    tx_single_frame_seq = None
    tx_single_frame_seq_file = None
    tx_frame_seq_file = None

    def __init__(self,args_cfg):
        self.ofdm_cfg = args_cfg
        if(self.ofdm_cfg.PILOT_MODE == "NO_PILOT"):
            self.guard_seq = None
            self.pilot_seq = None
        elif(self.ofdm_cfg.PILOT_MODE == "ZEROS"):
            self.guard_seq = np.zeros(self.ofdm_cfg.N_GUARD_SC,dtype = np.complex64)
            self.pilot_seq = None
        elif(self.ofdm_cfg.PILOT_MODE == "PILODE_MODE_!"):
            print("IN TX, PILOT_MODE_1 NOT USED YET!!!")
            self.guard_seq = None
            self.pilot_seq = None

    

    def gen_rand_qam_seq(self):
        self.rand_qam_seq_len = self.ofdm_cfg.N_DATA_QAM_SYMB
        random.seed(self.RAND_SEED)
        self.rand_qam_seq = np.zeros(self.rand_qam_seq_len,dtype = np.complex64)

        # gen rand qam seq
        for i in range(self.rand_qam_seq_len):
            rand_data = 0
            if(self.ofdm_cfg.QAM_MODE == 4):
                rand_data = random.randint(0,3)
            elif(self.ofdm_cfg.QAM_MODE == 16):
                rand_data = random.randint(0,15)
            else:
                print("illegal qam mode")
            self.rand_qam_seq[i] = qam_mapping(self.ofdm_cfg.QAM_MAP,rand_data)

        # len: N_OFDM * N_DATA_SC
        self.rand_qam_seq_file = self.ofdm_cfg.TX_OUTPUT_QAM_NO_PILOT_FILE
        df.save_array_to_file(self.rand_qam_seq_file,self.rand_qam_seq)
        return
    
    def gen_tx_qam_seq(self):
        self.tx_qam_seq_len = self.ofdm_cfg.N * self.ofdm_cfg.N_OFDM
        self.tx_qam_seq = np.zeros(self.tx_qam_seq_len,dtype = np.complex64)
            
        for i in range(self.ofdm_cfg.N_OFDM):
            tx_qam_seq_index = np.arange(start = i*self.ofdm_cfg.N,stop = (i+1)*self.ofdm_cfg.N)
            data_qam_seq_index = np.arange(start = i*self.ofdm_cfg.N_DATA_SC, stop = (i+1) * self.ofdm_cfg.N_DATA_SC)

            qam_per_ofdm_symb = np.zeros(self.ofdm_cfg.N,dtype = np.complex64)

            #insert data
            if(self.ofdm_cfg.OFDM_DATA_INDEX is not None):
                qam_per_ofdm_symb[self.ofdm_cfg.OFDM_DATA_INDEX] = self.rand_qam_seq[data_qam_seq_index]
            #insert guard
            if(self.ofdm_cfg.OFDM_GUARD_INDEX is not None):
                qam_per_ofdm_symb[self.ofdm_cfg.OFDM_GUARD_INDEX] = self.guard_seq
            #inesrt pilot
            if(self.ofdm_cfg.OFDM_PILOT_INDEX is not None):
                qam_per_ofdm_symb[self.ofdm_cfg.OFDM_PILOT_INDEX] = self.pilot_seq

            self.tx_qam_seq[tx_qam_seq_index] = qam_per_ofdm_symb

        #print("tx_qam_seq = ",self.tx_qam_seq)

        self.tx_qam_seq_file = self.ofdm_cfg.TX_OUTPUT_QAM_PILOT_FILE
        df.save_array_to_file(self.tx_qam_seq_file,self.tx_qam_seq)
        
    def ofdm_mod(self):
        self.ofdm_symb_seq = ofdm_mod(self.tx_qam_seq,self.ofdm_cfg.N_OFDM,self.ofdm_cfg.N,self.ofdm_cfg.CP_LEN)

    def frame_assemble(self):
        self.tx_frame_seq_len = self.ofdm_cfg.FRAME_LEN * self.ofdm_cfg.N_FRAMES

        #insert LLTF
        self.single_frame_seq = self.ofdm_cfg.LLTF

        #insert training_seq
        #print("train_symbx2 = ", self.ofdm_cfg.train_symbx2)
        self.single_frame_seq = np.concatenate((self.single_frame_seq, self.ofdm_cfg.train_symbx2))

        #insert ofdm symbs
        self.single_frame_seq = np.concatenate((self.single_frame_seq,self.ofdm_symb_seq))

        #print("single_frame_len = ",len(self.single_frame_seq))
        #print("single_frame = ",self.single_frame_seq)


        self.tx_single_frame_seq_file = self.ofdm_cfg.TX_OUTPUT_SINGLE_FRAME_FILE
        df.save_array_to_file(self.tx_single_frame_seq_file,self.single_frame_seq)
        self.tx_frame_seq = np.tile(self.single_frame_seq, self.ofdm_cfg.N_FRAMES)
        self.tx_frame_seq_file = self.ofdm_cfg.TX_OUTPUT_FILE
        df.save_array_to_file("rtl_test_input_complex64.txt", self.tx_frame_seq)
        df.complex_file_to_IQ_file("rtl_test_input_complex64.txt","rtl_test_input_I.txt","rtl_test_input_Q.txt",self.tx_frame_seq_file)

    def plt_single_frame(self):
        pf.plot_all("single frame in TX",self.single_frame_seq,self.ofdm_cfg.SAMPLE_RATE)

    def plt_frames(self):
        pf.plot_all("frames in TX",self.tx_frame_seq,self.ofdm_cfg.SAMPLE_RATE)

    def tx_run(self):
        if(self.ofdm_cfg.EXE_MODE == "DEMODE_RTL_OUTPUT"):
            return
        else:
            self.gen_rand_qam_seq()
            self.gen_tx_qam_seq()
            self.ofdm_mod()
            self.frame_assemble()
            self.plt_single_frame()
            self.plt_frames()
            plt.show()
            return
'''
def interpolate_complex_signal_with_lowpass(input_signal, upsample_factor, fs):
    """
    使用低通滤波器对复数信号进行插值的函数

    参数:
    input_signal (array_like): 输入复数信号数组
    upsample_factor (int): 插值因子
    fs (float): 信号的采样率

    返回:
    interpolated_signal (ndarray): 插值后的复数信号数组
    t_upsampled (ndarray): 插值后的时间序列
    """

    # 生成插值后的时间序列
    t_upsampled = np.linspace(0, len(input_signal) / fs, len(input_signal) * upsample_factor)

    # 分别对实部和虚部进行插值
    interpolated_real = np.interp(t_upsampled, np.arange(len(input_signal)) / fs, np.real(input_signal))
    interpolated_imag = np.interp(t_upsampled, np.arange(len(input_signal)) / fs, np.imag(input_signal))

    # 设计低通滤波器
    cutoff_freq = 0.5 * fs / upsample_factor
    b, a = signal.butter(8, cutoff_freq / (fs / 2), 'low')

    # 对实部和虚部分别进行低通滤波
    interpolated_real = signal.filtfilt(b, a, interpolated_real)
    interpolated_imag = signal.filtfilt(b, a, interpolated_imag)

    # 合成复数信号
    interpolated_signal = interpolated_real + 1j * interpolated_imag

    return interpolated_signal, t_upsampled

def digital_heterodyning_complex(signal, lo_frequency, fs):
    """
    复数信号的数字混频函数

    参数:
    signal (array_like): 输入信号数组，复数形式
    lo_frequency (float): 本振频率
    fs (float): 采样频率

    返回:
    mixed_signal (ndarray): 混频后的信号数组，复数形式
    """

    t = np.arange(len(signal)) / fs  # 时间序列
    lo_signal = np.exp(1j * 2 * np.pi * lo_frequency * t)  # 本振信号
    #lo_signal = np.cos(2 * np.pi * lo_frequency * t)
    mixed_signal = signal * lo_signal  # 混频信号

    return mixed_signal

def upsample(sig_in, upsample_rate):
    upsample_res = signal.resample(sig_in,upsample_rate*(len(sig_in)))
    return upsample_res

def spectrum_shift_right(sig_in, f_shift):
    """
    将输入信号在频谱上进行搬移

    参数:
    sig_in (ndarray): 输入信号（复数数组）
    shift (int): 频谱搬移量

    返回:
    ndarray: 搬移后的信号
    """
    shift = int(f_shift * len(sig_in) / consts.FS)

    # 对输入信号进行傅里叶变换
    sig_fft = np.fft.fft(sig_in)
    
    # 频谱搬移（向右移动）
    sig_shifted_fft = np.roll(sig_fft, shift)
    
    # 进行傅里叶逆变换
    sig_out = np.fft.ifft(sig_shifted_fft)
    
    return sig_out

def bpsk_mapping(value):
    return consts.bpsk_mapping_1div[value]

def qam16_mapping(value):
    # 16QAM调制符号映射表
    #print("value = ",value)
    return consts.QAM16_mapping_1div[value]

def qam4_mapping(value):
    return consts.QAM4_mapping_1div[value]

def QAM16_mod_single_ascii(data):
    # 将ASCII字符转换为整数
    ascii_value = ord(data)

    # 获取低8位中的高4位和低4位
    upper = (ascii_value >> 4) & 0x0F
    lower = ascii_value & 0x0F

    # 分别对upper和lower进行16QAM符号映射
    symb1 = qam16_mapping(upper)
    symb2 = qam16_mapping(lower)

    return symb1, symb2

def QAM16_mod_str(str_in):
    strlen = len(str_in)
    mod_res_len = len(str_in) *2
    mod_res = np.zeros(mod_res_len, dtype = np.complex64)

    for i in range(strlen):
       print("str_in[i]=",str_in[i])
       sym1,sym2=QAM16_mod_single_ascii(str_in[i])
       mod_res[i*2] = sym1
       mod_res[i*2+1] = sym2

    return mod_res

def QAM16_mod_ascii_file(file_in):
    str = df.read_ascii_file(file_in)
    return QAM16_mod_str(str)

def align_qam_symb(qam_sig):
    qam_sym_len = len(qam_sig)
    n_frames = math.ceil(qam_sym_len / consts.N_DATA / consts.N_OFDM)
    aligned_diff = (n_frames * consts.N_DATA * consts.N_OFDM) - qam_sym_len
    aligned_repeat = qam_sig[0:aligned_diff]
    qam_mod_aligned = np.zeros(n_frames * consts.N_DATA * consts.N_OFDM, dtype = np.complex64)
    qam_mod_aligned[0:qam_sym_len] = qam_sig
    qam_mod_aligned[qam_sym_len:qam_sym_len+aligned_diff] = aligned_repeat
    return n_frames, qam_mod_aligned



def ofdm_mod(qam_sym):

    # check qam sym 
    #print("\n\n\n",qam_sym)

    sc_array = np.zeros(consts.N,np.complex64)
    sc_array[consts.OFDM_DATA_INDEX] = qam_sym
    sc_array[consts.OFDM_PILOT_INDEX] = consts.pilot_seq 

    #scatterplot(sc_array.real,sc_array.imag,ax=None)
    #plt.show()

    ofdm_mod = np.fft.ifft(sc_array,consts.N)

    ofdm_with_cp = add_cp(ofdm_mod,consts.N,consts.CP_LEN)

    return ofdm_with_cp


def Transmitter(ascii_file, output_file):
    #plot LLTF
    #pf.plot_all("LLTF_symb",consts.LLTF_symb,consts.SR)

    # 16QAM mod ascii file
    #qam_mod_res = QAM16_mod_ascii_file(ascii_file)

    #16QAM random array
    #qam_mod_res = np.zeros(consts.NSYM * consts.N_FRAME,dtype = np.complex64)
    #rand_qam_index = np.random.randint(0, 4, size=consts.NSYM* consts.N_FRAME)
    #for i in range(int((consts.NSYM * consts.N_FRAME)/2)):
    #    qam_mod_res[i] = qam4_mapping((rand_qam_index[i] ^ np.random.randint(0,4)))

    #for i in range(int((consts.NSYM * consts.N_FRAME)/2)):
    #    qam_mod_res[i+int((consts.NSYM * consts.N_FRAME)/2)] = -1*qam_mod_res[i]
        
    #df.save_array_to_file("D:/work/CA/simulation/dats/qam_mod_res.txt",qam_mod_res)
    qam_mod_res = np.zeros((consts.NSYM * consts.N_FRAME))
    qam_mod_res = np.loadtxt("D:/work/CA/simulation/dats/qam_mod_res.txt",dtype = np.complex64)
    # aligned qam signal to complete frame
    n_frames, qam_mod_aligned = align_qam_symb(qam_mod_res)

    

    align_qam_symb_sum = np.sum(qam_mod_aligned.real)
    print("align_qam_symb_sum = ",align_qam_symb_sum)
    #time.sleep(3)
    

    scatterplot(qam_mod_aligned.real,qam_mod_aligned.imag,ax=None)

    trans_res_len = consts.FRAME_LEN *n_frames
    trans_res = np.zeros(trans_res_len,dtype=np.complex64)
    
    print("n_frames = ",n_frames)
    # ofdm mod:
    for i in range(n_frames):
        # LLTF
        trans_res[i * consts.FRAME_LEN:i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN] = consts.LLTF

        # training seq
        training_seq_start_index = i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN
        trans_res[training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN] = consts.train_symbx2

        # payload
        ofdm_payload = np.zeros((consts.CP_LEN + consts.N)*consts.N_OFDM,dtype=np.complex64)
        
        for j in range(consts.N_OFDM):
            index = i*consts.N_OFDM*consts.N_DATA + j*consts.N_DATA
            ofdm_payload[j*(consts.CP_LEN+consts.N):(j+1)*(consts.CP_LEN + consts.N)] = ofdm_mod(qam_mod_aligned[index:index+consts.N_DATA])
        payload_start_index = i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN + consts.TRAINING_SYMB_LEN
        trans_res[payload_start_index:(i+1) * consts.FRAME_LEN] = ofdm_payload

        #store ofdm payload
        ofdm_payload_with_cp = np.zeros(consts.N * consts.N_OFDM, dtype = np.complex64)
        ofdm_payload_with_cp = ofdm_payload
        df.save_array_to_file(consts.tx_ofdm_payload_with_cp_file,ofdm_payload_with_cp)
    

    #up sample
        
    
    #trans_res = signal.resample_poly(trans_res,up=consts.SAMPLE_FACTOR,down=1)
    #trans_res = digital_heterodyning_complex(trans_res,consts.FREQ_SHIFT,512e6)
    #scatterplot(trans_res.real,trans_res.imag,ax=None)
    print("trans_res_len = ",len(trans_res))

    #trans_res = signal.resample_poly(trans_res,up=8,down=1)
    #trans_res = digital_heterodyning_complex(trans_res,1024e6,consts.FS)

    df.save_array_to_file(output_file,trans_res)
    #trans_res = upsample(trans_res, 8)
    #trans_res = upsample(trans_res, 2)


    #plot Transmitter res
    pf.plot_all("OFDM MOD RES imag",trans_res.imag, 512e6)
    pf.plot_all("OFDM MOD RES real",trans_res.real, 512e6)
    return

'''
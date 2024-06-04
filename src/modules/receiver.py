import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import signal
from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import data_func as df
import plt_funcs as pf
import configs

#def dac_fir2():
#def dac_fir1():
#def dac_fir0():
#def adc_fir2():
#def adc_fir1():
#def adc_fir0():

def ofdm_demod(ofdm_rx,N,cp_len):
    # Remove CP 
    ofdm_u = ofdm_rx[cp_len:(N+cp_len)]
    #print(ofdm_u)
    # Perform FFT 
    data = np.fft.fft(ofdm_u,N)
    
    return data



def zf_est(training_symb1,training_symb2, local_symb_1, local_symb_2):
    

    h_1 = training_symb1 / local_symb_1
    h_2 = training_symb2 / local_symb_2
    #h_1 = LLTF_data_1 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])
    #h_2 = LLTF_data_2 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])

    h_final = (h_1 + h_2) / 2
    #print(h_final)

    return h_final

def zf_equ(data_symb,channel_est_res):
    channel_equ_res = data_symb * (np.conj(channel_est_res)/(abs(channel_est_res)**2))
    return channel_equ_res

class rx:
    ofdm_cfg:configs.ofdm_config = None

    received_sig_I = None
    received_sig_Q = None
    received_sig = None

    #first time sync
    #PEAK = None
    time_sync_res_before_cutting = None
    recv_sig_start_index = None
    recv_sig_end_index = None


    #second time sync
    peak_height = None
    cutting_start_index = None
    cutting_end_index = None
    recv_sig_after_cutting = None
    tims_sync_res_after_cutting = None
    pks = None
    num_frames = None

    #disassmble:
    LTF_seqs = None
    training_seqs = None
    payload_seqs = None # size: num_frames * (ofdm_symbs * (N+CP_LEN)), with cp


    #channel est:
    training_ofdm_symb_1 = None   # with CP
    training_ofdm_symb_2 = None   # with CP
    channel_est_res = None        # 2d: num_frames * N
    channel_est_phase = None      # 2d: num_frames * N
    channel_est_gain = None       # 2d: num_frames * N
    



    #demode:
    qam_symbs_after_ofdm_demode = None # size: num_frames * ofdm_symbs * N
    qam_symbs_after_ofdm_demode_1d = None
#----------- up above -----------: using N
#----------- down below ---------: using N_DATA_SC

    #channel equ:
    qam_symbs_after_channel_equ = None # 3d, size: num_frames * ofdm_symbs * N_DATA_SC

    #1d demode res
    demode_res_after_channel_equ_1d = None

    #ranging:
    ranging_res = None     # 3d, size: num_frames * ofdm_symbs * N
    ranging_res_1d = None
    local_qam_symbs = None # 1d, size: N

    def __init__(self,args_cfg):
        self.ofdm_cfg = args_cfg
        if(self.ofdm_cfg.EXE_MODE == "GEN_RTL_INPUT"):
            return
        elif(self.ofdm_cfg.EXE_MODE == "SIMU"):
            self.received_sig = np.loadtxt(self.ofdm_cfg.CHANNEL_OUTPUT_FILE,dtype = np.complex64)
        elif((self.ofdm_cfg.EXE_MODE == "DEMODE_RTL_OUTPUT") or (self.ofdm_cfg.EXE_MODE == "RTL_MODE") ):
            self.received_sig_I = np.loadtxt(self.ofdm_cfg.RTL_OUTPUT_I_FILE)
            self.received_sig_Q = np.loadtxt(self.ofdm_cfg.RTL_OUTPUT_Q_FILE)
            self.received_sig = np.zeros(len(self.received_sig_I),dtype = np.complex64)
            self.received_sig = self.received_sig_I + 1j * self.received_sig_Q
            
            self.local_qam_symbs = np.loadtxt(self.ofdm_cfg.TX_OUTPUT_QAM_PILOT_FILE, dtype = np.complex64)

    def reload_received_sig(self):
        self.received_sig_I = np.loadtxt(self.ofdm_cfg.RTL_OUTPUT_I_FILE)
        self.received_sig_Q = np.loadtxt(self.ofdm_cfg.RTL_OUTPUT_Q_FILE)
        self.received_sig = np.zeros(len(self.received_sig_I),dtype = np.complex64)
        self.received_sig = self.received_sig_I + 1j * self.received_sig_Q
        
    def first_time_sync(self):
        self.time_sync_res_before_cutting = signal.convolve(in1=self.received_sig, in2 = np.conj(self.ofdm_cfg.LTF_matched_filter), mode = "same")
        self.time_sync_res_before_cutting = np.abs(self.time_sync_res_before_cutting) ** 2

    
        #pf.plot_dat("time sync res before cutting",self.time_sync_res_before_cutting)
        #plt.show()

        pks_before_cutting = signal.find_peaks(self.time_sync_res_before_cutting,height = self.ofdm_cfg.PEAK, distance=64)
        print(pks_before_cutting)
        
    
    def cutting_recv_sig(self,start_index, end_index, height):
        self.cutting_start_index = start_index
        self.cutting_end_index = end_index
        self.peak_height = height
        if(start_index == -1):
            cutting_index = np.arange(start = 0, stop = len(self.time_sync_res_before_cutting)) 
        else:
            cutting_index = np.arange(start = self.cutting_start_index, stop = self.cutting_end_index)
            
        self.tims_sync_res_after_cutting = self.time_sync_res_before_cutting[cutting_index]
        self.recv_sig_after_cutting = self.received_sig[cutting_index]

    def plt_recv_sig(self):
        pf.plot_all("recv sig after cutting",self.recv_sig_after_cutting,1024e6)
        plt.show()

    def second_time_sync(self):
        self.pks = signal.find_peaks(self.tims_sync_res_after_cutting,height = self.peak_height, distance=64)
        print(self.pks)
        self.num_frames = int(len(self.pks[0]) // 2)
        print("num_frames = ",self.num_frames)
        #pf.plot_dat("time sync res after cutting",self.tims_sync_res_after_cutting)
        #plt.show()
    
    def disassemble(self):
        self.LTF_seqs = np.zeros((self.num_frames,self.ofdm_cfg.LTF_WITH_CP_LEN), dtype = np.complex64)
        self.training_seqs = np.zeros((self.num_frames,self.ofdm_cfg.TRAINING_SEQ_LEN), dtype = np.complex64)
        self.payload_seqs = np.zeros((self.num_frames,self.ofdm_cfg.PAYLOAD_LEN), dtype = np.complex64)

        for i in range(self.num_frames):
            LTF_seq_start_index = self.pks[0][i*2+1] - 128
            #("LTF_seq_start_index = ",LTF_seq_start_index)
            if(LTF_seq_start_index < 0):
                print("first frame not complete")
                break
            if(LTF_seq_start_index + self.ofdm_cfg.FRAME_LEN  > len(self.recv_sig_after_cutting)):
                print("last frame not complete")
                self.num_frames = self.num_frames-1
                break
            #print("LTF_seq_start_index = ",LTF_seq_start_index)

            training_seq_start_index = self.pks[0][i*2+1] + 32
            payload_start_index = training_seq_start_index + self.ofdm_cfg.TRAINING_SEQ_LEN

            self.LTF_seqs[i] = self.recv_sig_after_cutting[LTF_seq_start_index : LTF_seq_start_index + self.ofdm_cfg.LTF_WITH_CP_LEN]
            self.training_seqs[i] = self.recv_sig_after_cutting[training_seq_start_index : training_seq_start_index + self.ofdm_cfg.TRAINING_SEQ_LEN]
            self.payload_seqs[i] = self.recv_sig_after_cutting[payload_start_index : payload_start_index + self.ofdm_cfg.PAYLOAD_LEN]

            print("\n\n\n in",i,"frames:\n","LTF start index = ",LTF_seq_start_index,"\ntraining_seq_start_index = ",training_seq_start_index,
                  "\npayload_start_index =",payload_start_index,"\n\n\n")


    def channel_est(self, is_print_est_res = 0):
        
        self.channel_est_res = np.zeros((self.num_frames,self.ofdm_cfg.N),dtype = np.complex64)
        self.channel_est_phase = np.zeros((self.num_frames, self.ofdm_cfg.N))
        self.channel_est_gain = np.zeros((self.num_frames, self.ofdm_cfg.N))
        self.training_ofdm_symb_1 = np.zeros((self.num_frames,self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN),dtype = np.complex64)
        self.training_ofdm_symb_2 = np.zeros((self.num_frames,self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN),dtype = np.complex64)
        for i in range(self.num_frames):
            self.training_ofdm_symb_1 = self.training_seqs[i][0:self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN]
            self.training_ofdm_symb_2 = self.training_seqs[i][self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN : (self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN)*2]

            training_symb_demode_res_1 = ofdm_demod(self.training_ofdm_symb_1, self.ofdm_cfg.N,self.ofdm_cfg.CP_LEN)
            training_symb_demode_res_2 = ofdm_demod(self.training_ofdm_symb_2, self.ofdm_cfg.N,self.ofdm_cfg.CP_LEN)

            self.channel_est_res[i] = zf_est(training_symb_demode_res_1,training_symb_demode_res_2,
                                             self.ofdm_cfg.train_seq,self.ofdm_cfg.train_seq)
            self.channel_est_phase[i] = np.angle(self.channel_est_res[i])
            self.channel_est_gain[i] = np.abs(self.channel_est_res[i])
            if(is_print_est_res):
                print("i=",i,"est_res:",self.channel_est_res[i])
            
    def demode_payload(self):
        self.qam_symbs_after_ofdm_demode = np.zeros((self.num_frames,self.ofdm_cfg.N_OFDM, self.ofdm_cfg.N),dtype = np.complex64)
        
        for i in range(self.num_frames):
            for j in range(self.ofdm_cfg.N_OFDM):
                payload_index = np.arange(start = j*(self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN), stop=(j+1)*(self.ofdm_cfg.N + self.ofdm_cfg.CP_LEN))
                self.qam_symbs_after_ofdm_demode[i][j] = ofdm_demod(self.payload_seqs[i][payload_index],self.ofdm_cfg.N,self.ofdm_cfg.CP_LEN)
                
        self.qam_symbs_after_ofdm_demode_1d = self.qam_symbs_after_ofdm_demode.reshape(-1)


    def channel_equ(self,equ_on):
        if(equ_on):
            self.qam_symbs_after_channel_equ = np.zeros((self.num_frames,self.ofdm_cfg.N_OFDM,self.ofdm_cfg.N_DATA_SC),dtype = np.complex64)
            for i in range(self.num_frames):
                for j in range(self.ofdm_cfg.N_OFDM):
                    #print("channel_est_res = ",self.channel_est_res[i])
                    #print("qam_symbs_after_ofdm_demode = ",self.qam_symbs_after_ofdm_demode[i][j])
                    self.qam_symbs_after_channel_equ[i][j] = zf_equ(self.qam_symbs_after_ofdm_demode[i][j][self.ofdm_cfg.OFDM_DATA_INDEX],self.channel_est_res[i][self.ofdm_cfg.OFDM_DATA_INDEX])
            self.demode_res_after_channel_equ_1d = self.qam_symbs_after_channel_equ.reshape(-1)
        else:
            self.demode_res_after_channel_equ_1d = self.qam_symbs_after_ofdm_demode.reshape(-1)
            
    def ranging(self):
        self.ranging_res = np.zeros((self.num_frames,self.ofdm_cfg.N_OFDM, self.ofdm_cfg.N),dtype = np.complex64)
        csi = np.zeros(self.ofdm_cfg.N,dtype = np.complex64)
        for i in range(self.num_frames):
            for j in range(self.ofdm_cfg.N_OFDM):
                csi = self.qam_symbs_after_ofdm_demode[i][j] / self.local_qam_symbs
                self.ranging_res[i][j] = np.fft.ifft(csi, self.ofdm_cfg.N)            
        
    def save_rx_to_file(self,file_dir):
        dir = ""
        if(file_dir is None):
            dir = self.ofdm_cfg.DATS_DIR
        else:
            dir = file_dir
        
        df.save_array_to_file(dir + "recv_sig.txt", self.recv_sig_after_cutting)
        df.save_array_to_file(dir + "ts_res.txt", self.time_sync_res_before_cutting)
        df.save_array_to_file(dir + "channel_phase.txt",self.channel_est_phase[0])
        df.save_array_to_file(dir + "channel_gain.txt",self.channel_est_gain[0])
        df.save_array_to_file(dir + "const_no_equ.txt",self.qam_symbs_after_ofdm_demode_1d)
        df.save_array_to_file(dir + "const_after_equ.txt",self.demode_res_after_channel_equ_1d)

    def plt_all_qam_constellation(self):
        scatterplot(self.demode_res_after_channel_equ_1d.real,self.demode_res_after_channel_equ_1d.imag,ax=None)

    def rx_run(self):
        self.first_time_sync()
        self.cutting_recv_sig(-1,0,self.ofdm_cfg.PEAK)
        self.second_time_sync()
        self.disassemble()
        self.channel_est()
        self.demode_payload()
        self.channel_equ(1)
        self.plt_all_qam_constellation()
        plt.show()
'''
def ranging(local_ofdm_payload, rx_ofdm_payload, no_frame,i):
    c=3 * 10 ** 8
    B=1.024 * 10 ** 9
    deltaR = c/(2*B)
    range = np.linspace(0,consts.N,consts.N)*deltaR
    rx_ofdm_payload_after_fft = np.fft.fft(rx_ofdm_payload,consts.N)
    local_ofdm_payload_after_fft = np.fft.fft(local_ofdm_payload,consts.N)

    csi = rx_ofdm_payload_after_fft/local_ofdm_payload_after_fft

    ranging_res = np.fft.ifft(csi,consts.N)
    ranging_res_abs = np.abs(ranging_res)
    ranging_res_dB = 20*np.log10(ranging_res_abs/max(ranging_res_abs))
    # plt.figure("range info")
    # plt.plot(range,ranging_res_dB)
    # plt.title([no_frame,'_',i])
    # plt.ylabel('Ambiguity (dB)')
    # plt.xlabel('Distance (m)')
    # plt.show()

    return ranging_res_abs


def estimate_phi(training_symb):

    

    train_symb_mod = consts.train_seq_ifft[0].real * consts.train_seq_ifft[0].real - consts.train_seq_ifft[0].imag * consts.train_seq_ifft[0].imag

    cos_phi = (training_symb.real * consts.train_seq_ifft[0].real + training_symb.imag * consts.train_seq_ifft[0].imag) / train_symb_mod
    sin_phi = -1 * (training_symb.imag * consts.train_seq_ifft[0].real + training_symb.real * consts.train_seq_ifft[0].imag) / train_symb_mod
    #phi_gain = np.sqrt(1/(cos_phi * cos_phi + sin_phi * sin_phi))

    cos_phi = cos_phi #* phi_gain
    cos_phi_aver = np.sum(cos_phi)/consts.N
    sin_phi = sin_phi #* phi_gain
    sin_phi_aver = np.sum(sin_phi)/consts.N
    print("cos_phi = ,",cos_phi,"\ncos_phi_aver = ",cos_phi_aver)
    print("sin_phi = ,",sin_phi,"\nsin_phi_aver = ",sin_phi_aver)

    #pf.plot_all("training_symb after samples",training_symb.real,512e6)
    #pf.plot_all("training_symb before sampels",consts.LTFsymb.real,512e6)
    #plt.show()

    return cos_phi,cos_phi_aver,sin_phi,sin_phi_aver

def phase_correct(input_sig, cos_phi, sin_phi):

    print("input_sig = ",input_sig)

    correct_phi_res_real  = cos_phi * (input_sig.real) + sin_phi * (input_sig.imag)
    correct_phi_res_imag  = cos_phi * (input_sig.imag) + sin_phi * (input_sig.real)

    return ((correct_phi_res_real + correct_phi_res_imag * 1j) / (cos_phi *cos_phi - sin_phi * sin_phi))

def estimate_frequency_offset(training_symb1,training_symb2):
    symb2= training_symb2
    symb1= training_symb1

    L = consts.N

    r = symb1 * np.conj(symb2)
    r_sum = np.sum(r)
    r_angle = np.angle(r_sum) / (np.pi * 2)

    freq_off = (r_angle* 512e6) /(L) 

    return r_angle,freq_off


def decimate_with_lowpass(input_signal, decimation_factor, fs):
    """
    使用低通滤波器进行降采样的函数

    参数:
    input_signal (array_like): 输入复数信号数组
    decimation_factor (int): 降采样因子
    fs (float): 信号的采样率

    返回:
    decimated_signal (ndarray): 降采样后的复数信号数组
    t_decimated (ndarray): 降采样后的时间序列
    """

    # 设计低通滤波器
    cutoff_freq = 0.5 * fs / decimation_factor
    b, a = signal.butter(8, cutoff_freq / (fs / 2), 'low')

    # 对输入信号进行低通滤波
    filtered_signal_real = signal.filtfilt(b, a, np.real(input_signal))
    filtered_signal_imag = signal.filtfilt(b, a, np.imag(input_signal))

    # 进行降采样
    decimated_signal_real = signal.decimate(filtered_signal_real, decimation_factor, ftype = "fir")
    decimated_signal_imag = signal.decimate(filtered_signal_imag, decimation_factor, ftype = "fir")

    # 生成降采样后的时间序列
    t_decimated = np.linspace(0, len(input_signal) / fs, len(decimated_signal_real))

    # 合成复数信号
    decimated_signal = decimated_signal_real + 1j * decimated_signal_imag

    return decimated_signal, t_decimated

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
    lo_signal = np.exp(-1j * 2 * np.pi * lo_frequency * t)  # 本振信号
    mixed_signal = signal * lo_signal  # 混频信号

    return mixed_signal

def downsample(sig_in, downsample_rate):
    downsample_res = signal.resample(sig_in,int(len(sig_in) / downsample_rate))
    return downsample_res

def spectrum_shift_left(sig_in, f_shift):
    """
    将输入信号在频谱上向左进行搬移

    参数:
    sig_in (ndarray): 输入信号（复数数组）
    f_shift (float): 频谱搬移量，单位为Hz
    fs (float): 原始采样率

    返回:
    ndarray: 搬移后的信号
    """
    shift = int(-f_shift * len(sig_in) / consts.FS)

    # 对输入信号进行傅里叶变换
    sig_fft = np.fft.fft(sig_in)
    
    # 频谱搬移（向左移动）
    sig_shifted_fft = np.roll(sig_fft, shift)
    
    # 进行傅里叶逆变换
    sig_out = np.fft.ifft(sig_shifted_fft)
    
    return sig_out

def channel_est_pilot(ofdm_symb):
    channel_est_res = np.zeros(consts.OFDM_DATA_BLOCK_NUM,dtype = np.complex64)
    recv_pilot = ofdm_symb[consts.OFDM_PILOT_INDEX]
    h = recv_pilot / (consts.pilot_seq)

    for i in range(5):
        channel_est_res[i] = (h[i] + h[i+1])/2
    
    

    for i in range(5,9):
        channel_est_res[i-1] = (h[i] + h[i+1])/2

    #print("channel_est_res = ",channel_est_res)

    return channel_est_res

def channel_est(LLTF_symb_1, LLTF_symb_2):

    LLTF_data_1 = LLTF_symb_1[consts.OFDM_PAYLOAD_INDEX]
    LLTF_data_2 = LLTF_symb_2[consts.OFDM_PAYLOAD_INDEX] 

    h_1 = LLTF_data_1 / (consts.train_seq[consts.OFDM_PAYLOAD_INDEX] * consts.TRAINING_GAIN)
    h_2 = LLTF_data_2 / (consts.train_seq[consts.OFDM_PAYLOAD_INDEX] * consts.TRAINING_GAIN)
    #h_1 = LLTF_data_1 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])
    #h_2 = LLTF_data_2 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])

    h_final = (h_1 + h_2) / 2
    #print(h_final)

    return h_final

# channel_equ: input: ofdm_symb[index] !!! the input has removed pilot 
def channel_equ(ofdm_symb, channel_est_res):
    channel_equ_res = ofdm_symb * (np.conj(channel_est_res)/(abs(channel_est_res)**2))
    return channel_equ_res

def channel_equ_pilot(ofdm_symb,channel_est_res):
    tmp = np.zeros(consts.N,dtype = np.complex64)
    for i in range(consts.OFDM_DATA_BLOCK_NUM):
        index_block = consts.OFDM_DATA_INDEX[i*4:i*4+4]
        tmp[index_block] = ofdm_symb[index_block] * (np.conj(channel_est_res[i])/(abs(channel_est_res[i])**2))
    #tmp[consts.OFDM_PILOT_INDEX] = ofdm_symb[consts.OFDM_PILOT_INDEX]
    return tmp[consts.OFDM_DATA_INDEX]

#ofdm_demod
def ofdm_demod(ofdm_rx,N,cp_len):
    # Remove CP 
    ofdm_u = ofdm_rx[cp_len:(N+cp_len)]
    #print(ofdm_u)
    # Perform FFT 
    data = np.fft.fft(ofdm_u,N)
    
    return data

def qam16_demodulation(symbol):

    # 计算输入符号与所有16QAM调制符号之间的距离
    distances = [np.abs(symbol - value) for value in consts.QAM16_mapping_2div.values()]

    # 找到距离最小的调制符号
    min_distance_index = np.argmin(distances)

    # 根据映射表找到对应的二进制数据
    demodulated_bits = list(consts.QAM16_mapping_2div.keys())[min_distance_index]

    # 将二进制数据转换为一个8位的无符号整数（np.uint8）
    result = np.uint8(demodulated_bits[0] << 2 | demodulated_bits[1])

    return result

def qam16_symb_to_uint8(qam_data):
    qam_demod_res = np.zeros(len(qam_data)//2,dtype=np.uint8)
    for i in range(len(qam_data)//2):
        symb1 = qam_data[i*2]
        symb2 = qam_data[i*2+1]

        upper = qam16_demodulation(symb1)
        lower = qam16_demodulation(symb2)

        tmp = (upper << 4) | lower
        qam_demod_res[i] = tmp
    return qam_demod_res



def Receiver(channel_output_file, rx_output_file):
    
    rx_din_sig = np.loadtxt(channel_output_file,dtype = np.complex64) 
    #rx_din_sig.imag = np.zeros(320000)

    pf.plot_all("sig before downsample",rx_din_sig.real,512e6)

    #rx_din_sig = digital_heterodyning_complex(rx_din_sig,1024e6,consts.FS)

    #real = rx_din_sig.real
    #imag = rx_din_sig.imag
    #rx_din_sig.real = 0.98 * real + 0.17 * imag
    #rx_din_sig.imag = -1*0.17 * real - 0.98 * imag 

    #rx_din_sig = signal.decimate(rx_din_sig,8,ftype="fir")

    
    #rx_din_sig = rx_din_sig *  np.exp(1j*0.9*np.pi)


    # test: 120000 - 160000
    rx_din_sig = (rx_din_sig[17500:21000])
    pf.plot_all("sig after downsample real",rx_din_sig.real,512e6)
    pf.plot_all("sig after downsample imag",rx_din_sig.imag,512e6)
    
    #rx_din_sig = digital_heterodyning_complex(rx_din_sig,consts.FREQ_SHIFT,512e6)
    #rx_din_sig = signal.decimate(rx_din_sig,consts.SAMPLE_FACTOR,ftype="fir")

    #test_real = rx_din_sig.real[0:9770]
    #test_imag = rx_din_sig.imag[75:9770+75]

    #rx_din_sig = np.zeros(9770,dtype= np.complex64)
    #rx_din_sig.real = test_real
    #rx_din_sig.imag = test_imag

    #pf.plot_all("test_real",test_real,512e6)
    #pf.plot_all("test_imag",test_imag,512e6)

    #imag_peak = -96
    #real_peak = 640
    #sin_phi = (imag_peak) / np.sqrt(imag_peak * imag_peak + real_peak * real_peak)
    #cos_phi = (real_peak) / np.sqrt(imag_peak * imag_peak + real_peak * real_peak)
    #print("cos_phi = ",cos_phi,"\nsin_phi = ",sin_phi)

    #phase_correct_I = cos_phi * test_real + sin_phi * test_imag
    #phase_correct_Q = sin_phi * test_real - cos_phi * test_imag
#
    #test_sig = test_real +  1j*test_imag
#
    #pf.plot_all("phase_correct_I",phase_correct_I,512e6)
    #pf.plot_all("phase_correct_Q",phase_correct_Q,512e6)


    # analyse
    # iamg: 214  real 5318
    #plt.show()


    # time sync
    # not very good i think
    ts_res = signal.convolve(in1=rx_din_sig.real, in2 = np.conj(consts.LTF_matched_filter), mode = "same")
    ts_res = np.abs(ts_res) ** 2
    #ts_res = signal.convolve(in1 = rx_din_sig, in2 = np.conj(consts.head),mode = "same")
    pf.plot_dat("time sync res",ts_res)
    #plt.show()
    pks = signal.find_peaks(ts_res,height = consts.PEAK, distance=64)
    print(pks)

    plt.show()
    
  
    num_frame = int(len(pks[0]) // 2)
    rx_payload_after_fft = np.zeros((consts.N_DATA) * consts.N_OFDM * num_frame,np.complex64)

    print("num_frame = ",num_frame)

    for no_frame in range(num_frame):
        # get index
        LTF_seq_start_index = pks[0][no_frame*2+1] - 95
        if(LTF_seq_start_index + consts.FRAME_LEN - 32  > len(rx_din_sig)):
            break
        print("LTF_seq_start_index = ",LTF_seq_start_index)
        training_seq_start_index = pks[0][no_frame*2+1] + 32 + 1
        payload_start_index = training_seq_start_index + consts.TRAINING_SYMB_LEN

       

        # get LTF seq 
        LTF_symb1 = rx_din_sig[LTF_seq_start_index: LTF_seq_start_index + consts.LTF_SEQ_LEN]
        print("LTF_symb1_real = ",LTF_symb1.real)
        LTF_symb2 = rx_din_sig[LTF_seq_start_index + consts.LTF_SEQ_LEN: LTF_seq_start_index + 2*consts.LTF_SEQ_LEN]

        # est freq off
        r_angle,freq_off = estimate_frequency_offset(LTF_symb1,LTF_symb2)
        np.set_printoptions(suppress=True)
        print("freq_off = ",freq_off)

        # correct freq off
        payload = rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN]
        #rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN] = digital_heterodyning_complex(payload,freq_off,512e6)
        np.set_printoptions(suppress=True)
        print("r_angle = ",r_angle)


        # get trainning seq 
        training_symb1 = rx_din_sig[training_seq_start_index: training_seq_start_index + consts.CP_LEN + consts.N]
        training_symb2 = rx_din_sig[training_seq_start_index + consts.CP_LEN + consts.N: payload_start_index]
        df.save_array_to_file("D:/work/CA/simulation/dats/training_symb.txt",np.concatenate((training_symb1,training_symb2)))
        training_symb_all = np.concatenate((training_symb1,training_symb2))
        df.complex_file_to_IQ_file("D:/work/CA/simulation/dats/training_symb.txt",
                                   "D:/work/CA/simulation/dats/training_symb_I.txt",
                                   "D:/work/CA/simulation/dats/training_symb_Q.txt",
                                   "D:/work/CA/simulation/dats/training_symb_QI.txt")

        # est phase off
        cos_phi , cos_phi_aver, sin_phi,sin_phi_aver =  estimate_phi(training_symb2[consts.CP_LEN])
        #payload = rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN]
        #rx_din_sig [training_seq_start_index: training_seq_start_index + consts.LLTF_SYMB_LEN + consts.PAYLOAD_LEN] = phase_correct(payload,cos_phi,sin_phi)
        
        

        


        #rx_din_sig = rx_din_sig * np.exp(-1j*np.pi*r_angle)

        #get trainning seq 
        training_symb1 = rx_din_sig[training_seq_start_index: training_seq_start_index + consts.CP_LEN + consts.N]
        training_symb2 = rx_din_sig[training_seq_start_index + consts.CP_LEN + consts.N: payload_start_index]

        # channel est using training seq
        channel_est_res = channel_est(ofdm_demod(training_symb1, consts.N, consts.CP_LEN),ofdm_demod(training_symb2, consts.N, consts.CP_LEN))

        

        #get payload
        payload_before_fft = rx_din_sig[payload_start_index:payload_start_index + (consts.CP_LEN + consts.N) * consts.N_OFDM]

        data_rx = np.zeros(consts.N_OFDM * consts.N_DATA,np.complex64)
        j = 0
        k = 0 
        ranging_all = np.zeros([consts.N_OFDM,consts.N])

        for i in range(consts.N_OFDM):
            # ranging
            ofdm_payload_no_cp = payload_before_fft[k + consts.CP_LEN:(k + consts.N + consts.CP_LEN)]
            local_ofdm_payload_no_cp = np.loadtxt(consts.tx_ofdm_payload_with_cp_file,dtype = np.complex64)
            local_ofdm_payload_no_cp_per_symb = local_ofdm_payload_no_cp[k + consts.CP_LEN:(k + consts.N + consts.CP_LEN)]
            ranging_res = ranging(local_ofdm_payload_no_cp_per_symb,ofdm_payload_no_cp,no_frame,i)
            ranging_all[i,:] = ranging_res
            # FFT
            rx_demod = ofdm_demod(payload_before_fft[k:(k + consts.N + consts.CP_LEN)],consts.N,consts.CP_LEN)
            # channel equ using training seq
            rx_demod[consts.OFDM_PAYLOAD_INDEX] = channel_equ(rx_demod[consts.OFDM_PAYLOAD_INDEX],channel_est_res)

            # channel est using pilot 
            #channel_est_pilot_res = channel_est_pilot(rx_demod)
            # channel equ using pilot
            #channel_equ_res = channel_equ_pilot(rx_demod,channel_est_pilot_res)

            #data_rx[j:j+consts.N_DATA] = channel_equ_res
            data_rx[j:j+consts.N_DATA] = rx_demod[consts.OFDM_DATA_INDEX]

            j = j + consts.N_DATA
            k = k + consts.N + consts.CP_LEN 

        #ranging_mean = np.mean(ranging_all,0)
        #ranging_mean_dB = 10*np.log10(ranging_mean/max(ranging_mean))
        #print('对一帧求均值:',np.size(ranging_mean_dB))
        #plt.title([no_frame,'rangeEstimation'])
        #plt.plot(consts.range,ranging_mean_dB)
        #plt.ylabel('Ambiguity (dB)')
        #plt.xlabel('Distance (m)')
        #plt.show()
        

        print("data_rx len = ",len(data_rx))

        rx_payload_after_fft[no_frame*consts.N_DATA*consts.N_OFDM:(no_frame + 1)*consts.N_DATA*consts.N_OFDM] = data_rx #都用有效数据覆盖
        print("rx_payload_after_fft = ",len(rx_payload_after_fft))

    #for i in range (len(rx_payload_after_fft)):
    #    if(abs(rx_payload_after_fft[i].real) > 2e6):
    #        print("i= ",i)    

    pf.plot_constellation_peaks(rx_payload_after_fft) #传入若干组有效数据
    plt.show()
    
    #plot demod res 
    scatterplot(rx_payload_after_fft.real,rx_payload_after_fft.imag,ax=None)

    #demod qam
    qam_demod_res = qam16_symb_to_uint8(rx_payload_after_fft)
    #write demod res
    df.write_uint8_to_ascii(qam_demod_res,rx_output_file)
    return

'''
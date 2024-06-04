import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import signal

from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import data_func as df
def plot_dat(sig_name,sig):
    plt.figure(sig_name + " real")
    plt.title(sig_name+ " real")
    plt.plot(sig.real)

    plt.figure(sig_name + "imag")
    plt.title(sig_name+ " imag")
    plt.plot(sig.imag)
    return

def plot_spectrum(sig_name,sig,sample_rate):
    plt.figure(sig_name + " spectrum")
    plt.title(sig_name + " spectrum")
    f, Pxx = signal.periodogram(sig, fs=sample_rate)
    plt.semilogy(f, np.sqrt(Pxx))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectrum')
    plt.grid(True)
    return

def plot_all(sig_name,sig, sample_rate):
    plot_dat(sig_name,sig)
    plot_spectrum(sig_name,sig,sample_rate)
    return

def plot_constellation_peaks(data):
    """
    绘制星座图
    
    参数：
    data: numpy数组，形状为(30, 40)，包含星座图数据
    
    返回值：
    无
    """
    #num_peaks 
    # num_points = len(data) #输入数据的长度
    # num_points = 12800
    colors = plt.cm.tab20(np.linspace(0, 1, 52))  # 生成不同颜色
    # print('颜色选型共有:',np.size(colors))
    
    fig, ax = plt.subplots(figsize=(8, 6))  # 图像尺寸
    
    # for i in range(num_peaks):
    for i in range(0,52,1):
        indices = np.arange(i, 4*52, 52)
        print('第',i,'个数据子载波',indices)
        x = np.real(data[indices])
        y = np.imag(data[indices])
        ax.scatter(x, y, color=colors[i])
        plt.show()
        
    ax.set_title('Constellation Plot')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend()
    #plt.show()
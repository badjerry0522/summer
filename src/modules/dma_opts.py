import numpy as np
from pynq import allocate
from pynq import Overlay
import threading
import time

def init_tx_mem(rtl_input_file):
    BUFFER_MAX_SIZE = 32000000
    DATA_WIDTH = 16
    axi_dma_2_buffer_max_len = int(int(BUFFER_MAX_SIZE) * 8 / DATA_WIDTH)
    
    rtl_input = np.loadtxt(rtl_input_file)
    rtl_input_len = len(rtl_input)
    print("rtl_input_len = ",rtl_input_len)
    
    tile_times = int(axi_dma_2_buffer_max_len / rtl_input_len)
    print("tile_times = ",tile_times)
    
    input_buffer = allocate(shape=(axi_dma_2_buffer_max_len,),dtype = np.int16,cacheable=False)
    for i in range(tile_times):
       for j in range(rtl_input_len):
          input_buffer[i*rtl_input_len + j] = rtl_input[j]
          
    print("tx mem init complete")
    return input_buffer

def init_rx_mem(BUFFER_MAX_SIZE):
    DATA_WIDTH = 16
    axi_dma_2_buffer_max_len = int(int(BUFFER_MAX_SIZE) * 8 / DATA_WIDTH)
    output_buffer = allocate(shape=(axi_dma_2_buffer_max_len,),dtype = np.int16,cacheable=False)
    print("rx mem init complete")
    return output_buffer

def tx_dma_thread(stop_event, dma, input_buffer):
    print("tx dma running")
    while not stop_event.is_set():
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        
    print("tx dma stop")
    
def start_tx_dma(dma,input_buffer):
    stop_tx_dma = threading.Event()
    thread = threading.Thread(target=tx_dma_thread, args=(stop_tx_dma,dma,input_buffer,))
    thread.start()
    return stop_tx_dma, thread

def rx_dma_thread(stop_event, dma, output_buffer):
    print("rx dma running")
    while not stop_event.is_set():
        dma.recvchannel.transfer(output_buffer)
        dma.recvchannel.wait()
    print("rx dma stop")
    
def rx_dma_transfer_once(dma,output_buffer):
    print("rx dma transferring once")
    dma.recvchannel.transfer(output_buffer)
    #dma.recvchannel.wait()
    print("rx dma transfer once complete")

def start_rx_dma(dma,output_buffer):
    stop_rx_dma = threading.Event()
    thread = threading.Thread(target=rx_dma_thread, args=(stop_rx_dma,dma,output_buffer,))
    thread.start()
    return stop_rx_dma, thread
    
    
    
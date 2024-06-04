import shlex
import argparse
def parse_args_from_file(file_path):
    # 读取参数文件
    with open(file_path, 'r') as file:
        # 将文件内容读取到一个字符串
        file_content = file.read()

    # 使用 shlex.split() 以正确处理参数中可能包含的引号和空格
    args = shlex.split(file_content)

    # 创建 argparse 解析器
    parser = argparse.ArgumentParser(description='Process parameters from file.')
    # 添加参数定义
    parser.add_argument("--EXE_MODE", help="Execution mode", type=str)
    parser.add_argument("--N", help="N of FFT", type=int)
    parser.add_argument("--QAM_MODE", help="QAM modulation type", type=int)
    parser.add_argument("--N_OFDM", help="Number of OFDM symbols per frame", type=int)
    parser.add_argument("--PILOT_MODE", help="Pilot mode", type=str)
    parser.add_argument("--N_FRAMES", help="Number of frames", type=int)
    parser.add_argument("--PEAK", help="Peak value", type=int)
    parser.add_argument("--SNR", help="Signal to Noise Ratio", type=int)
    parser.add_argument("--TX_INPUT_QAM_FILE", help="Input QAM file for transmission", type=str)
    parser.add_argument("--TX_OUTPUT_QAM_NO_PILOT_FILE", help="Output QAM file without pilot for transmission", type=str)
    parser.add_argument("--TX_OUTPUT_QAM_PILOT_FILE", help="Output QAM file with pilot for transmission", type=str)
    parser.add_argument("--TX_OUTPUT_SINGLE_FRAME_FILE",help ="Transmitter output file for single frame", type = str)
    parser.add_argument("--TX_OUTPUT_FILE", help="Transmission output file", type=str)
    parser.add_argument("--CHANNEL_OUTPUT_FILE", help="Channel output file", type=str)
    parser.add_argument("--RTL_OUTPUT_I_FILE", help="RTL output I channel file", type=str)
    parser.add_argument("--RTL_OUTPUT_Q_FILE", help="RTL output Q channel file", type=str)
    parser.add_argument("--RTL_OUTPUT_COMPLEX_FILE", help="RTL output complex file", type=str)
    parser.add_argument("--DATS_DIR", help="DATA DIR", type=str)
    parser.add_argument("--RTL_BITFILE",help="RTL bitfile",type=str)
    # 解析从文件读取的参数
    args = parser.parse_args(args)

    # 打印解析后的参数，确认
    print("Parsed args from file: '{}'".format(args))
    return args
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import signal

def bus_array_to_complex_array(input_arr):
    output_arr_len = int(len(input_arr)/2)
    output_arr = np.zeros(output_arr_len,dtype = np.complex64)
    for i in range(output_arr_len):
        output_arr[i] = 1j*input_arr[i*2] + input_arr[i*2+1]
        
    return output_arr
        

def txt_file_to_coe_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    

    # 在除了最后一行的末尾加一个逗号
    for i in range(len(lines) - 1):
        if lines[i].strip() != "":
            lines[i] = lines[i].rstrip() + ",\n"
    
    # 最后一行加分号
    if lines[-1].strip() != "":
        lines[-1] = lines[-1].rstrip() + ";\n"
    else:
        lines[-2] = lines[-2].rstrip() + ";\n"

    # 在前两行添加memory_initialization_radix=2; memory_initialization_vector=
    lines.insert(0, "memory_initialization_radix=16;\n")
    lines.insert(1, "memory_initialization_vector=\n")
    
    # 写入修改后的内容到输出文件
    with open(output_file, 'w') as f:
        f.writelines(lines)

def complex_file_to_IQ_file(complex_file_in, I_file_out, Q_file_out, rtl_test_file):
    complex_data = np.loadtxt(complex_file_in,dtype=np.complex64)
    complex_data_Q = complex_data.imag
    complex_data_I = complex_data.real
    np.savetxt(I_file_out,complex_data_I,fmt="%d")
    np.savetxt(Q_file_out,complex_data_Q,fmt="%d")

    with open(rtl_test_file, 'w') as f:
            for complex_number in complex_data.flatten():
                # 将虚部和实部四舍五入为整数，并写入文件
                f.write(f"{int(round(complex_number.imag))}\n")
                f.write(f"{int(round(complex_number.real))}\n")
    return

def IQ_file_to_complex_file(I_file_in,Q_file_in,comeplx_file_out):
    real_data = np.loadtxt(I_file_in)
    imag_data = np.loadtxt(Q_file_in)
    
    # 将实部和虚部组合成复数数组
    complex_data = real_data + 1j * imag_data
    np.savetxt(comeplx_file_out,complex_data,fmt="%.6f")

def nparr_to_ceofile(arr_in, file_out):
    return

def text_to_128bit(file_in, file_out):
    with open(file_in, 'r') as f_in, open(file_out, 'w') as f_out:
        while True:
            chunk = f_in.read(16)
            if not chunk:
                break
            # 确保每个chunk都是16个字符，不足处用空格填充
            chunk = chunk.ljust(16, ' ')
            # 输出每个字符及其ASCII码值到终端
            for c in chunk:
                print(f"{c}: {ord(c)}")
            # 将每16个字符转换为ASCII码，然后拼接成一个128位数
            decimal_value = 0
            for c in chunk:
                decimal_value = (decimal_value << 8) + ord(c)
            # 将128位数以一行一个的形式输出到文件中
            #f_out.write(format(decimal_value, '0128b') + '\n')
            f_out.write(format(decimal_value, '032x') + '\n')

def save_array_to_file(file_name, array):
    np.savetxt(file_name, array,fmt="%.6f")

def convert_ascii_to_uint8(file1, file2):
    # 读取文本文件
    with open(file1, 'r') as f:
        content = f.read()

    # 将ASCII字符转换为uint8格式的NumPy数组
    uint8_array = np.array([ord(char) for char in content], dtype=np.uint8)

    # 将uint8格式的NumPy数组以每行一个数的格式写入文件
    with open(file2, 'w') as f:
        for value in uint8_array:
            f.write(str(value) + '\n')

def fprint_uint8_arr(arr1, output_file):
# 将uint8格式的NumPy数组中的每个数以每行一个数的格式写入文件
    with open(output_file, 'w') as file:
        for value in arr1:
            file.write(str(value) + '\n')

def ascii_to_bin(ascii_text):
    binary_result = ''.join(format(ord(char), '08b') for char in ascii_text)
    return binary_result

def read_ascii_file(input_file):
    file1 = open(input_file,'r',encoding="ascii") 
    ascii_text = file1.read()
    return ascii_text

def ascii_to_binary(input_file, output_file):
    try:
        # 打开文件1以读取ASCII码
        with open(input_file, 'r', encoding='ascii') as file1:
            # 读取文件内容
            ascii_text = file1.read()

            # 将ASCII码转换为二进制
            binary_data = ''.join(format(ord(char), '08b') for char in ascii_text)

        # 打开文件2以写入二进制数据
        with open(output_file, 'wb') as file2:
            # 将二进制数据写入文件2
            file2.write(bytes(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8)))

        print(f"成功将ASCII码从 {input_file} 转换为二进制并写入 {output_file}。")

    except FileNotFoundError:
        print("文件未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

def binary_to_ascii(input_file, output_file):
    try:
        # 打开二进制文件以读取数据
        with open(input_file, 'rb') as file1:
            # 读取二进制数据
            binary_data = file1.read()

            # 将二进制数据转换为ASCII码
            ascii_text = ''.join(chr(byte) for byte in binary_data)

        # 打开文件2以写入ASCII码
        with open(output_file, 'w', encoding='ascii') as file2:
            # 将ASCII码写入文件2
            file2.write(ascii_text)

        print(f"成功将二进制数据从 {input_file} 转换为ASCII码并写入 {output_file}。")

    except FileNotFoundError:
        print("文件未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

def write_uint8_to_ascii(arr1, ascii_file):
    # 将uint8格式的NumPy数组转换为ASCII字符串
    ascii_data = ''.join(map(chr, arr1))

    # 将ASCII字符串写入文件
    with open(ascii_file, 'w') as file:
        file.write(ascii_data)

def bit16_to_256bit(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 将每行的有符号数存入列表中
    numbers = []
    for line in lines:
        numbers.append(int(line.strip()))

    # 将每16个有符号数合并为一个256位数，并以十六进制格式写入输出文件
    with open(output_file, 'w') as f:
        for i in range(0, len(numbers), 16):
            combined_number = 0
            for j in range(16):
                combined_number |= (numbers[i + j] & 0xFFFF) << (16 * (15 - j))
            f.write(format(combined_number, '064x') + '\n')
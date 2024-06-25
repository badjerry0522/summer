def generate_m_sequence(register_length):
    # 选择一个本原多项式，这里使用的是x^11 + x^2 + 1
    # 寄存器的初始状态不能全为0，这里选择全为1
    feedback_tap = 0b100000000110  # 多项式系数，对应于x^11, x^2的项为1，其余为0
    state = 0b11111111111  # 初始状态全1
    period = 2 ** register_length - 1
    sequence = []

    for _ in range(period):
        # 计算输出位，即LFSR的最左边位
        output = state & 1
        # 将输出的0映射为-1
        mapped_output = -1 if output == 0 else 1
        sequence.append(mapped_output)
        # 计算反馈位，即通过多项式决定的几个位的异或
        feedback = state & feedback_tap
        # 计算汉明重量，即1的数量
        feedback = bin(feedback).count('1') % 2
        # 左移一位
        state = (state >> 1) | (feedback << (register_length - 1))
    
    return sequence
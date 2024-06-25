import sys
import os
sys.path.append('D:/work/CA/summer/src/modules/')  # 添加上级目录到路径
import parse_args
import configs as cfgs
import transmitter as tx

file_args = parse_args.parse_args_from_file("D:/work/CA/summer/src/modules/arg_configs.txt")
ofdm_cfgs = cfgs.ofdm_config(file_args)
tx0 = tx.tx(ofdm_cfgs)
tx0.tx_run()
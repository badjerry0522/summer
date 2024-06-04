# SUMMER !!!

# 整体架构

summer v1.0是用于OFDM通信-定位一体化的数字基带系统，负责软件调制解调和硬件采样转换，可与太赫兹前端进行连接。本章将首先介绍summer v1.0的整体工作流程，然后简单介绍硬件、软件的运行架构。

## 整体流程 

v1.0版本的整体架构如图所示。其中，发射信号提前生成好，并存储在ROM中，循环读出至DAC进行数模转换，然后送入前前端进行发射。接收端在ADC采样后，通过DMA写入DDR内存中，并用python读取和处理。python读取后，首先进行时间同步、解帧，然后分别进行通信和定位计算，最后将结果显示在jupyter notebook上。

![image-20240603171633799](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240603171633799.png)

summer v1.0共分为硬件和软件两部分，硬件部分在D:\work\fpga\Thz\CA\prjs\summer目录下，为vivado工程文件；软件部分部署在开发板上，其目录为/home/xilinx/jupyter_notebooks/summer。

## 硬件

硬件架构如图所示。发射端TX ROM与DAC通过AXI-S总线连接，直接将发射信号循环打入DAC中；接收端则是分为IQ两路，通过RX DMA循环从ADC中读取采样数据，并写入DDR内存中供软件处理。

![image-20240603172549129](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240603172549129.png)

硬件的具体实现架构如D:\work\CA\summer\docs\summer_design.pdf所示。

## 软件

软件平台为python-jupyter notebook，可以使用vscode连接并远程实时显示星座图。jupyter notebook位于/home/xilinx/jupyter_notebooks/summer/summer.ipynb，其计算模块位于/home/xilinx/jupyter_notebooks/summer/modules目录下。

# 硬件参数与比特文件生成

## vivado工程

vivado所使用的版本为2023.2，v1.0版本有如下这些源码文件：

![image-20240604103214929](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604103214929.png)

其中，summer_design_wrapper为block design自动生成的顶层文件；adc2axis_IQ为adc DMA同步模块，具体原理见毕业论文4.3.2.2节；tx_blk_ram_reader为发射端ROM读取模块，用于从ROM中读取发射数据并通过AXI-S总线发往DAC；tx_coe.coe为存储发射信号的coe文件。

vivado工程中，左侧栏中有block design，点开即可进行block design设计。

![image-20240604110301653](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604110301653.png)

## DC参数设置

在block design中，Data Converter的参数决定了OFDM的带宽。首先，在system clocking处选择ADC和DAC的采样率，summer v1.0的DAC、ADC采样率均为4.096Ghz。我们使用的参考时钟为409.6MHz的参考时钟，此时钟有LMK时钟芯片提供，后面在软件部分会讲解如何配置。

![image-20240604110416509](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604110416509.png)

在Basic界面，可以对ADC/DAC进行配置。DAC配置如下：

![image-20240604110544825](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604110544825.png)

在4.096Ghz下，使用4x上变频，即发射带宽为：
$$
B = 4.096GHz/4=1.024Ghz
$$
输入信号的总线为16Samples/AXI Cycle，即256位总线宽度，其需要的AXI-S时钟为128Mhz。其使用的数字混频为1.024Ghz，即中频为1.024Ghz。更多IP核相关信息详见XIlinx手册：pg269

ADC同理：

![image-20240604110901784](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604110901784.png)

在Data Converter IP核界面左侧，点击ADC Physical Resources，可以看到ADC编号和其连接的总线接口：

![image-20240604111140854](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604111140854.png)

## 发射链路ROM与控制逻辑

为了节省DDR的带宽，且增大发射端的稳定性，summer v1.0版本使用固定发射信号，发射信号存在一个ROM中，且通过一个专门的rom reader进行读取，如下图所示：

![image-20240604112001575](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604112001575.png)

在使用LTF+2训练序列+4数据序列的帧结构下，一个帧长为640采样点，一个采样点由I、Q两个16位采样数据组成，一帧共20480字节。所以，v1.0使用的ROM是一个位宽为256，深度为80的ROM。**注意！如果帧结构变化，ROM的深度是可改的！但是为了与AXI-S总线位宽匹配，ROM位宽不可改！**

![image-20240604112221076](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604112221076.png)

ROM中的数据是从coe文件载入的，coe文件地址为：D:\work\fpga\Thz\CA\prjs\summer\tx_data_files\coe_file\tx_coe.coe coe文件

![image-20240604113709074](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20240604113709074.png)

## coe文件生成

在D:\work\fpga\Thz\CA\prjs\summer\tx_data_files目录下的data2coe.py会将rtl_input_file中由发射机生成的rtl_test_input.txt文件转化为对应格式的coe文件，并存储在D:\work\fpga\Thz\CA\prjs\summer\tx_data_files\coe_file\目录下。注意，生成后的tx_coe.coe是很长的，需要手动将其缩减至与ROM深度相符的长度。如，当ROM深度为80时，.coe的数据行数只应有80行。

# 软件使用与配置

软件部分集成了一个发射机(transmitter.py)、一个接收机(receiver.py)，并为其设置了一个参数读取器(configs.py和parse_args.py)，用于从arg_configs.txt中读取参数。具体的参数设置在classes.md中。**注意！classes.md尚未完善，仅供参考，后面会一点一点完善**

## summer.ipynb的使用

summer.ipynb中有详细的使用说明，在这个文档后面我们会慢慢加入python程序的解释与说明。
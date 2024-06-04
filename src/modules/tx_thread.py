import os
from IPython.display import clear_output
def loop_1():
    i = 0
    while(i<10):
        print("i=",i)
        if(i==9):
            i=0
            clear_output(wait=True)  # 使用 wait=True 来延迟清除直到新的输出准备好
        else:
            i=i+1
            
    
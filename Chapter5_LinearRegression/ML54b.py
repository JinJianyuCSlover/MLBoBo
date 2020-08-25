import numpy as np
import time
from functions.SimpleLinearRegression import *

m=10000000
big_x = np.random.random(size=m)#x随机好多数字
big_y = big_x*2.0+3.0+np.random.normal(size=m)#是每一项进行线性运算加干扰项的意思

reg1=SimpleLinearRegression1()
reg2=SimpleLinearRegression2()

"""开始时间大比拼"""
start_time_1 = time.time()#开始计时
reg1.fit(big_x,big_y)
end_time_1 = time.time()#计时结束

start_time_2 = time.time()#开始计时
reg2.fit(big_x,big_y)
end_time_2 = time.time()#计时结束

print("利用方法1不用向量耗时{:.5}s".format(end_time_1-start_time_1))
print("利用方法2使用向量耗时{:.5}s".format(end_time_2-start_time_2))



import numpy as np
import matplotlib.pyplot as plt
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

x_mean = np.mean(x)
y_mean=np.mean(y)

'''
先来看bobo老师所谓的a也就是西瓜书里的w，有分子分母两部分。分开来命名
'''
#a的分子
a_frac_up=0.0
#a的分母
a_frac_down=0.0
#求和吗，循环少不了。把xy放进一个zip里面方便取值
for x_i,y_i in zip(x,y):
    a_frac_up += (x_i-x_mean)*(y_i-y_mean)
    a_frac_down+=(x_i-x_mean)**2

a=a_frac_up/a_frac_down
b=y_mean-(a*x_mean)

y_hat = a*x+b
plt.scatter(x,y)
plt.plot(x,y_hat,color='red')
plt.axis([0,6,0,6])
plt.show()

#现在有新的值了
x_predict = 6
y_predict = a*x_predict+b
print(y_predict)
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
'''
先给你提供所有数据，包括训练空间（kNN特点导致没有测试空间）以及待检测样本x
'''
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]#特征空间
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]#标签0良性1恶性
X_train=np.array(raw_data_X)#矩阵
y_train=np.array(raw_data_y)#向量
x = np.array([8.093607318, 3.365731514])#新来的没分类的样本

'''
计算距离，欧拉距离
'''
distances = []#记录距离
for x_train in X_train:
    d=sqrt(np.sum((x_train-x)**2))#x方向上的举例
    distances.append(d)

'''
计算完距离开始分类，使用np自带函数进行分类
摘取数据，只摘取想要的数据，就是k=6
'''
nearest=np.argsort(distances)#排序之后返回数据的索引从进到远都是哪几个
k=6#kNN的k
topK_y=[y_train[i] for i in nearest[:k]]#最近的几个样本的标签都是多少
'''
真的是选取6个也就是nearest[0]~nearest[5]
之后开始看0和1谁多并且给出相应的判断
'''
votes=Counter(topK_y)#开始投票啦
#各个参数释疑：选取最多票数的1个；选取元组里的第0个元素；元组里的第0个元素是列表，选列表里的第0个就是结果
res=votes.most_common(1)[0][0]
if str(res)=='0':
    print('良性肿瘤')
else:
    print('恶性肿瘤')

'''
画散点图来直观感受一下
'''
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='green')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='red')
plt.scatter(x[0],x[1],color='blue')
plt.show()



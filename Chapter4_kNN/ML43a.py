import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
X=iris.data#特征矩阵
y=iris.target#结果标签向量
'''
对X进行train-test split还不能一刀切，因为y都是排好序的012，一刀切测试集都是一个标签。
'''
#形成150个样本的索引的随机排列
shuffle_indexes = np.random.permutation(len(X))#为不失一般性写成X的长度
test_raito=0.2#测试数据集的比例是0.2
test_size=int(len(X)*test_raito)
test_indexes=shuffle_indexes[:test_size]#测试数据集的索引是前30个
train_indexs=shuffle_indexes[test_size:]#训练数据集是后面的120个
#获取训练空间和测试空间
X_train=X[train_indexs]#120*4
y_train=y[train_indexs]#120

X_test=X[test_indexes]#30*4
y_test=y[test_indexes]#30


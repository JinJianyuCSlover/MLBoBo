import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()#iris变量里面就是数据集
#iris可以理解成一个字典
print(iris.keys())#特征的组合都是什么dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(iris.DESCR)


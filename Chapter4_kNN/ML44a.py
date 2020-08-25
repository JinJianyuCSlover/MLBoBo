import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

from functions.model_selection import train_test_split
from functions.kNN import KNNClassifier
from functions.metrics import accuracy_score

digits =datasets.load_digits()
print(digits.keys())#看看字典里有哪些keys
X=digits.data#1797*64
y=digits.target#1797*1
print(digits.target_names)#我们看看digits输出的标签都是什么，发现是0-9
'''让我看看这个数据集是什么东西？我来选一个样本看看'''
example_feature = X[1200]
example_target = y[1200]
# print(example_feature)
# print(example_target)
#使用一个绘图函数看看这是什么，具体参照莫烦python的图像image那节
example_feature_image = example_feature.reshape(8,8)
plt.imshow(example_feature_image,cmap=matplotlib.cm.binary)
plt.show()
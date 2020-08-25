'''
4-3使用封装好的包来调用
'''
from functions.model_selection import train_test_split
from functions.kNN import KNNClassifier
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X=iris.data#特征矩阵
y=iris.target#结果标签向量
X_train,X_test,y_train,y_test = train_test_split(X,y)
my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train,y_train)
y_predict = my_knn_clf.predict(X_test)#有30个预测结果看看和y_test有的不一样
'''
计算模型的准确率我的直白方法vs老师的简练方法
'''
#count=0
# for i in range(len(y_predict)):
#     if y_predict[i]==y_test[i]:
#         count+=1
# print(count*100/len(y_test))
sum(y_predict==y_test)/len(y_test)#会返回一个全是bool的列表，true为1false为0加起来就是预测对的的数目之后除以总数
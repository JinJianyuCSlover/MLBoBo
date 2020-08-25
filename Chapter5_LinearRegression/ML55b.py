import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from functions.model_selection import train_test_split
from functions.SimpleLinearRegression import SimpleLinearRegression2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


'''加载数据集'''
boston=datasets.load_boston()
X=boston.data[:,5]#只选取第5列——房间数量
y=boston.target
# plt.scatter(X,y)
# plt.show()
'''会有很多超出上限的点，需要去除'''
X=X[y<50.0]
y=y[y<50.0]
# plt.scatter(X,y)
# plt.show()
'''进行测试训练分割'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_ratio=0.2,seed=666)
reg2=SimpleLinearRegression2()
reg2.fit(X_train,y_train)
print(reg2.a_)
print(reg2.b_)
#看看训练怎么样
plt.scatter(X_train,y_train)
# plt.plot(X_train,reg2.predict(X_train),color='red')
# plt.show()

#预测一波
y_predict = reg2.predict(X_test)

"""scikit learn里面MSE与MAE"""
mean_squared_error(y_test,y_predict)
mean_absolute_error(y_test,y_predict)

"""R Square"""
print(1-mean_squared_error(y_test,y_predict)/np.var(y_test))

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from functions.model_selection import train_test_split
from functions.SimpleLinearRegression import SimpleLinearRegression2

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




"""MSE"""
mse_test=np.sum((y_predict-y_test)**2)/len(y_test)
"""RMSE"""
rmse_test = math.sqrt(mse_test)
"""MAE"""
mae_test=np.sum(np.absolute(y_test-y_predict))/len(y_test)

print("MSE:{:.5}, and RMSE:{:.5}, meanwhile MAE:{:.5}".format(mse_test,rmse_test,mae_test))

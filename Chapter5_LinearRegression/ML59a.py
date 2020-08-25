import numpy as np
from functions.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression


"""进行数据预处理，别忘了剔除异常值"""
boston = datasets.load_boston()
X=boston.data
y=boston.target
X=X[y<50.0]
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_ratio=0.2,seed=666)

"""sklearn里面的线性回归"""
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)#注意sklearn的fit逻辑和我们的不一样

print(lin_reg.coef_)
print(lin_reg.intercept_)
print(lin_reg.score(X_test,y_test))




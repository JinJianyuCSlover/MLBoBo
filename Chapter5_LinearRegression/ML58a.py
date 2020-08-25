from functions.LinearRegression import LinearRegression
from functions.model_selection import train_test_split
from sklearn import datasets

"""进行数据预处理，别忘了剔除异常值"""
boston = datasets.load_boston()
X=boston.data
y=boston.target
X=X[y<50.0]
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_ratio=0.2,seed=666)

"""开始进行回归"""
reg = LinearRegression()
reg.fit_normal(X_train,y_train)

print(reg.coef_)
print(reg.interception_)
print(reg.score(X_test,y_test))
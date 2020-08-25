from functions.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np


"""进行数据预处理，别忘了剔除异常值"""
boston = datasets.load_boston()
X=boston.data
y=boston.target
X=X[y<50.0]
y=y[y<50.0]
# X_train, X_test, y_train, y_test=train_test_split(X,y,test_ratio=0.2,seed=666)

"""先来线性回归"""
lin_reg = LinearRegression()
lin_reg.fit(X,y)#不是进行预测，不看准确度，就不分了
print(np.argsort(lin_reg.coef_))#按照角标对影响因素大小进行排序
print(boston.feature_names[np.argsort(lin_reg.coef_)])#角标对应到特征的名字，越往后越能拉升房价，前面的倒贴钱

#自己加的，看看哪个有利哪个不利
positive_factors=[]
negative_factors=[]
for i in range(0,len(lin_reg.coef_)):
    if lin_reg.coef_[i]>=0:
        positive_factors.append(int(i))
    else:
        negative_factors.append(int(i))

print(boston.feature_names[positive_factors])
print(boston.feature_names[negative_factors])
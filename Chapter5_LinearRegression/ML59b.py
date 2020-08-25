import numpy as np
from functions.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

"""进行数据预处理，别忘了剔除异常值"""
boston = datasets.load_boston()
X=boston.data
y=boston.target
X=X[y<50.0]
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_ratio=0.2,seed=666)

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train,y_train)
print(knn_reg.score(X_test,y_test))#显然劣于KNN，但是这个参数不好

"""网格搜索超参数（KNN特别的步骤）"""
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]
knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg,param_grid,n_jobs=-1,verbose=1)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_.score(X_test,y_test))#这个数值才是和多元线性回归同一个维度上最优的值


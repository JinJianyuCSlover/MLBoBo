import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target
"""之前使用的train_test_split超参数调整"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 10):
    for p in range(1, 5):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_p = p
            best_k = k

print("best K = ",best_k)
print("best p = ",best_p)
print("best score = ",best_score)
"""
besst K =  3
besst p =  4
besst score =  0.9860917941585535
"""
print("================================")

"""使用交叉验证"""
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()

# 使用交叉验证，有改动
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 10):
    for p in range(1, 5):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores=cross_val_score(knn_clf,X_train,y_train)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_p = p
            best_k = k

print("best K = ",best_k)
print("best p = ",best_p)
print("best score = ",best_score)
"""
besst K =  2
besst p =  2
besst score =  0.9851507321274763
不一样？相信交叉验证虽然score低，但是不会过拟合所以才低一些
"""

best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=2)
best_knn_clf.fit(X_train,y_train)
print(best_knn_clf.score(X_test,y_test))  # 0.980528511821975

"""回顾网格搜索"""

from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(2, 11)],
        'p': [i for i in range(1, 6)]
    }
]

grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_
best_knn_clf = grid_search.best_estimator_  # 最佳参数对应的最佳分类器
best_knn_clf.score(X_test, y_test)

"""cv参数"""

cross_val_score(knn_clf, X_train, y_train, cv=5)  # 默认分3份，现在分5份训练5模型
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=5)

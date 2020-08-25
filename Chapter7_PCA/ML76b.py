import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))  # 0.9866666666666667
"""使用PCA对digits降维"""

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)

print(knn_clf.score(X_test_reduction, y_test))  # 0.6066666666666667
print(pca.explained_variance_ratio_)  # [0.14566817 0.13735469]第一个轴解释14%，第二个轴解释13%

# 64个特征都给你裹进去
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print(pca.explained_variance_ratio_)
plt.plot([i for i in range(X_train.shape[1])],
         [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(X_train.shape[1])])
plt.show()  # 图片2

# pca封装好了可以使贴合度自动达到95%的方法
pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))  # 0.98

# PC降到2D就不行吗？方便画图
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)

for i in range(10):
    plt.scatter(X_reduction[y == i, 0], X_reduction[y == i, 1], alpha=0.8)
plt.show()  # 图片3

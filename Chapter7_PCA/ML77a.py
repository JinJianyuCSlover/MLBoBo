from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype='float32')
y_train = np.array(y[:60000], dtype='float32')
X_test = np.array(X[60000:], dtype='float32')
y_test = np.array(y[60000:], dtype='float32')

"""使用knn"""
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))

"""使用PCA降维"""
pca=PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
print(X_train.shape)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))

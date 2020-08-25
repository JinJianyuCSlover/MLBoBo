import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
#Scikit learn中的Standard Scalar
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)#已经存储了需要归一化的，会有关键信息
print(standard_scaler.mean_)#返回4个，因为有4个特征啊
print(standard_scaler.scale_)#方差
X_transform=standard_scaler.transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
#开始kNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_transform,y_train)
print(knn_clf.score(X_test_standard,y_test))





import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

"""二分类怎么解决三分类的鸢尾花？只选前两类和两个特征（画图方便）"""
X = X[y < 2, :2]
y = y[y < 2]

"""使用自己的方法"""
from functions.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
from functions.LogisticRegression import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


def x2(x1):
    return (-log_reg.coef_[0] * x1 - log_reg.intercept_) / log_reg.coef_[1]


x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)

plt.plot(x1_plot, x2_plot)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
plt.show()


def plot_decision_boundary(model, axis):
    """绘制函数不要求掌握，理解思维"""
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),  # x轴划分范围
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),  # y轴划分范围
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

"""kNN也是有边界的就是没有表达式，回顾一下"""
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)
plot_decision_boundary(knn_clf, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

"""上面那个还是两类没意思，下面看看三类的"""
lnn_clf_all = KNeighborsClassifier()
lnn_clf_all.fit(iris.data[:, :2], iris.target)
plot_decision_boundary(lnn_clf_all, axis=[4, 8, 1.5, 4.5])
plt.scatter(iris.data[iris.target == 0, 0], iris.data[iris.target == 0, 1])
plt.scatter(iris.data[iris.target == 1, 0], iris.data[iris.target == 1, 1])
plt.scatter(iris.data[iris.target == 2, 0], iris.data[iris.target == 2, 1])
plt.show()

lnn_clf_all = KNeighborsClassifier(n_neighbors=50)
lnn_clf_all.fit(iris.data[:, :2], iris.target)
plot_decision_boundary(lnn_clf_all, axis=[4, 8, 1.5, 4.5])
plt.scatter(iris.data[iris.target == 0, 0], iris.data[iris.target == 0, 1])
plt.scatter(iris.data[iris.target == 1, 0], iris.data[iris.target == 1, 1])
plt.scatter(iris.data[iris.target == 2, 0], iris.data[iris.target == 2, 1])
plt.show()

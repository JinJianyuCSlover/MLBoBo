import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0, 100., size=100)  # 特征1
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)  # 特征2，具有基本的线性关系


# 为什么有线性关系？降维效果明显
# plt.scatter(X[:,0],X[:,1])
# plt.show()

# demean
def demean(x):
    return x - np.mean(x, axis=0)  # 矩阵减去向量，向量是每一列的均值


# 对X进行demean
X_demean = demean(X)
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()
# 现在在xy两个方向上的均值均为0
print(np.mean(X_demean[:, 0]))  # 1.4566126083082053e-14
print(np.mean(X_demean[:, 1]))  # -9.663381206337363e-15

# 梯度上升法
"""
len(X)是样本数
"""


def f(w, X):
    return np.sum((X.dot(w) ** 2)) / len(X)


"""这个方程PPT里面有"""


def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)


"""这个方法可能是之前有的，用于验证求解梯度是否正确"""
"""为什么epsilon取值比较小？因为w是一个方向梯度，模为1，每个维度都很小，所以epsilon也很小"""


def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res


"""梯度上升法"""
"""和梯度下降大体一样，记录循环次数，先求梯度记录上一次再使用梯度上升计算新的没有超过限度就增加下去"""
"""w仅代表方向，是一个单位向量。怎么让它变成一个单位向量？新建方法"""


def direction(w):
    """化为单位向量"""
    return w / np.linalg.norm(w)  # 求模用的这个函数


def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)  # w变成单位向量，搜索更加顺畅
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)  # 注意1：每次w都要成为单位方向向量
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break

        cur_iter += 1

    return w


initial_w = np.random.random(X.shape[1])  # 注意2：不能从零向量开始
eta = 0.001
# 注意3：PCA不能用StandardScaler而且之前的demean也已经标准化了一半
w = gradient_ascent(df_debug, X_demean, initial_w, eta)  # 求出来的第一个主成分，第一主成分

# plt.scatter(X_demean[:, 0], X_demean[:, 1])
# plt.plot([0, w[0] * 100], [0, w[1] * 100], color='red')
# plt.show()

X2 = np.empty((100, 2))
X2[:, 0] = np.random.uniform(0., 100., size=100)
X2[:, 1] = 0.75 * X2[:, 0] + 3.

X2_demean = demean(X2)
w2 = gradient_ascent(df_math, X2_demean, initial_w, eta)
plt.scatter(X2_demean[:, 0], X2_demean[:, 1])
plt.plot([0, w2[0] * 30], [0, w2[1] * 30], color='r')
plt.show()

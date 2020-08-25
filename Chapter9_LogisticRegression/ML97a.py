import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:, 0] ** 2 + X[:, 1]) < 1.5, dtype='int')

for _ in range(20):
    # 这个_就是一个循环标志，也可以用i，j 等其他字母代替
    y[np.random.randint(200)] = 1  # 随机挑取数字设置成1就是噪音

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 使用sklearn中的逻辑回归
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(log_reg.score(X_train, y_train))  # 0.7933333333333333
print(log_reg.score(X_test, y_test))  #


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

"""使用多项式项进行逻辑回归"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)

print(poly_log_reg.score(X_train, y_train))  # 0.906666666
print(poly_log_reg.score(X_test, y_test))  # 0.94

# 画图，多项式逻辑回归
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
print("==============================")

"""开始过拟合"""

poly_log_reg2 = PolynomialLogisticRegression(degree=20)
poly_log_reg2.fit(X_train, y_train)
print(poly_log_reg2.score(X_train, y_train))  # 0.94
print(poly_log_reg2.score(X_test, y_test))  # 0.92
# plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
# plt.scatter(X[y==0,0], X[y==0,1])
# plt.scatter(X[y==1,0], X[y==1,1])
# plt.show()
print("==============================")

"""正则化模型，就是加入C"""


def PolynomialLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])


poly_log_reg3 = PolynomialLogisticRegression(degree=20, C=0.1)
poly_log_reg3.fit(X_train, y_train)
print(poly_log_reg3.score(X_train, y_train))
print(poly_log_reg3.score(X_test, y_test))
plot_decision_boundary(poly_log_reg3, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
print("==============================")

"""依旧正则化，加入penalty是L2不是12"""


def PolynomialLogisticRegression(degree, C, penalty='l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C, penalty=penalty))
    ])


poly_log_reg4 = PolynomialLogisticRegression(degree=20, C=0.1, penalty='l1')  # 使用l1正则项
poly_log_reg4.fit(X_train, y_train)
print(poly_log_reg4.score(X_train, y_train))
print(poly_log_reg4.score(X_test, y_test))
plot_decision_boundary(poly_log_reg4, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
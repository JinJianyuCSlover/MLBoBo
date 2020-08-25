import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # 引入多项式回归
from sklearn.linear_model import LinearRegression

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * (x ** 2) + x + 2 + np.random.normal(0, 1, size=100)

poly = PolynomialFeatures(degree=2)  # 添加最多几次幂？
poly.fit(X)
X2 = poly.transform(X)  # X转换成多项式特征

lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict = lin_reg2.predict(X2)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
plt.show()  # 图片1

X = np.arange(1, 11).reshape(-1, 2)
print(X.shape)  # (5,2)
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly.fit(X)
X2 = poly.transform(X)
print(X2.shape)  # (5,6)

"""方便使用多项式回归Pipeline不需要重复步骤"""
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * (x ** 2) + x + 2 + np.random.normal(0, 1, size=100)

poly_reg = Pipeline([
    ("Poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])  # 传输数据沿着管道下去
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
plt.show()  # 图片2

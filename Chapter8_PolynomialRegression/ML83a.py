import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

# plt.scatter(x,y)
# plt.show()

"""使用MSE"""
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
print(mean_squared_error(y, y_predict))  # 使用线性拟合3.349730424119979


# 封装好的多项式回归
def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])


poly2_reg = PolynomialRegression(2)
poly2_reg.fit(X, y)
y2_predict = poly2_reg.predict(X)
print(mean_squared_error(y, y2_predict))  # 使用多项式拟合1.1977043618189978

poly10_reg = PolynomialRegression(10)
poly10_reg.fit(X, y)
y10_predict = poly10_reg.predict(X)
print(mean_squared_error(y, y10_predict))

plt.scatter(x, y)
plt.plot(np.sort(x), y10_predict[np.argsort(x)], color='r')
plt.show()  # 图片1

poly100_reg = PolynomialRegression(100)
poly100_reg.fit(X, y)
y100_predict = poly100_reg.predict(X)
print(mean_squared_error(y, y100_predict))

plt.scatter(x, y)
plt.plot(np.sort(x), y100_predict[np.argsort(x)], color='r')
plt.show()  # 图片2

"""这个是上面例子更好看的样子"""
X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = poly100_reg.predict(X_plot)
plt.scatter(x, y)
plt.plot(X_plot[:, 0], y_plot, color='r')
plt.axis([-3, 3, 0, 10])
plt.show()  # 图片3
print('=============================')
"""分级的意义"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_predict = lin_reg.predict(X_test)
print(mean_squared_error(X_test,y_test))  # 18.313999446781107

# 多项式回归 泛化能力好于线性回归
poly2_reg = PolynomialRegression(2)
poly2_reg.fit(X_train, y_train)
y2_predict = poly2_reg.predict(X_test)
print(mean_squared_error(y_test,y_predict))  # 3.085138047148842

# 开始过拟合
poly10_reg.fit(X_train, y_train)
y10_predict = poly10_reg.predict(X_test)
print(mean_squared_error(y_test, y10_predict))  # 1.0453028528367316

# 过拟合过头了面对新数据不行了
poly100_reg.fit(X_train, y_train)
y100_predict = poly100_reg.predict(X_test)
print(mean_squared_error(y_test, y100_predict))  # 1.0495817443521038e+19
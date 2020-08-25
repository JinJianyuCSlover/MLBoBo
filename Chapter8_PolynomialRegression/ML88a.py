import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()

"""之前见过的使用train-test-split"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])


from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.metrics import mean_squared_error

poly_reg = PolynomialRegression(degree=20)
poly_reg.fit(X_train, y_train)

y_poly_predict = poly_reg.predict(X_test)
print(mean_squared_error(y_test, y_poly_predict))  # 167.9401086297235明显过拟合

X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = poly_reg.predict(X_plot)

plt.scatter(x, y)
plt.plot(X_plot[:,0], y_plot, color='r')
plt.axis([-3, 3, 0, 6])
plt.show() # 图1

"""封装一下画图的方法"""
def plot_model(model):
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)

    plt.scatter(x, y)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.axis([-3, 3, 0, 6])
    plt.show()

plot_model(poly_reg)

"""岭回归不实现底层了"""

from sklearn.linear_model import Ridge

# ridge = Ridge(alpha=1)  # 就是之前式子里面的阿尔法
def RidgeRegression(degree,alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])

ridge1_reg = RidgeRegression(20,0.0001)  # alpha取一个比较小的
ridge1_reg.fit(X_train,y_train)
y1_predict = ridge1_reg.predict(X_test)
print(mean_squared_error(y_test,y1_predict))  # 1.3233492754143998比之前小了很多
plot_model(ridge1_reg) # 图2

ridge2_reg = RidgeRegression(20,1)
ridge2_reg.fit(X_train,y_train)
y2_predict = ridge1_reg.predict(X_test)
print(mean_squared_error(y_test,y2_predict))  # 1.3233492754143998
plot_model(ridge2_reg) # 图3

# alpha太大了
ridge3_reg = RidgeRegression(20, 100)
ridge3_reg.fit(X_train, y_train)

y3_predict = ridge3_reg.predict(X_test)
print(mean_squared_error(y_test, y3_predict))  # 1.3196456113086197
plot_model(ridge3_reg)

ridge4_reg = RidgeRegression(20, 10000000)  # 太大了，θ只能为0
ridge4_reg.fit(X_train, y_train)

y4_predict = ridge4_reg.predict(X_test)
print(mean_squared_error(y_test, y4_predict))  # 1.8408455590998372即使这样也比不优化好
plot_model(ridge4_reg)  # 图5
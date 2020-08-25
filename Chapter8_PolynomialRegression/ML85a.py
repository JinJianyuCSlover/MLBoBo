import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

np.random.seed(666)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()

from sklearn.model_selection import train_test_split

# 测试数据集大概取25%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# 绘制学习曲线
train_score = []
test_score = []
for i in range(1, 76):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train[:i], y_train[:i])
    # 模型随着放入数据越来越多性能的变化
    y_train_predict = lin_reg.predict(X_train[:i])
    train_score.append(mean_squared_error(y_train[:i], y_train_predict))

    y_test_predict = lin_reg.predict(X_test)
    test_score.append(mean_squared_error(y_test, y_test_predict))

plt.plot([i for i in range(1, 76)], np.sqrt(train_score), label="train")
plt.plot([i for i in range(1, 76)], np.sqrt(test_score), label="test")
plt.legend()
plt.show()  # 图1

"""直接定义成一个函数"""


def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(test_score), label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 4])
    plt.show()


plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)  # 图2

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

poly2_reg = PolynomialRegression(2)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)  # 图3

poly20_reg = PolynomialRegression(20)
plot_learning_curve(poly20_reg, X_train, X_test, y_train, y_test)  # 图4
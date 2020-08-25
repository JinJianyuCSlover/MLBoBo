import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * (x ** 2) + x + 2 + np.random.normal(0, 1, size=100)

print(x)
print(y)
print(X.shape)
# plt.scatter(X, y)
# plt.show()    # 图片1

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
# plt.scatter(X,y)
# plt.plot(X,y_predict,color='red')
# plt.show()  # 图片2

X2 = np.hstack([X, X ** 2])  # 这两个合并在一起
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
# plt.scatter(x, y)
# plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='red')
# plt.show()  # 图片3


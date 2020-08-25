from functions.SimpleLinearRegression import SimpleLinearRegression2
import numpy as np
import matplotlib.pyplot as plt

reg2 = SimpleLinearRegression2()
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_predict=6

reg2.fit(x, y)
print(reg2.a_)
print(reg2.b_)

y_hat = reg2.a_ * x + reg2.b_
plt.scatter(x,y)
plt.plot(x,y_hat,color='red')
plt.axis([0,6,0,6])
plt.show()

y_predict=reg2.predict(np.array([x_predict]))#一定要加[]不然就不是1维了
print(y_predict)


from functions.SimpleLinearRegression import SimpleLinearRegression1
import numpy as np
import matplotlib.pyplot as plt

reg1 = SimpleLinearRegression1()
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_predict=6
reg1.fit(x,y)
print(reg1.predict(np.array([x_predict])))
print(reg1.a_)
print(reg1.b_)

y_hat = reg1.predict(x)
plt.scatter(x,y)
plt.plot(x,y_hat,color='red')
plt.axis([0,6,0,6])
plt.show()

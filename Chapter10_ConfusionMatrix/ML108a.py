import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=666)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()  # 默认OvR多分类
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
y_predict = log_reg.predict(X_test)

from sklearn.metrics import precision_score

precision_score(y_test,y_predict,average="micro") # 针对多分类的精准率

"""混淆矩阵天然支持多分类问题"""
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict) # 会有一个10*10的矩阵，ij是真值i预测值为j的样本数量对角线就是正确的

cfm = confusion_matrix(y_test, y_predict)
plt.matshow(cfm, cmap=plt.cm.gray) # matrix show绘制一个矩阵，cmap是绘制的颜色对应，gray是灰度
plt.show()

row_sums = np.sum(cfm,axis=1) # 列的方向上求和得到每一行的和
err_matrix = cfm / row_sums
np.fill_diagonal(err_matrix,0) # 对角线数字填成0只保留错误
print(err_matrix) # 看看每一行犯错误的百分比
plt.matshow(err_matrix, cmap=plt.cm.gray)
plt.show()
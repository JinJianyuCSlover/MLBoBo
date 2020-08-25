import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
decision_scores = log_reg.decision_function(X_test)

precisions = []
recalls = []
thresholds = np.arange(np.min(decision_scores),np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores>=threshold, dtype='int')
    precisions.append(precision_score(y_test,y_predict))
    recalls.append(recall_score(y_test,y_predict))

plt.plot(thresholds, precisions)
plt.plot(thresholds, recalls)
plt.show() # 图1

plt.plot(precisions,recalls)
plt.show() # 图2

"""sklearn中的精准-召回曲线"""
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_test,decision_scores)  # 自动返回之前我们求的三个向量

print(precisions.shape) # (151,)
print(recalls.shape) # (151,)
print(thresholds.shape) # (150,)为什么少一个？定义的时候就是精准率=1，召回=0的时候不定义

plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.show() # 图3

plt.plot(precisions,recalls)
plt.show() # 图4
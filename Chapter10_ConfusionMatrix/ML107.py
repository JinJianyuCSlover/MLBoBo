import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
decision_scores = log_reg.decision_function(X_test)

from functions.metrics import FPR,TPR
thresholds = np.arange(np.min(decision_scores),np.max(decision_scores),0.1)
fprs=[]
tprs=[]
for threshold in thresholds:
    y_predict=np.array(decision_scores >= threshold,dtype='int')
    fprs.append(FPR(y_test,y_predict))
    tprs.append(TPR(y_test,y_predict))


plt.plot(fprs,tprs)
plt.show() # 图1

"""使用分装好的"""
from functions.metrics import FPR, TPR

fprs = []
tprs = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype='int')
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))
plt.plot(fprs, tprs)
plt.show()  # 图2 sklearn的ROC

"""关注曲线面积大小"""

from sklearn.metrics import roc_curve
fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, decision_scores)) # 0.9823319615912208总面积就是1
import numpy as np
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        # 分母=0
        return 0.0
precision = 0.5
recall = 0.5
print(f1_score(precision, recall))
precision = 0.1
recall = 0.9
print(f1_score(precision, recall))
precision = 0.0
recall = 1.0
print(f1_score(precision, recall))
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg.score(X_test, y_test))
y_predict = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)

from sklearn.metrics import precision_score
precision_score(y_test, y_predict)

from sklearn.metrics import recall_score
recall_score(y_test, y_predict)

from sklearn.metrics import f1_score
f1_score(y_test, y_predict)
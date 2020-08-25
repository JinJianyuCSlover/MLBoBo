import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

#寻找最好的k
best_score=0.0#起始值
best_k=-1#起始值
best_p=-1#闵可夫斯基距离

for k in range(1, 11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(n_neighbors=k,weights='distance')
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if (score > best_score):
            best_k = k
            best_score = score
            best_p=p
print('best_k= '+str(best_k))
print('best_score= '+str(best_score))
print('best_p= ' +str(best_p))

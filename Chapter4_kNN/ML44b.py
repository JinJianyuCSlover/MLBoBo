from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits =datasets.load_digits()
print(digits.keys())
X=digits.data
y=digits.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train,y_train)
y_predict = knn_clf.predict(X_test)
print(accuracy_score(y_test,y_predict))
print(knn_clf.score(X_test,y_test))


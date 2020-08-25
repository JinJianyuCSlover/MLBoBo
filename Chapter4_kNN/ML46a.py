from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV#这就是网格搜索的所在包
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

param_grid=[{
    "weights":['uniform'],
    "n_neighbors":[i for i in range(1,11)]
},
    {
        "weights":['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf,param_grid,n_jobs=-1,verbose=2)#定义好网格搜索对象
grid_search.fit(X_train,y_train)

#开始看看最佳取值
print(grid_search.best_estimator_)
#看看最佳匹配度
print(grid_search.best_score_)#可能还不如之前挑的，但是机器学习是一个复杂的评判过程
#看看最佳参数
print(grid_search.best_params_)

#有了最佳参数就跑最佳参数
knn_clf=grid_search.best_estimator_
knn_clf.score(X_test,y_test)

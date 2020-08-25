from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
X=iris.data#特征矩阵
y=iris.target#结果标签向量

#下面最后一个参数就是随机种子
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=666)
print(X_train.shape)
print(y_test.shape)
import numpy as np

def train_test_split(X,y,test_ratio=0.2,seed=None):
    '''
    :param X: 样本空间特征矩阵X
    :param y: 样本空间的标记值y
    :param test_ratio: 测试样例占总数的比例
    :param seed: 两次随机是否得到一样的数
    :return: 按照测试率将X，y分割成训练集和测试集
    '''
    assert 0<=test_ratio<=1 ,\
        "test ratio must be a value between 0 and 1."
    assert X.shape[0]==y.shape[0],\
        "the size of X must be equal with the size of y."
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))

    test_size=int(test_ratio*len(X))
    train_indexes = shuffle_indexes[test_size:]
    test_indexes = shuffle_indexes[:test_size]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]
    #这里藏着句返回我没看到
    return X_train, X_test, y_train, y_test
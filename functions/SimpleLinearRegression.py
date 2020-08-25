import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        '''初始化SimpleLinearRegression对象'''
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据寻来你数据集x_train,y_train训练简单线性回归模型'''
        assert x_train.ndim == 1, \
            "Simple linear regression has only 1 dimension to contain x_train"
        assert len(x_train) == len(y_train), \
            "the length of x_train must be equal with y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        a_frac_up = 0.0
        a_frac_down = 0.0

        for x_i, y_i in zip(x_train, y_train):
            a_frac_up += (x_i - x_mean) * (y_i - y_mean)
            a_frac_down += (x_i - x_mean) ** 2
        self.a_ = a_frac_up / a_frac_down
        self.b_ = y_mean - (self.a_ * x_mean)

        return self

    def predict(self, x_predict):
        '''给定带预测数据集x_predict，说白了就是装了好几个待预测x的向量。返回结果向量'''
        assert x_predict.ndim == 1, \
            "只解决一维的"
        assert self.a_ is not None and self.b_ is not None, \
            "必须先去fit才可以"
        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        '''这个私有方法是针对单个值，上面是针对一组值x'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "简单线性回归1号"

class SimpleLinearRegression2:
    def __init__(self):
        '''初始化SimpleLinearRegression对象'''
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据寻来你数据集x_train,y_train训练简单线性回归模型'''
        assert x_train.ndim == 1, \
            "Simple linear regression has only 1 dimension to contain x_train"
        assert len(x_train) == len(y_train), \
            "the length of x_train must be equal with y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        '''这里不用for而是用向量化'''
        a_frac_up=(x_train-x_mean).dot(y_train-y_mean)#一定要用dot点乘出来的才是数字，而*出来的还是向量
        a_frac_down=(x_train-x_mean).dot(x_train-x_mean)

        self.a_ = a_frac_up / a_frac_down
        self.b_ = y_mean - (self.a_ * x_mean)

        return self

    def predict(self, x_predict):
        '''给定带预测数据集x_predict，说白了就是装了好几个待预测x的向量。返回结果向量'''
        assert x_predict.ndim == 1, \
            "只解决一维的"
        assert self.a_ is not None and self.b_ is not None, \
            "必须先去fit才可以"
        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        '''这个私有方法是针对单个值，上面是针对一组值x'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "简单线性回归2号"

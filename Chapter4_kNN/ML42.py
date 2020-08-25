import numpy as np
import functions.kNN

raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]#特征空间
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]#标签0良性1恶性
X_train=np.array(raw_data_X)#矩阵
y_train=np.array(raw_data_y)#向量
x = np.array([8.093607318, 3.365731514])#新来的没分类的样本

kNN_clf = functions.kNN.kNNClassifier(k=6)
kNN_clf.fit(X_train,y_train)
y_predict = kNN_clf.predict(x)
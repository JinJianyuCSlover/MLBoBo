import numpy as np
import matplotlib.pyplot as plt
x = np.random.randint(0,100,size=100)#0-100取100个
#一个向量x减去一个最小值=>向量里面每个数都减去最小值。最值归一化成功
nor_x=(x-np.min(x))/(np.max(x)-np.min(x))

X=np.random.randint(0,100,(50,2))#矩阵50*2
X=np.array(X,dtype=float)#都变成浮点数方便归一化
#先对列进行操作
X[:,0]=(X[:,0]-np.min(X[:,0]))/(np.max(X[:,0])-np.min(X[:,0]))
X[:,1]=(X[:,1]-np.min(X[:,1]))/(np.max(X[:,1])-np.min(X[:,1]))
print(X[:10,:])
print(np.mean(X[:,0]))
print(np.mean(X[:,1]))
plt.scatter(X[:,0],X[:,1])
plt.show()
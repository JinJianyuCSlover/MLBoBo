import numpy as np
A=np.arange(16).reshape([4,4])
#默认在行的维度上分割成几列
upper,lower=np.vsplit(A,[2])
left,right=np.hsplit(A,[2])
print(left)
print('---------')
print(right)
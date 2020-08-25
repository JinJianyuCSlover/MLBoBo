import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -p*np.log(p) - (1-p)*np.log(1-p)

x = np.linspace(0.01,0.99,200)  #0和1不能取，不然会出错，1-1=0所以不取1

plt.plot(x,entropy(x))
plt.show()
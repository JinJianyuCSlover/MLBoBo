import math
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-2*(math.pi),2*(math.pi),1000)
y=[]
for i in x:
    y.append(math.cos(i))
plt.plot(x,y,color='red')
plt.show()
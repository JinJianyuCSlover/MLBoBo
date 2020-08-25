import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1,6,141)#绘制的x点均匀取值
plot_y=(plot_x-2.5)**2-1#模拟损失函数
#绘制图像
# plt.plot(plot_x,plot_y)
# plt.show()

def dJ(theta):
    """导数"""
    return 2*(theta)-5

def J(theta):
    """损失函数"""
    try:
        return (theta-2.5)**2-1
    except:
        return float('inf')#异常处理，防止过大的值

"""封装梯度下降"""
initial_theta = 0.0
def gradient_descent(initial_theta,eta,epsilon=1e-8,n_iterations=1e4):
    theta=initial_theta
    theta_history.append(theta)
    i_iter=0

    while i_iter<n_iterations:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - (gradient * eta)
        theta_history.append(theta)
        if (abs(J(theta) - J(last_theta)) < epsilon):
            break
        i_iter+=1

def plot_theta_history():
    plt.plot(plot_x,J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()

eta=1.1
theta_history=[]
gradient_descent(0.0,eta,n_iterations=10)
plot_theta_history()
print(len(theta_history))
print(theta_history[-1])
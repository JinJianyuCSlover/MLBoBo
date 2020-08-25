import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

digits = datasets.load_digits()
X=digits.data
y=digits.target

noisy_digits = X + np.random.normal(0,4,size=X.shape)
example_digits = noisy_digits[y==0,:][:10]
for num in range(1,10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits,X_num])

# 绘制数字的代码（无需掌握）
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))

    plt.show()

plot_digits(example_digits) # 噪音更大了

"""开始PCA"""
pca = PCA(0.5)
pca.fit(noisy_digits)
print(pca.n_components_)#12个维度
components = pca.transform(example_digits)
filter_digits = pca.inverse_transform(components)
# plot_digits(filter_digits)



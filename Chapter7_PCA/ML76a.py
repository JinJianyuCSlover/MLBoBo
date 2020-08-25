import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=1)
pca.fit(X)

print(pca.components_)

X_reduction = pca.transform(X)
X_restore = pca.inverse_transform(X_reduction)

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()# 图片1





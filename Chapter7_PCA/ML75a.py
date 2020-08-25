import numpy as np
import matplotlib.pyplot as plt
from functions.PCA import PCA

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)

pca = PCA(n_components=1)
pca.fit(X)

X_reduction = pca.transform(X)
print(X_reduction.shape)  # (100,1)

X_restore = pca.inverse_transform(X_reduction)
print(X_restore.shape)  # (100, 2)

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()# 图片1

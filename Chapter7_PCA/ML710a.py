from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

faces = fetch_lfw_people()
print(faces)

"""对人脸进行随机排列"""

random_indexes = np.random.permutation(len(faces.data))
X=faces.data[random_indexes]

example_faces = X[:36,:]
def plot_faces(faces):

    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()

plot_faces(example_faces)# 图片1

"""特征脸"""
pca=PCA(svd_solver='randomized')
pca.fit(X)
plot_faces(pca.components_[:36,:]) #图片2

"""更多问题"""
# faces2 = fetch_lfw_people(min_faces_per_person=60) #有的人有好几张图片，有的人只有1张
# faces2.data.shape    # (1348, 2914)
# len(faces2.target_names)    # 8根据一个人多张图片去检验可信度更高

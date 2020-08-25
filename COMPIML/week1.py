from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
data = mnist['data']
print(data.shape)
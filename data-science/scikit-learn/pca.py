import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
print("Reduced dataset shape:", X_reduced.shape)
print("dataset shape:", X.shape)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
            cmap='RdYlBu')
print("Meaning of the 2 components:")
for component in pca.components_:
    print(" + ".join("%.3f x %s" % (value, name)
          for value, name in zip(component, iris.feature_names)))
plt.show()

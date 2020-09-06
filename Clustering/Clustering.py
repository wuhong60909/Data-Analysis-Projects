import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets

# create datasets
X, y = datasets.make_blobs(n_samples = 50, centers = 3, n_features = 2, random_state = 20, cluster_std = 1.5)

# parameter setting
n = 3 # number of clusters (not essential)

# Agglomerative Clustering method
model = AgglomerativeClustering(n_clusters = n, linkage = 'ward')

# linkage: ['ward', 'complete', 'average']
model.fit(X)
labels = model.fit_predict(X)

# results visualization
plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels)
plt.axis('equal')
plt.title('Prediction')
plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage
# Performs hierarchical/agglomerative clustering on X by using "Ward's method"
linkage_matrix = linkage(X, 'ward')
figure = plt.figure(figsize = (7.5, 5))

# Plots the dendrogram
dendrogram(linkage_matrix, labels = labels)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
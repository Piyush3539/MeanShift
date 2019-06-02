# #############################################################################
# Importing packages

import pandas as pd                    # Importing pandas package for data analysis and statistics
import numpy as np                     # Importing numpy package for array processing for numbers, strings and objects
from sklearn.cluster import MeanShift  # Importing package for machine learning and data mining
import matplotlib.pyplot as plt        # Python plotting package

df = pd.read_excel('C:/Users/P_Verma/Desktop/Datasets.xls','DB_16')  # Assigning path of datasets to df variable
numpyMatrix = df.as_matrix()

# #############################################################################
# Compute clustering with MeanShift

ms = MeanShift()
ms.fit(numpyMatrix)                     # perform clustering
labels = ms.labels_                     # labeling each points
cluster_centers = ms.cluster_centers_   # computing clusters
print("The centers of the clusters are : \n", cluster_centers)    # print cluster center

n_clusters_ = len(np.unique(labels))    # Computing total clusters
print("The number of estimated clusters : ", n_clusters_)  # printing result on console screen

colors = 10*['r.', 'g.', 'c.', 'k.', 'y.', 'm.']    # assigning different colors of plotting the points

# #############################################################################
# plotting clusters

for i in range(len(numpyMatrix[:, 1])):         # looping for all the values of X0
    plt.plot((numpyMatrix[i, 0]), numpyMatrix[i, 1], colors[labels[i]], markersize=5)  # assigning different attributes to scatter plots

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="X", s=250)        # marking cluster with X sign
plt.show()            # scatter graph screen

# #############################################################################

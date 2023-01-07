# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 03:00:40 2021

@author: Kenneth
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


#Loading the data
cols=['x', 'y']
df = pd.read_csv('PizzaData.csv', names=cols, header=None)

#using KMeans
kmPizza = KMeans(n_clusters=4).fit(df)
centroids = kmPizza.cluster_centers_
print(centroids)

#Plot the results
plt.scatter(df['x'], df['y'], c = kmPizza.labels_.astype(float), s=50, alpha=.5)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=50)
plt.title('Pizza Data')
plt.show()


################################################################################

#Load
ring = pd.read_csv('RingData.csv', names=cols, header=None)

#KMeans
kmRing = KMeans(n_clusters=2).fit(ring)
centroids = kmRing.cluster_centers_
print(centroids)

#Plot
plt.scatter(ring['x'], ring['y'], c = kmRing.labels_.astype(float), s=50, alpha=.5)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=50)
plt.title('Ring Data')
plt.show()

#Transform
dist = np.sqrt(ring['x']**2 + ring['y']**2)
#plt.plot(dist, np.zeros_like(dist), 'bo')
#plt.show()
distFromOrigin = np.array([dist, np.zeros_like(dist)]).transpose()

#Plot with Transform
newRing = KMeans(n_clusters=2).fit(distFromOrigin)
centroids = newRing.cluster_centers_
plt.scatter(distFromOrigin[:,0], distFromOrigin[:,1], c = newRing.labels_.astype(float), s=50, alpha=.5)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=50)
plt.title('Transformed Ring Data')
###################################################################################
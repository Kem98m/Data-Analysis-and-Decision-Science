# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:55:55 2021

@author: Kenneth
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC



class1 = np.array([1.0, 1.0, .8, .8, .6, .6, .6, .6, .2, .2])
class0 = np.array([.8, .6, .4, .4, .4, .4, .0, .0, .0, .0])

True_pos = np.array([])



"Question 3"

X_0_coord1 = np.array([])
X_0_coord2 = np.array([])
X_0_target = np.array([])
X_1_coord1 = np.array([])
X_1_coord2 = np.array([])
X_1_target = np.array([])

## Creating the dataset for X

for i in range(100):
    X_0_coord1 = np.append(X_0_coord1, np.array([0]))
    X_0_coord2 = np.random.rand(100)
    X_0_target = np.append(X_0_target, np.array([0]))
    
    X_1_coord1 = np.append(X_1_coord1, np.array([5]))
    X_1_coord2 = np.random.rand(100)
    X_1_target = np.append(X_1_target, np.array([5]))
    

## Converting into a Dataframe to use sklearn to divide into folds

x_coord1 = np.append(X_0_coord1, X_1_coord1)
x_coord2 =np.append(X_0_coord2, X_1_coord2)
x_target_ = np.append(X_0_target, X_1_target)

dataset_x = pd.DataFrame({'Coord1': x_coord1, 'Coord2': x_coord2})
#dataset_x_1 = pd.DataFrame({'Coord1': X_1_coord1, 'Coord2': X_1_coord2})
targetset_x = pd.DataFrame({'Target': x_target_})
#targetset_x_1 = pd.DataFrame({'Target': X_1_target})



#dataset_x_0 = pd.DataFrame({'Coord1': X_0_coord1, 'Coord2': X_0_coord2})
#dataset_x_1 = pd.DataFrame({'Coord1': X_1_coord1, 'Coord2': X_1_coord2})
#targetset_x_0 = pd.DataFrame({'Target': X_0_target})
#targetset_x_1 = pd.DataFrame({'Target': X_1_target})

#print(targetset_x_0)
#print(targetset_x_1)

#x_sample = np.append(dataset_x_0, dataset_x_1)
#x_targets = np.append(targetset_x_0, targetset_x_1)
#x_sample_set = pd.DataFrame({'Coord1': })



#print (dataset_x_0)
train_x, test_x, train_target_x, test_target_x = train_test_split(dataset_x, targetset_x, test_size=0.2, random_state=42, shuffle=True)
#train_x_1, test_x_1, train_target_x_1, test_target_x_1 = train_test_split(dataset_x_1, targetset_x_1, test_size=0.2, random_state=42, shuffle=True)


plt.figure(2)
plt.title('Plot of Dataset X (The union of X_0 and X_1)')
plt.scatter(X_0_coord1, X_0_coord2,  color=(1, .5, 0, 1))
plt.scatter(X_1_coord1, X_1_coord2, color=(0, 0, 1, 1))

plt.figure(3)
plt.title('Plot of training (X) and test (+) set')
plt.scatter(train_x['Coord1'], train_x['Coord2'], marker="x", color=(1, .5, 0, 1))
#plt.scatter(train_x_1['Coord1'], train_x_1['Coord2'], marker="x", color=(0, 0, 1, 1))

plt.scatter(test_x['Coord1'], test_x['Coord2'], marker="+", color=(1, .5, 0, 1))





#plt.scatter(test_x_1['Coord1'], test_x_1['Coord2'], marker="+", color=(0, 0, 1, 1))

#totX = np.append(train_x_0, train_x_1)
#totTargetX =  np.append(train_target_x_0, train_target_x_1)
#totTestX = np.append(test_x_0, test_x_1)
#totTargetTestX =  np.append(test_target_x_0, test_target_x_1)

#labels = np.array([])
## Using 1-NN
#labels

#for i in range(len(totX)):
#    print(totX[2*i])
#    if (totX[i] == 0.0):
#        labels = np.append(labels, 0)
#    else:
#        labels = np.append(labels, 5)

neigh = KNeighborsClassifier(n_neighbors=1)
#print(totX.shape)
#print(totTargetX.shape)
#print(train_target_x_0)
y_pred = cross_val_predict(neigh, train_x, np.array(train_target_x).ravel(), cv=5)
print(y_pred)
#neigh.fit(train_x, np.array(train_target_x).ravel())
#predictedX = neigh.predict(train_x)
cm = confusion_matrix(train_target_x, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

print(confusion_matrix(train_target_x, y_pred))


y_0_coord1 = np.array([])
y_0_coord2 = np.array([])
y_1_coord1 = np.array([])
y_1_coord2 = np.array([])

for i in range(100):
    y_0_coord1 = np.random.rand(100)
    y_0_coord2 = np.random.rand(100)
    
    y_1_coord1 = np.random.rand(100)
    y_1_coord2 = np.random.rand(100)


plt.figure(4)
plt.title('Plot of Dataset Y (Samples from the unit square')
plt.scatter(y_0_coord1, y_0_coord2, color=(1, .5, 0, 1))
plt.scatter(y_1_coord1, y_1_coord2, color=(0, 0, 1, 1))

train_y_1, test_y_1, train_target_y_1, test_target_y_1 = train_test_split(dataset_x_1, targetset_x_1, test_size=0.2, random_state=42, shuffle=True)

plt.figure(5)
plt.title('Plot of training (X) and test (+) set')

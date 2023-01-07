# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:33:07 2021

@author: Kenneth
"""

import numpy as np
import matplotlib.pyplot as plt

rad = .5

dartsThrown = 5000

myPiValues = np.array([])

myTrials = np.array([])

for j in range(5):

    x_points = np.random.uniform(-rad, rad, dartsThrown)

    y_points = np.random.uniform(-rad, rad, dartsThrown)

    dartsInCirc = np.array([])

    dartData = np.array([])

    for i in range(len(x_points)):
        if (np.sqrt((x_points[i])**2 + (y_points[i])**2) <= rad):
            dartsInCirc = np.append(dartsInCirc, np.array([x_points[i], y_points[i]]))
            dartData = np.append(dartData, np.array([1]))
        else: dartData = np.append(dartData, np.array([0]))

    dartsTotal = len(x_points)
    cumulativeDarts = np.cumsum(dartData)
    
    consecutiveNums = np.array([])
    piRatio = np.array([])
    piArray = np.array([])
    
    for k in range(dartsTotal):
        consecutiveNums = np.append(consecutiveNums, np.array([k+1]))
        piArray = np.append(piArray, np.pi)
        piRatio = np.append(piRatio, 4 * cumulativeDarts[k]/consecutiveNums[k])
    
    fig = plt.figure()
    plt.title('Calculating PI:Trial %i ' %j + 'with up to %i darts' %dartsTotal)
    plt.plot(consecutiveNums, piRatio, 'ro')
    plt.xlabel('Number of darts thrown')
    plt.ylabel('Calculated Pi')
    plt.plot(consecutiveNums, piArray, 'b-')
    plt.show()

#print(dartsCircleTotal)
#print(dartsTotal)
#print(cumulativeDarts)
#print(myPI)
#print(dartData)
#print(cumulativeDarts)
print(piRatio)




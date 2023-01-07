# -*- coding: utf-8 -*-
"""
Created on Tue Feb  21 23:14:04 2021

@author: Kenneth
"""

import numpy as np
import random
import matplotlib.pyplot as plt

sample_group = 5000000
#group = random.sample(range(0, 1), sample_group)
cruise_rate = 1/20
normal_rate = 1/10000
true_positive_rate = .999
false_positive_rate = .01

positive_cruise = true_positive_rate*cruise_rate + false_positive_rate*(1-cruise_rate)
positive_regular = true_positive_rate*normal_rate + false_positive_rate*(1-normal_rate)

positive_cases = np.array([])
true_positive = np.array([])

# this for loop is to get the initial probability of getting VBS on the cruise
for i in range(sample_group):
    if (np.random.random_sample()<cruise_rate):
        positive_cases = np.append(positive_cases, np.array([1]))
        
    else:
        positive_cases = np.append(positive_cases, np.array([0]))
    numOfCases = np.sum(positive_cases)
    ratio = numOfCases/positive_cases.size
    true_positive = np.append(true_positive, ratio)
    average_prob = np.sum(true_positive)/true_positive.size
print("The probability of having the disease with a positive test from the cruise is ", ratio/positive_cruise)





# This is the for loop for the initial probability of getting VBS on a regular basis
positive_cases = np.array([])
true_positive = np.array([])

for i in range(sample_group):
    if (np.random.random_sample()<normal_rate):
        positive_cases = np.append(positive_cases, np.array([1]))
    else:
        positive_cases = np.append(positive_cases, np.array([0]))
    numOfCases = np.sum(positive_cases)
    ratio = numOfCases/positive_cases.size
    true_positive = np.append(true_positive, ratio)
    average_prob = np.sum(true_positive)/true_positive.size
print("The probability of having the disease with a positive test from normal exposure is ", ratio/positive_regular)
    
    
    
#print(positive_cases)
#print(ratio)
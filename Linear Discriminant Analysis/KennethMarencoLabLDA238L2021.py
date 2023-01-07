# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 04:11:09 2021

@author: Kenneth
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle


x0 = np.random.normal(0, 1, 500)
g0 = np.zeros(500)

x1 = np.random.normal(3, 1, 500)
g1 = np.ones(500)


plt.figure(1)
plt.title("Set of X0 (blue) and X1 (red)")
plt.scatter(x0, g0, c="b")
plt.scatter(x1, g1, c="r")


plt.figure(2)
plt.title("log-odds: Set of X0 (blue) and X1 (red)")

log0 = (3/1)*x0 +(np.log(1)-np.log(1)-((9-1)/2))
log1 = (3/1)*x1 +(np.log(1)-np.log(1)-((9-1)/2))

plt.plot(log0, x0, c="b")
plt.plot(log1, x1, c ="r")

jointx = np.append(x0, x1)
jointlog = np.append(log0, log1)


x0_train, x0_test, log0_train, log0_test = train_test_split(jointx, jointlog, test_size=.5, random_state=0)

sc = StandardScaler()
x0_train = sc.fit_transform(x0_train.reshape(1,-1))
x0_test = sc.transform(x0_test.reshape(1,-1))
                       
lda= LDA(n_components=1)

x0_train = lda.fit_transform(x0_train, log0_train)
x0_test = lda.transform(x0_test)

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(x0_train, log0_train)
log0_pred = classifier.predict(x0_test)

cm = confusion_matrix(log0_test, log0_pred)
print(cm)
print('Accuracy' + str(accuracy_score(log0_test, log0_pred)))

#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=0))


#log0_score = classifier.fit(x0_train, log0_train).decision_function(x0_test)
log0_score = np.array([])
threshold = -4.9
print(log0.size)
x0_score=np.array([])
for i in range(log0.size):
    if x0_test[i] < threshold:
        log0_score[i] = 0



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(log0.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(log0_test[:, i], log0_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(log0_test.ravel(), log0_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
plt.figure(3)
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.show()
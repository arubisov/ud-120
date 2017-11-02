#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train
# rows: data points
# columns: features
# print "number of features: {}".format(len(features_train[0]))


#########################################################
### your code goes here ###

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "training time: {}s".format(round(time() - t0, 3))

labels_pred = clf.predict(features_test)

# from sklearn.metrics import accuracy_score
# print accuracy_score(labels_test, labels_pred)
print clf.score(features_test, labels_test)

# for nth_pred in [10, 26, 50]:
#     print "prediction for {}th element: {}".format(nth_pred,
#                                                    labels_pred[nth_pred])


#########################################################

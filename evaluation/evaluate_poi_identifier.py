#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

# predict POIs within the test set only
labels_pred = clf.predict(features_test)
print "# of people in test set: {}".format(len(features_test))
print "# of POIs predicted: {}".format(sum(labels_pred))

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(labels_test, labels_pred)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print "# of true positives: {}".format(TP)
# precision: of all the ones you predicted to be of class X, what ratio was correct?
print "precision: {}".format(TP*1.0/(TP+FP))
# recall: of all the ones that ARE of class X, what ratio did you successfully predict?
print "recall: {}".format(TP*1.0/(TP+FN))

# from sklearn.metrics import accuracy_score
# print accuracy_score(labels_test, labels_pred)
# print clf.score(features_test, labels_test)

### Bullshit Udacity round:
print '######'
print 'The bullshit udacity round'
print '######'
y_pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
y_true = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

CM = confusion_matrix(y_true, y_pred)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print "precision: {}".format(TP*1.0/(TP+FP))
print "recall: {}".format(TP*1.0/(TP+FN))

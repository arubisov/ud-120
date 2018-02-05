#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import numpy as np
import pandas as pd
from scipy import stats

from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# conda install -c conda-forge imbalanced-learn
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, fbeta_score, make_scorer


def featureFormatDF(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []
    keys_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )
            keys_list.append(key)

    df = pd.DataFrame(np.array(return_list))
    df['names'] = keys_list
    df.set_index('names', drop=True, inplace=True, verify_integrity=True)
    df.columns = features

    return df

def run_logistic_regression_classifier(X_train, X_test, y_train, y_test, beta=1):
    print "{} ({:.2f}%) positive labels in training set".format(len(y_train[y_train == True]), 100.0*len(y_train[y_train == True])/len(y_train))
    print "{} ({:.2f}%) positive labels in test set".format(len(y_test[y_test == True]), 100.0*len(y_test[y_test == True])/len(y_test))

    t0 = time()
    param_grid = {
             'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
              'penalty': ['l1', 'l2'],
              }
    f5_scorer = make_scorer(fbeta_score, beta=beta)
    clf = GridSearchCV(estimator=LogisticRegression(class_weight='balanced'),
                       param_grid=param_grid,
                       scoring=f5_scorer)
    clf = clf.fit(X_train, y_train)
    print "training time: {}s".format(round(time() - t0, 3))
    print "Best estimator found by grid search:"
    print clf.best_estimator_

    y_pred = clf.predict(X_test)
    print "{} ({:.2f}%) positive labels in predictions".format(len(y_pred[y_pred == True]), 100.0*len(y_pred[y_pred == True])/len(y_pred))

    conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print conf
    TN, FP, FN, TP = conf.ravel()
    print "TN: {}, FP: {}, FN: {}, TP: {}".format(TN, FP, FN, TP)

    # accuracy: what proportion was correctly predicted out of total population?
    print "accuracy: {}".format((TP+TN)*1.0/(TN+FP+FN+TP))
    # precision: of all the ones you predicted to be of class X, what ratio was correct?
    print "precision: {}".format(TP*1.0/(TP+FP))
    # recall: of all the ones that ARE of class X, what ratio did you successfully predict?
    print "recall: {}".format(TP*1.0/(TP+FN))

    return clf.best_estimator_


def run_decision_tree_classifier(X_train, X_test, y_train, y_test, beta=5):
    print "{} ({:.2f}%) positive labels in training set".format(len(y_train[y_train == True]), 100.0*len(y_train[y_train == True])/len(y_train))
    print "{} ({:.2f}%) positive labels in test set".format(len(y_test[y_test == True]), 100.0*len(y_test[y_test == True])/len(y_test))

    t0 = time()
    param_grid = {
             'min_samples_split': [2,5,10,15,20],
             'max_depth': range(1,5,1),
              }
    f5_scorer = make_scorer(fbeta_score, beta=beta)
    clf = GridSearchCV(estimator=DecisionTreeClassifier(class_weight='balanced'),
                       param_grid=param_grid,
                       scoring=f5_scorer)
    clf = clf.fit(X_train, y_train)
    print "training time: {}s".format(round(time() - t0, 3))
    print "Best estimator found by grid search:"
    print clf.best_estimator_

    y_pred = clf.predict(X_test)
    print "{} ({:.2f}%) positive labels in predictions".format(len(y_pred[y_pred == True]), 100.0*len(y_pred[y_pred == True])/len(y_pred))

    conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print conf
    TN, FP, FN, TP = conf.ravel()
    print "TN: {}, FP: {}, FN: {}, TP: {}".format(TN, FP, FN, TP)

    # accuracy: what proportion was correctly predicted out of total population?
    print "accuracy: {}".format((TP+TN)*1.0/(TN+FP+FN+TP))
    # precision: of all the ones you predicted to be of class X, what ratio was correct?
    print "precision: {}".format(TP*1.0/(TP+FP))
    # recall: of all the ones that ARE of class X, what ratio did you successfully predict?
    print "recall: {}".format(TP*1.0/(TP+FN))

    return clf.best_estimator_


def plot_roc(clf, X_test, y_test, X_train=None, y_train=None):
    y_score = clf.decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='Test ROC curve (area = %0.2f)' % roc_auc)

    if X_train is not None and y_train is not None:
        y_score_train = clf.decision_function(X_train)

        # Compute ROC curve and ROC area for each class
        fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
        roc_auc_train = auc(fpr, tpr)

        plt.plot(fpr_train, tpr_train, color='aqua',
                 lw=lw, label='Train ROC curve (area = %0.2f)' % roc_auc_train)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Logistic Regression Model')
    plt.legend(loc="lower right")
    plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 3: Create new feature(s)
for k,v in data_dict.items():
    if v['bonus'] != 'NaN' and v['bonus'] != 'NaN':
        data_dict[k]['bonus_to_salary_ratio'] = 1.0 * v['bonus'] / v['salary']
    else:
        data_dict[k]['bonus_to_salary_ratio'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Extract features and labels from dataset for local testing
features_list = ['poi'] + ['bonus_to_salary_ratio'] + financial_features + email_features
df = featureFormatDF(my_dataset, features_list, sort_keys = True)

### Task 2: Remove outliers
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
X = df[features_list[1:]]
y = df[features_list[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = run_logistic_regression_classifier(X_train, X_test, y_train, y_test)
# plot_roc(clf, X_test, y_test, X_train, y_train)

dt = run_decision_tree_classifier(X_train, X_test, y_train, y_test)
# test_classifier(dt, my_dict, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
feature_weights = (df[features_list[1:]].std().as_matrix() * clf.coef_)[0]
for idx, value in enumerate(feature_weights):
    if abs(value) > 10:
        print "Feature[{}] {}: {:.2f}".format(idx, features_list[idx], value)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

my_dataset = df.to_dict(orient="index")

test_classifier(dt, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)

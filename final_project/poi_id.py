#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from helper import featureFormatDF, run_logistic_regression_classifier, run_decision_tree_classifier, plot_roc


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
df = df.drop(['restricted_stock_deferred', 'director_fees', 'loan_advances'], axis=1)
df = df.drop(['bonus_to_salary_ratio'], axis=1)
features_list = list(df.columns)

### Task 2: Remove outliers
df = df.drop(['TOTAL'], axis=0)

# df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
X = df[df.columns[1:]]
y = df[df.columns[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = run_logistic_regression_classifier(X_train, X_test, y_train, y_test)
# plot_roc(clf, X_test, y_test, X_train, y_train)

# dt = run_decision_tree_classifier(X_train, X_test, y_train, y_test)
# test_classifier(dt, my_dict, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
feature_indices = [1 + idx for idx in list(clf.named_steps['kbest'].get_support(indices=True))]

print "Columns chosen by SelectKBest:"
print ", ".join(df.columns[feature_indices])

print "Top features:"
feature_weights = (df.iloc[:, feature_indices].std().as_matrix() * clf.named_steps['lr'].coef_)[0]
for idx, value in enumerate(feature_weights):
    if abs(value) > 10:
        print "Feature[{}] {}: {:.2f}".format(idx, df.columns[idx+1], value)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

my_dataset = df.to_dict(orient="index")

test_classifier(clf, my_dataset, df.columns)
# test_classifier(dt, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)

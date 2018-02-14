from time import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# conda install -c conda-forge imbalanced-learn
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


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

    pipeline = Pipeline([('kbest', SelectKBest(f_classif)),
                         ('lr', LogisticRegression(class_weight='balanced'))])

    param_grid = {
                  'kbest__k': range(15, X_train.shape[1] + 1),
                  'lr__C': np.logspace(-10, 10, 5),
                  'lr__penalty': ['l1', 'l2'],
                 }

    clf = GridSearchCV(estimator=pipeline,
                       param_grid=param_grid,
                       scoring=make_scorer(fbeta_score, beta=beta))
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
        roc_auc_train = auc(fpr_train, tpr_train)

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

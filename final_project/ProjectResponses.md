# Identify Fraud from Enron E-mail
2018.02.04  
Anton Rubisov

### Requirements

Up-to-date Python 2.7 with Anaconda distribution

### Included in .zip Folder
- my_dataset.pkl
- my_classifier.pkl
- my_feature_list.pkl
- poi_id.py
- this markdown file

### Question Responses
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

  Our goal is to predict whether a given person in the Enron fraud scandal is a person-of-interest (POI). Machine learning is useful toward achieving this goal because it so happens that the Enron financial and e-mail dataset was made available publicly, and our hypothesis is that the data included within may have sufficient predictive power to produce a machine learning classifier, trained on hand-crafted labels, which is quicker to develop and delivers better results than what we can do by manually do by combing through the data - namely someone like myself with no education in forensic accounting!

  The dataset contains 146 persons, of whom 18 are POIs, and 21 features. The dataset is incomplete, as there are many missing values (represented as 'NaN'), and many values that are zero where the value is unlikely to be zero in reality, such as a salary. Of the 3066 total values in the dataset, 160 are zeros, and 1358 are NaNs. In my analysis, all NaNs are converted to zeros, and any person for whom all my values of interest are equal to zero is removed from the dataset. Otherwise, all other zero values are retained.

1. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

  I began by including every numeric feature that was available in the financial features and e-mail features - which was everything except for the person's e-mail address. I didn't think it made sense to do feature engineering or outlier detection with the data in dictionary format, so the first thing I did was rewrite the featureFormat() function to return a pandas DataFrame, with the features as column names, and the person's name as the index. I was pleasantly surprised to find that it was a very small dataset - only 145 persons, and 20 features. This meant I would have very short run times for classifier training, and would be able to use GridSearchCV without worrying about that runtime being any more than a few seconds. This also easily allowed me to check for missing values in the dataset: some of the features had too many missing values and were removed. Particularly, restricted_stock_deferred, director_fees, and loan_advances had 127, 128, and 141 missing values, and were left out of the dataset.

  I knew I would be trying logistic regression and decision trees as starting algorithms, choosing both for their explanatory power, so I was not concerned about feature scaling since both algorithms are insensitive to feature scale. Outliers, however, were an issue. From the coursework itself, I already knew that Kenneth Lay had an astronomically large salary and bonus, so I created a new feature equal to bonus divided by salary, thinking that this might be a better indicator than either feature alone. To assess the impact of the new feature, I trained the same logistic regression classifier with this new feature included and excluded. The performance of the former degraded across every evaluated metric (accuracy, precision, recall, F1, F2) compared with the latter - from as little as .01 for accuracy to as much as .09 for recall - so I chose not to use this new feature in my final dataset.

  Continuing on the outlier exploration, I plotted a boxplot of all the features. Rather unsurprisingly, total_stock_value and exercised_stock_options had the widest tails - the company executives were at the upper extremes of both distributions.The boxplot revealed a major outlier in the salary feature, and checking which data points had `salary > 1e7` revealed the presence of a data point called `TOTAL`, which was the sum of all other data points. Obviously being a case of messy data, this data point was removed. However, other data points at the tail of the boxplot distributions were often POIs, such as Kenneth Lay. Because there were only 18 of these data points to begin with, removing them as outliers was not permissible. Thus, I made the decision to run the analysis with all other "outliers" retained - since, in fact, Kenneth Lay *did* earn that salary and bonus.

  20 features is not high-dimensional data, so I felt dimensionality-reduction via feature selection was not necessary. That said, to get some experience with it, I implemented RFECV in my pipeline, and was surprised to find that it only contained e-mail-based features in its support. I revised my initial dataset to exclude all financial data, and found my classifier's performance relatively unchanged. This was explained by the fact that the tester code function, `test_classifier()`, uses the `featureFormat()` function with its default parameters, specifically `remove_all_zeroes=False`. That setting causes `featureFormat()` to drop any row that contains all zeros _excluding_ the POI label, as a result of which 4 POIs get dropped. Thus, the classifier performance on the remaining 14 POIs is indeed high, but 4 other POIs aren't being predicted at all. This is a definite flaw in the tester code, which is inherently relying on no POIs being dropped by virtue of the list of features selected.

  Knowing that this approach is not correct, but that I wouldn't be able to change the tester.py code that the automatic grading uses for the submission (although I could of course change it locally), I reverted to my initial assessment that 20 features is not high-dimensional, and chose to retain all numeric features save the aforementioned three with high numbers of missing values.

  I found my most important features to be:
  Feature[5] total_stock_value: 12.37
  Feature[13] from_messages: -15.20

1. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

  I wanted to try both a decision tree and logistic regression, as both offer the ability to understand feature importance. With the same underlying set of data, logistic regression performed substantially better than a decision tree, beating it on every single evaluation metric:  

Metric | Logistic Regression | Decision Tree
  ------ | ------------------- | -------------
  Accuracy | 0.82457 |  0.71800
  Precision | 0.41926 | 0.23547
  Recall | 0.59200 | 0.43350
  F1 | 0.49088 | 0.30517
  F2 | 0.54693 | 0.37108
  TP | 1184 | 867
  FP | 1640 | 2815
  FN | 816 | 1133
  TN | 10360 | 9185

  Notably, the logistic regression classifier gave me an area under the curve of 0.92 for the train and test set, so I was confident I had a decent classifier.

3. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

  Tuning an algorithm refers to the practice of changing model parameters - such as, for example, the maximum depth of a decision tree - in order to maximize the performance, relative to a chosen scoring metric, of the algorithm on a given dataset. Tuning parameters is important because classifiers behave very differently based on the parameters used to instantiate them. I used GridSearchCV to iterate through values of C ranging from 10 to 1,000,000 by orders of 10, and penalty as either L1 or L2 for logistic regression, as well as min_samples_split (in [2,5,10,15,20]) and max_depth (in [1,2,3,4]) for decision tree.

1. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

  A classic mistake with machine learning is training and evaluating a model on the same dataset. It's a mistake because it runs the risk of overfitting to the training data, and not generalizing to new, unseen data. Validation refers to evaluating a trained classifier against a set of data which was not used for training. I achieved this rather simply by creating a train/test split of my data, using 70% for training and reserving 30% for training. Because the target classes are imbalanced (the POI class is only 12% of the population), when training my classifier, I used `class_weight='balanced'` to force resampling of the classes in order to increase the number of positive labels the classifier sees.  As part of my tuning of the machine learning pipeline, I used the tester.py code, which used 1000 cross-validation folds. Specifically, it uses `StratifiedShuffleSplit`, which aims to maintain the proportions of the two classes in each cross-validation fold. Without stratification, and because our target class is so small, a standard shuffle split would run the risk of having no POIs in it!

1. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

  I specifically tracked accuracy, precision, and recall while running my analysis. Here are my interpretations of those metrics in the context of this problem:

  **Accuracy**: Of all the persons, what percentage am I labelling correctly?

  **Precision**: Of all the persons I predict to be POIs, what percentage are actually POIs?

  **Recall**: Of all the persons who actually are POIs, what percentage did I successfully predict as being POIs?


### References
-  https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-dataframe
- https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model


### Solemn Attestation / Declaration of Independence
I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
features_list = ['poi','salary','exercised_stock_options','total_stock_value','bonus', 'transformed_salary', 'composite_poi_email_data'] # You will need to use more features
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
def process_features():



    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    print data_dict
    for key in data_dict:
        for feature in data_dict[key]:
            if data_dict[key][feature] == 'NaN':
                data_dict[key][feature] = 0

    ### Task 2: Remove outliers
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    #AC - removed the "TOTAL"
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    from sklearn.preprocessing import MinMaxScaler

    df = pd.DataFrame.from_dict(data_dict, orient='index')

    scaler = MinMaxScaler()

    df['transformed_salary'] = scaler.fit_transform(df['salary'].values.reshape(-1,1))
    """
    Using this upped the Average Precision-recall score I was playing with for metrics, but lowered the
    Accuracy and Precision when I used Naive Bayes.   
    """
    df['composite_email_data'] = df['to_messages'] + df['from_messages'] \
                                 + df['from_this_person_to_poi'] + df['from_poi_to_this_person']
    df['composite_poi_email_data'] = df['from_this_person_to_poi'] + df['from_poi_to_this_person']
    print df
    #Using a subset of transformed salary caused my Naive Bayes classifier to perform worse than having all of the data in.
    #df = df.loc[df['transformed_salary'] > .10]
    my_dataset = df.to_dict(orient='index')
    print df.corr()
    return my_dataset



#poi_data_helper.plot_outliers(data[2],data[3], .6)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def create_classifier():

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()


    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    clf = RandomForestClassifier(n_jobs=10,max_features= 'sqrt' ,n_estimators=50, max_depth=None)
    param_grid = {
        'n_estimators': [100, 200,300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [5,10,15,]
    }
    
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
    CV_rfc.fit(features, labels)
    print CV_rfc.best_params_
    clf = RandomForestClassifier(max_features = 'auto', n_estimators = 100, max_depth = 5)
    """

    return clf

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

def test_features(my_dataset, features_list):
### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    test_code(features, labels)

def test_code(features, labels):
    import tester
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.5, random_state=42)
    from sklearn.metrics import average_precision_score

    #print "Average Precision-recall score = {0:0.5f}".format(average_precision_score(labels_train, labels_test))

    tester.main()

# Example starting point. Try investigating other evaluation techniques!
def process_poi_id_code():
    clf = create_classifier()
    my_dataset = process_features()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)
    test_features(my_dataset, features_list)

if __name__ == '__main__':
    process_poi_id_code()

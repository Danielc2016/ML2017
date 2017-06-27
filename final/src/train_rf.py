# coding: utf-8
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def load_data():
    # Traning data
    train = pd.read_csv("data/train_data.csv")
    train = train.fillna(train.median())
    train = shuffle(train)

    train['longitude_square'] = train['longitude'] ** 2
    train['latitude_square'] = train['latitude'] ** 2
    train['mul'] = train['latitude'] * train['longitude']
    train['q1'] = train['construction_year'] * train['quality_group']
    train['q2'] = train['waterpoint_type'] * train['longitude']
    train['q3'] = train['waterpoint_type'] ** 2
    train['q4'] = train['gps_height'] ** 2
    train['q5'] = train['quality_group'] * train['payment']
    print("Traning data process!")

    # Features for trainig
    column_labels = list(train.columns.values)
    column_labels.remove("id")
    column_labels.remove("date_recorded")
    column_labels.remove("status_group")
    status_group = ["functional", "non functional", "functional needs repair"]
    print("Features for trainig~~")

    train = train.iloc[np.random.permutation(len(train))]

    # Testing data
    test = pd.read_csv("data/test.csv")
    test = test.fillna(test.median())
    test['longitude_square'] = test['longitude'] ** 2
    test['latitude_square'] = test['latitude'] ** 2
    test['mul'] = test['latitude'] * test['longitude']
    test['q1'] = test['construction_year'] * test['quality_group']
    test['q2'] = test['waterpoint_type'] * test['longitude']
    test['q3'] = test['waterpoint_type'] ** 2
    test['q4'] = test['gps_height'] ** 2
    test['q5'] = test['quality_group'] * test['payment']
    # test['q7'] = test['payment']**2
    print("Testing data: successfully")

    return (train, test, column_labels, status_group)


def train():
    (train, test, column_labels, status_group) = load_data()

    # Assign data for validation
    amount = int(0.96*len(train))
    validation = train[amount:]
    train = train[:amount]
    print("Assign data for validation: successfully")


    clf = RandomForestClassifier(criterion='gini',
                                 min_samples_split=8,
                                 n_estimators=777,
                                 max_features='auto',
                                 oob_score=True,
                                 max_depth=50,
                                 random_state=1,
                                 n_jobs=-1)

    print("Classifier: successfully")

    clf.fit(train[column_labels], train["status_group"])
    print("Traning: successfully")

    predict_set = clf.predict(validation[column_labels])
    accuracy = accuracy_score(clf.predict(validation[column_labels]), validation["status_group"])
    print("Accuracy = " + str(accuracy))
    print("Accuracy: successfully")

    joblib.dump(clf, 'models/random_forest_model.pkl', compress=3) 


def main():
    train()


if __name__ == '__main__':
    main()
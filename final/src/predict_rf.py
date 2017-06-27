# coding: utf-8

import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

X_TEST_PATH = sys.argv[1]


def load_data():
    # Traning data
    train = pd.read_csv('data/train_data.csv')
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
    print('Traning data process!')

    # Features for trainig
    column_labels = list(train.columns.values)
    column_labels.remove('id')
    column_labels.remove('date_recorded')
    column_labels.remove('status_group')
    status_group = ['functional', 'non functional', 'functional needs repair']
    print('Features for trainig~~')

    train = train.iloc[np.random.permutation(len(train))]

    # Testing data
    test = pd.read_csv('data/test.csv')
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
    print('Testing data: successfully')

    return (train, test, column_labels, status_group)


def predict(output_file_name='prediction_random_forest.csv', model='models/rf_mod_compress829.pkl'):
    (train, test, column_labels, status_group) = load_data()

    clf = joblib.load(model) 

    prediction = clf.predict(test[column_labels])
    print('Prediction for test data: successfully')
    # print((np.where(prediction==0))[0].shape)
    # print((np.where(prediction==1))[0].shape)
    # print((np.where(prediction==2))[0].shape)

    ### Making submission file###
    # Dataframe as per submission format
    submission = pd.DataFrame({
                'id': test['id'],
                'status_group': prediction
            })
    for i in range(len(status_group)):
        submission.loc[submission['status_group'] == i, 'status_group'] = status_group[i]
    print('Dataframe as per submission format: successfully')

    submission.to_csv(output_file_name, index = False)
    print('Store submission dataframe into file: successfully')


def main():
    predict()


if __name__ == '__main__':
    main()
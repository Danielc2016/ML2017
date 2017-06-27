# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_pickle('data/df_train.pkl')

    # Create a purely numerical version of the dataframe:
    df_num = df.copy()
    # Many variables need to be encoded as numerical values.
    # We save the encoders for later use on the test data:
    encoder = []
    for col in [3,5,8,10,11,12,15,16,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,43]:
        temp = LabelEncoder()
        df_num.iloc[:,col] = temp.fit_transform(df_num.iloc[:,col].fillna('').astype(str))
        encoder.append(temp)
    
    # Convert dates to elapsed time since an arbitrary date (1/1/2000):
    df_num['date_recorded'] = (df.date_recorded - datetime.strptime('2000-01-01', '%Y-%m-%d'))/timedelta(days = 1)
    # Fill in all missing data with a mean:
    df_num = df_num.fillna(df_num.mean())

    # Examine the number of categories for each variable:
    # for col in df_num.columns:
        # print(col + ': ' + str(len(df_num[col].unique())))

    # Prepare the features and targets for classifier training:
    X_train = df_num.drop(['id', 'wpt_name', 'subvillage', 'status_group'], 1)
    y_train = df_num['status_group']

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return (X_train, y_train)


def train():
    (X_train, y_train) = load_data()

    # Create a gradient-boosted trees classifier:
    # (The hyperparameters used here were selected 
    #    through a long process of cross-validation.)
    clf = GradientBoostingClassifier(
        learning_rate = 0.02,
        max_depth = 13,
        max_features = 9,
        min_samples_leaf = 30,
        min_samples_split = 100,
        n_estimators = 800,
        subsample = .95
    ).fit(X_train, y_train)

    joblib.dump(clf, 'models/gradient_boost_model.pkl', compress=3) 


def main():
    train()


if __name__ == '__main__':
    main()


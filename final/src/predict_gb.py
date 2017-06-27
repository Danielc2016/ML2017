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

X_TEST_PATH = sys.argv[1]


def load_data():
    df = pd.read_pickle('data/df_train.pkl')
    dfT = pd.read_pickle('data/df_test.pkl')

    # Create a purely numerical version of the dataframe:
    df_num = df.copy()
    # Many variables need to be encoded as numerical values.
    # We save the encoders for later use on the test data:
    encoder = []
    for col in [3,5,8,10,11,12,15,16,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,43]:
        temp = LabelEncoder()
        df_num.iloc[:,col] = temp.fit_transform(df_num.iloc[:,col].fillna('').astype(str))
        encoder.append(temp)
    
    # Create a purely numerical version of the test dataframe:
    dfT_num = dfT.copy()
    i = 0
    
    for col in [3,5,8,10,11,12,15,16,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,43]:
        temp = encoder[i]
        dfT_num.iloc[:,col] = temp.classes_.searchsorted(dfT_num.iloc[:,col].fillna('').astype(str))
        i = i + 1
    dfT_num = dfT_num.fillna(dfT_num.mean())
    dfT_num['date_recorded'] = (dfT.date_recorded - datetime.strptime('2000-01-01', '%Y-%m-%d'))/timedelta(days = 1)
    X_test = dfT_num.drop(['id', 'wpt_name', 'subvillage', 'status_group'], 1)

    test_ID = pd.read_csv(X_TEST_PATH)['id']

    X_test = np.array(X_test)
    test_ID = np.array(test_ID)

    return (X_test, test_ID)


def predict(output_filename='prediction_gradient_boost.csv'):
    (X_test, test_ID) = load_data()

    # load pre-trained model
    clf = joblib.load('models/gradient_boost_model.pkl') 

    prediction = clf.predict(X_test)

    status_group = ["functional", "functional needs repair", "non functional"]
    submission = pd.DataFrame({
                "id": test_ID,
                "status_group": prediction
            })
    for i in range(len(status_group)):
        submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]

    submission.to_csv(output_filename, index = False)


def main():
    predict()


if __name__ == '__main__':
    main()
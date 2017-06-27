# coding: utf-8

import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

VALID_RATIO = 1
NUM_ENSENBLE = 11


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
    # make sure models/ dir exists
    if not os.path.exists('models/'):
        os.makedirs('models/')

    for i in range(NUM_ENSENBLE):
        (X_train, y_train) = load_data()
        np.random.seed(i**2)
        # y_train = pd.get_dummies(y_train).values
        indices = np.arange(X_train.shape[0])

        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        VALID = int(VALID_RATIO * len(X_train))
        X_valid = X_train[VALID:]
        X_train = X_train[:VALID]
        y_valid = y_train[VALID:]
        y_train = y_train[:VALID]
        
        xg_train = xgb.DMatrix(X_train, label=y_train)

        # Setting Booster parameters
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        param['eta'] = 0.2
        param['max_depth'] = 14
        # param['min_child_weight'] = 3
        # param['gamma'] = 0.6
        param['silent'] = 1
        param['nthread'] = multiprocessing.cpu_count() - 1
        param['num_class'] = 3
        param['colsample_bytree'] = 0.4
        param['seed'] = i
        # param['max_bins'] = 10000

        # Training
        bst = None
        num_round = 500
        if VALID_RATIO == 1:
            bst = xgb.train(param, xg_train, num_round)
        else:
            xg_valid = xgb.DMatrix(X_valid, label=y_valid)
            # Specify validations set to watch performance
            evallist  = [(xg_train, 'train'), (xg_valid, 'valid')]
        
            bst = xgb.train(param, xg_train, num_round, evals=evallist, early_stopping_rounds=20)

            print('best_acc', 1 - bst.best_score)
            print('best_iteration ', bst.best_iteration)
            print('best_ntree_limit', bst.best_ntree_limit)
        
        # Save model
        model_path = 'models/xgboost_' + str(i) + '.model'
        bst.save_model(model_path)
        print('(' + str(i + 1) + ' of 11) save model to : ' + model_path)


def main():
    train()


if __name__ == '__main__':
    main()
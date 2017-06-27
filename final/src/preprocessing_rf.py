# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]
X_TEST_PATH = sys.argv[3]


def preprocessing_rf():
    train_label = pd.read_csv(Y_TRAIN_PATH)

    column_labels = list(train_label.columns.values)
    column_labels.remove('id')

    for i in column_labels:
        unique_value = train_label[i].unique()
        size = len(unique_value)
        print(size)
        for j in range(size):
            if unique_value[j] != 'nan':
                train_label.loc[train_label[i] == unique_value[j], i] = j

    
    train_value = pd.read_csv(X_TRAIN_PATH)
    test = pd.read_csv(X_TEST_PATH)

    column_labels = list(train_value.columns.values)
    column_labels.remove('id')
    column_labels.remove('amount_tsh')
    column_labels.remove('date_recorded')
    column_labels.remove('gps_height')
    column_labels.remove('longitude')
    column_labels.remove('latitude')
    column_labels.remove('num_private')
    column_labels.remove('region_code')
    column_labels.remove('district_code')
    column_labels.remove('population')
    column_labels.remove('construction_year')

    test = test.fillna(test.median())

    for i in column_labels:
        unique_value = list(set(np.concatenate((train_value[i].unique() , test[i].unique()))))
        size = len(unique_value)
        print(size)
        for j in range(size):
            if unique_value[j] != 'nan':
                train_value.loc[train_value[i] == unique_value[j], i] = j
                test.loc[test[i] == unique_value[j], i] = j

    train_value = train_value.fillna(train_value.median())
    test = test.fillna(test.median())

    test.to_csv('data/test.csv', index = False)

    train_data = train_value.merge(train_label, how = "outer", on = "id", sort = True)
    train_data = train_data.fillna(train_data.median())

    train_data.to_csv("data/train_data.csv", index = False)


def main():
    # make sure data/ dir exists
    if not os.path.exists('data/'):
        os.makedirs('data/')

    preprocessing_rf()


if __name__ == '__main__':
    main()
# coding: utf-8

import os
import sys
import pandas as pd
from numpy import nan
from datetime import timedelta

X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]
X_TEST_PATH = sys.argv[3]


def preprocessing_gb():
    # Load the training data and labels:
    df = pd.read_csv(
        X_TRAIN_PATH,
        parse_dates = ['date_recorded'], 
        infer_datetime_format = True
    ).merge(
        pd.read_csv(Y_TRAIN_PATH),
        on = 'id'
    )

    # Many variables have values with inconsistent capitalization, 
    # so convert them all to lowercase:
    df.funder = df.funder.str.lower()
    df.installer = df.installer.str.lower()
    df.wpt_name = df.wpt_name.str.lower()
    df.basin = df.basin.str.lower()
    df.subvillage = df.subvillage.str.lower()
    df.region = df.region.str.lower()
    df.lga = df.lga.str.lower()
    df.ward = df.ward.str.lower()
    df.recorded_by = df.recorded_by.str.lower()
    df.scheme_management = df.scheme_management.str.lower()
    df.scheme_name = df.scheme_name.str.lower()

    # Suppress chained assignment issues:
    pd.options.mode.chained_assignment = None

    # Make room for three new features --
    # The distance to the closest pump result:
    df['closest_dist'] = nan
    # The elapsed time since the closest pump result:
    df['closest_time'] = nan
    # The status group of the closest pump result:
    df['closest_stat'] = nan

    # Fill in the new features:
    for row in range(len(df)):
        c = (df.latitude.iloc[row], df.longitude.iloc[row])
        t = df.date_recorded.iloc[row]
        i = df.id[row]
        close = df[
            (df.date_recorded > t - timedelta(days = 180)) &
            (df.date_recorded < t + timedelta(days = 180)) & 
            (df.id != i)
        ][['date_recorded','longitude','latitude','status_group']]
        if len(close) > 0:
            close['dist'] = (close.latitude - c[0])**2 + (close.longitude - c[1])**2
            close['time'] = ((close.date_recorded - t) / (timedelta(days=1))).astype(float)
            close = close.sort_values('dist').iloc[0,:]
            df.iloc[row, 41:44] = close[['dist','time','status_group']].values
        if row % 1000 == 0:
            print(str(row) + ' of ' + str(len(df)))

    # Save the dataframe to disk, since it is time-consuming to produce:
    df.to_pickle('data/df_train.pkl')
    
    ############################### test part
    dfT = pd.read_csv(
        X_TEST_PATH,
        parse_dates = ['date_recorded'], 
        infer_datetime_format = True
    )
    dfT.funder = dfT.funder.str.lower()
    dfT.installer = dfT.installer.str.lower()
    dfT.wpt_name = dfT.wpt_name.str.lower()
    dfT.basin = dfT.basin.str.lower()
    dfT.subvillage = dfT.subvillage.str.lower()
    dfT.region = dfT.region.str.lower()
    dfT.lga = dfT.lga.str.lower()
    dfT.ward = dfT.ward.str.lower()
    dfT.recorded_by = dfT.recorded_by.str.lower()
    dfT.scheme_management = dfT.scheme_management.str.lower()
    dfT.scheme_name = dfT.scheme_name.str.lower()

    dfT['status_group'] = nan
    dfT['closest_dist'] = nan
    dfT['closest_time'] = nan
    dfT['closest_stat'] = nan

    for row in range(len(dfT)):
        c = (dfT.latitude.iloc[row], dfT.longitude.iloc[row])
        t = dfT.date_recorded.iloc[row]
        i = dfT.id[row]
        close = df[
            (df.date_recorded > t - timedelta(days = 180)) &
            (df.date_recorded < t + timedelta(days = 180)) & 
            (df.id != i)
        ][['date_recorded','longitude','latitude','status_group']]
        if len(close) > 0:
            close['dist'] = (close.latitude - c[0])**2 + (close.longitude - c[1])**2
            close['time'] = ((close.date_recorded - t) / (timedelta(days=1))).astype(float)
            close = close.sort_values('dist').iloc[0,:]
            dfT.iloc[row, 41:44] = close[['dist','time','status_group']].values
        if row % 10000 == 0:
            print(str(row) + ' of ' + str(len(dfT)))
    dfT.to_pickle('data/df_test.pkl')


def main():
    # make sure data/ dir exists
    if not os.path.exists('data/'):
        os.makedirs('data/')

    preprocessing_gb()


if __name__ == '__main__':
    main()
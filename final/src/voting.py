import numpy as np
import pandas as pd
from scipy import stats


def voting(filename_list, output_filename):

    prediction = np.zeros((len(filename_list), 14850)).astype(str)
    for i, filename in enumerate(filename_list):
        prediction[i, :] = pd.read_csv(filename)['status_group'].values

    prediction_ensemble = stats.mode(prediction)[0][0]

    test_ID = pd.read_csv(filename_list[0])['id'].values
    submission = pd.DataFrame({
                'id': test_ID,
                'status_group': prediction_ensemble
            })

    submission.to_csv(output_filename, index = False)


def main():
    filename_list = ['prediction_xgboost.csv', 
                     'prediction_gradient_boost.csv', 
                     'prediction_random_forest_1.csv', 
                     'prediction_random_forest_2.csv']
    output_filename = 'voting_' + str(filename_list) + '.csv'
    
    voting(filename_list, output_filename)


if __name__ == '__main__':
    main()
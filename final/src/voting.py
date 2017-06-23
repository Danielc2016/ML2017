import numpy as np
import pandas as pd
from scipy import stats

filename_list = ['prediction_gb.csv',
                 'submission_8313.csv',
                 'submission_8276.csv',
                 'submission_xgb_0.8201.csv']

prediction = np.zeros((len(filename_list), 14850)).astype(str)
for i, filename in enumerate(filename_list):
    prediction[i, :] = pd.read_csv(filename)['status_group'].values

prediction_ensemble = stats.mode(prediction)[0][0]
print(prediction_ensemble[:10])

test_ID = pd.read_csv(filename_list[0])['id'].values
submission = pd.DataFrame({
            'id': test_ID,
            'status_group': prediction_ensemble
        })

submission.to_csv('voting_' + str(filename_list) + '.csv', index = False)

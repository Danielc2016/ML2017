# ML2017 Final Project
This is the final project for our Machine Learning Class 2017
The topic is Pump it Up: Data Mining the Water Table hosted by drivendata
https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

## Data
|     File    |  Description | 
|-------------|-------------|
| X_train.csv | The independent variables for the training set |
| y_train.csv | The dependent variable (status_group) for each of the rows in Training set values | 
| X_test.csv  | The independent variables that need predictions |

## Usage

### Reproduce Our Best Leaderboard Score (about 0.8260)
Using pre-trained models.
```
python best.py data/X_test.csv
```

### Step by step
A step by step series of our experiments.
#### 1. Preprocessing
For Random Forest model:
```
python preprocessing_rf.py data/X_train.csv data/y_train.csv data/X_test.csv
```
For Gradient Boosting / XGBoost model:
```
python preprocessing_gb.py data/X_train.csv data/y_train.csv data/X_test.csv
```
#### 2. Train
Random Forest
```
python train_rf.py
```
Gradient Boosting
```
python train_gb.py
```
XGBoost model
```
python train_xgb.py
```

#### 3. Predict
Random Forest
```
python predict_rf.py data/X_test.csv
```
Gradient Boosting
```
python predict_gb.py data/X_test.csv
```
XGBoost model
```
python predict_xgb.py data/X_test.csv
```
Finally, Hard voting
```
python voting.py
```

## Requirements
* python3
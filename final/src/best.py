# coding: utf-8
import predict_rf
import predict_gb
import predict_xgb
import voting


def main():
    filename_list = ['prediction_random_forest_1.csv', 
                     'prediction_random_forest_2.csv',
                     'prediction_gradient_boost.csv',
                     'prediction_xgboost.csv']

    predict_rf.predict(filename_list[0], 'models/rf_mod_compress828.pkl')
    predict_rf.predict(filename_list[1], 'models/rf_mod_compress829.pkl')
    predict_gb.predict(filename_list[2])
    predict_xgb.predict(filename_list[3])

    voting.voting(filename_list , 'best_leader_board.csv')


if __name__ == '__main__':
    main()
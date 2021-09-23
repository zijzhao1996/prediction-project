#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Date  : 2021/9/23


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pk
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from mlxtend.regressor import StackingCVRegressor


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    A derived model that average all sklearn models' predictions.
    """
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 

def load_data():
    """
    Load dataset.
    Return: dataset in pandas dataframe format.
    """
    try:
        dataset = pd.read_csv(sys.argv[1])
        print('-------- Loading the dataset sucessfully. -------')
        return dataset
    except Exception as e:
        print(e)


def data_preprocess(dataset, pca_ncomp = 150):
    """
    Preprocessing the data via multiple steps.
    Return: X: original features
            X_pca_scaled: features scaled by PCA
            y1: target 1
            y2: target 2
    """
    # step 1: convert categorical features to dummies
    summary_statistics = dataset.describe()
    cat_columns = []
    for i in summary_statistics.iloc[-1:]:
        # if max == 0.999, this column is ctaegorical
        if summary_statistics.iloc[-1:][i].values[0] == 0.9990000000000001:
            cat_columns.append(i)
    dataset = pd.get_dummies(dataset, columns=cat_columns, drop_first=True) 

    # step2: drop NAs
    dataset.dropna(axis=0, how='any', inplace=True)

    # step3: X and y splits
    X = dataset.drop(labels=['Unnamed: 0', 'y1', 'y2'], axis=1, inplace=False)
    y1 = dataset['y1']
    y2 = dataset['y2']

    # step 4: perform PCA
    # load the previous PCA weights
    path = 'PCA/'
    pca = pk.load(open(path + "pca.pkl",'rb'))
    X_pca_scaled = pca.transform(X)
    print('-------- Preprocessing the dataset sucessfully. -------')

    return X, X_pca_scaled, y1, y2


def load_model(str1='Stacking_y1.pkl', str2='Stacking_y2.pkl'):
    """
    Load following models: Ridge_y1, Ridge_y2, KRR_y1, KRR_y2, RF_y1, RF_y2, 
                           GBoost_y1, GBoost_y2, LGBM_y1, LGBM_y2, 
                           Averaged_y1, Averaged_y2, Stacking_y1, Stacking_y2
    Return: Trained models for Y1 and Y2
    """
    path = 'models/'
    with open(path + str1, 'rb') as f1:
        model1 = pk.load(f1)
    with open(path + str2, 'rb') as f2:
        model2 = pk.load(f2)
    print('-------- Loading the models sucessfully. -------')

    return model1, model2


def corr_metrics(model1, model2, X_pca_scaled, y1, y2):
    """
    Output correlation coefficient 
    Return: corr1: correlation coefficient between y1 and y1_pred
            corr2: correlation coefficient between y2 and y2_pred
    """
    y1_pred = model1.predict(X_pca_scaled)
    y2_pred = model2.predict(X_pca_scaled)
    corr1 = np.corrcoef(y1, y1_pred)[1,0]
    corr2 = np.corrcoef(y2, y2_pred)[1,0]

    return corr1, corr2


def mse_metrics(model1, model2, X_pca_scaled, y1, y2):
    """
    Output mean squared error
    Return: mse1: mse between y1 and y1_pred
            mse2: mse between y2 and y2_pred
    """
    y1_pred = model1.predict(X_pca_scaled)
    y2_pred = model2.predict(X_pca_scaled)
    mse1 = mean_squared_error(y1, y1_pred)
    mse2 = mean_squared_error(y2, y2_pred)

    return mse1, mse2


def model_plot(model1, model2, X_pca_scaled, y1, y2, subsample=250):
    """
    Generate a plot of predicted and ground truth Y values.
    """
    plt.figure(figsize=(20,10))
    plt.subplot(2, 1, 1)
    y1_pred = model1.predict(X_pca_scaled)
    plt.plot(range(subsample), y1[:subsample], 'r', label='ground truth')
    plt.plot(range(subsample), y1_pred[:subsample], 'b', label='pred')
    plt.legend()
    plt.title('Subsample of predicted and ground truth Y1')

    y2_pred = model2.predict(X_pca_scaled)
    plt.subplot(2, 1, 2)
    plt.plot(range(subsample), y2[:subsample], 'r', label='ground truth')
    plt.plot(range(subsample), y2_pred[:subsample], 'b', label='pred')
    plt.title('Subsample of predicted and ground truth Y2')

    path = 'plots/'
    plt.savefig(path + 'plot')


def main():
    dataset = load_data()
    X, X_pca_scaled, y1, y2 = data_preprocess(dataset)
    model1, model2 = load_model(sys.argv[2], sys.argv[3])

    # output correlation coefficient 
    corr1, corr2 = corr_metrics(model1, model2, X_pca_scaled, y1, y2)
    print('The correlation coefficent between Y1 and Y1_pred is:', corr1)
    print('The correlation coefficent between Y2 and Y2_pred is:', corr2)

    # output RMSE
    mse1, mse2 = mse_metrics(model1, model2, X_pca_scaled, y1, y2)
    print('The mean squared error between Y1 and Y1_pred is:', mse1)
    print('The mean squared error between Y2 and Y2_pred is:', mse2)

    model_plot(model1, model2, X_pca_scaled, y1, y2)
    print('Plot of predicted and ground truth Y values has been generated.')

    print('-------- Finished --------')


if __name__ == "__main__":
    main()

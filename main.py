#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 20:48:42 2021

@author: abdul
"""


from numpy import nan
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from datetime import datetime, timedelta
from scipy.stats import pearsonr, chi2_contingency
from utils import *
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


data_path = './data/'
plot_path = './plots/'

if __name__ == '__main__':
    #LOADING CDNOW DATASET
    data = pd.read_csv(data_path+'CDNOW.csv', sep=' ')
    data['new_date'] = data.Date.apply(str)
    data['Date'] = pd.to_datetime(data.new_date, format='%Y-%m-%d')
    data = data.drop(['new_date'], axis=1)

    #SPLITTING THE DATASET
    max_date = max(data['Date'])
    testing_period = 90
    cut_off = max_date-timedelta(testing_period)
    mask1 = data['Date']<cut_off
    mask2 = data['Date']>=cut_off
    df1 = data.loc[mask1]
    df2 = data.loc[mask2]
    max_date1 = max(df1['Date'])
    max_date2 = max(df2['Date'])
    max_id = max(data['Id'])


    #feature engg
    x = rfm(df1, max_date1)
    
    y= target(df2, max_id)
    
    new_data = pd.merge(x, y, on='Id')
    new_data.to_pickle('ready_data.pkl')
    
#    new_data = pd.read_pickle('ready_data.pkl')
    
    X = new_data[['Recency', 'Qty', 'Price_x', 'Mean']]
    y_prob = new_data['y']
    y_spend = new_data['Price_y']
    
    
    # define model
    model = XGBRegressor()

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y_spend, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    
    # fit model
    model.fit(X, y_spend)    
    
    yhat = model.predict(X)

    plt.figure(figsize = (12,9))
    plt.scatter(y_spend, yhat)
    p1, p2 = [0, max(y_spend)], [0, max(y_spend)]
    plt.plot(p1, p2, color ='red')
    plt.title('REGRESSOR: Spend Prediction')
    plt.xlabel('y_actual')
    plt.ylabel('y_pred')
    plt.savefig(plot_path+'price_prediction.png')
    plt.show()
    
    corr, _ = pearsonr(y_spend, yhat)
    print('Pearsons correlation: %.3f' % corr)
    rmse= np.sqrt(np.square(y_spend - yhat))
    print('Avg RMSE:', np.average(rmse))
    abs_error= (y_spend - yhat)
    print('Avg ABS ERROR:', np.average(abs_error))
    
    
    
    #CLASSIFICATION
    # define model
    model1 = XGBClassifier()
    model1.fit(X, y_prob)    
    
    yhat1 = model1.predict(X)  
    yproba = model1.predict_proba(X)
    y_proba = yproba[:, 1]
    
    print(classification_report(y_prob, yhat1))
    auc_score = roc_auc_score(y_prob, yhat1)
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)

    fpr, tpr, thresh = metrics.roc_curve(yhat1, y_prob)
    plt.figure(figsize = (12,9))
    plt.plot(fpr, tpr, label="XGB, AUC="+str(auc_score))
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    p1, p2 = [0,1], [0, 1]
    plt.plot(p1, p2, color ='red', label = 'Baseline, AUC=0.50')
    plt.legend(loc=4)    
    plt.title('CLASSIFIER: ROC Curve')
    plt.savefig(plot_path+'ROC.png')
    plt.show()


    dash_app = new_data[['Id', 'Qty', 'Price_y', 'y']]
    dash_app['Pred_Price_y'] = list(yhat)
    dash_app['Pred_y'] = list(y_proba)
    
    dash_app.to_pickle('dash_app.pkl')



    plt.figure(figsize = (12,9))
    num_colors = 3
    cmap = plt.get_cmap('viridis', num_colors)
    cmap.set_under('red')
    plt.scatter(dash_app['Price_y'], dash_app['Pred_y'], c=dash_app['Pred_y'], cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(extend='min')
    plt.title('Purchase Value vs Purchase Probability')
    plt.xlabel('Purchase Value', fontsize=10)
    plt.ylabel('Probability of Purchase', fontsize=10)
    plt.savefig(plot_path+'plot1.png')
    plt.show()









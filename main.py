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
    testing_period = 120
    cut_off = max_date-timedelta(testing_period)
    mask1 = data['Date']<cut_off
    mask2 = data['Date']>=cut_off
    df1 = data.loc[mask1]
    df2 = data.loc[mask2]
    max_date1 = max(df1['Date'])
    max_date2 = max(df2['Date'])


    #RECENCY CALCULATION
    x_train, price_train , y_train = formatting(df1, max_date1)
    x_test, price_test , y_test = formatting(df2, max_date2)

    #CLASSIFICATION TASK
    y_pred, fpr, tpr, auc_score = classification(x_train, y_train,x_test, y_test)
    print('=============NEURAL NETWORK PREDICTIONS==============')
    print(classification_report(y_test, y_pred))

    #PLOTTING ROC CURVE    
    plt.plot(fpr, tpr, label="NN Model, AUC="+str(auc_score))
    plt.legend(loc=0)
    plt.title('ROC Curve')
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    plt.savefig(plot_path+'roc.png')
    plt.show()


    #REGRESSION TASK
    pred2 = regression(x_train, price_train, x_test, price_test)
    plt.scatter(price_test, pred2)
    plt.title('Regression Scatter Plot')
    p1, p2 = [0, 4000], [0, 4000]
    plt.plot(p1, p2, color ='red')    
    plt.savefig(plot_path+'scatter.png')
    plt.show()

    corr, _ = pearsonr(price_test, pred2)
    print('Pearsons correlation: %.3f' % corr)
    rmse= np.sqrt(np.square(price_test-pred2))
    print('Avg RMSE:', np.average(rmse))





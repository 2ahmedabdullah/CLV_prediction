
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd

model_path = './models/'
plot_path = './plots/'

def rfm(df1, max_date):
    max_date_by_id = df1.groupby(['Id'])[['Date']].max().reset_index()
    
    dff = df1.groupby(['Id'])[['Qty', 'Price']].sum().reset_index()
    dff1 = df1.groupby(['Id'])['Price'].mean().reset_index()
    dff['Recency'] = (max_date_by_id['Date'] - max_date).dt.days
    dff['Mean'] = dff1['Price']
    return dff


def target(df1, max_id):
    
    dff = df1.groupby(['Id'])[['Qty', 'Price']].sum().reset_index()
    dff = dff.drop(['Qty'], axis=1)
    dff['y'] = list([1]*len(dff))
    
    all_id = list(range(1, max_id+1))
    present = list(dff['Id'])
    
    absent = [x for x in all_id if x not in present]
    no_price = [0]*len(absent)
    dff1 = pd.DataFrame([absent, no_price, no_price]).transpose()
    dff1.columns = ['Id', 'Price', 'y']
    bigdata = pd.concat([dff, dff1], ignore_index=True, sort=True)
    bigdata = bigdata.sort_values('Id')    
    return bigdata




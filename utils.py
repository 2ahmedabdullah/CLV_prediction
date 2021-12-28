
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


model_path = './models/'
plot_path = './plots/'

def formatting(df1, max_date):
    by_date = df1.groupby(['Id'])[['Date']].max().reset_index()
    
    dff = df1.groupby(['Id'])[['Qty', 'Price']].sum().reset_index()
    dff1 = df1.groupby(['Id'])['Price'].mean().reset_index()
    
    
    dff['Recency'] = (by_date['Date'] - max_date).dt.days
    dff['Mean'] = dff1['Price']
    dff = dff.drop(['Id'], axis=1)
    check1 = dff['Price']>0
    check1= check1*1
    dff['y'] = check1
    price = dff['Price']
    y = dff['y']
    dff = dff.drop(['Price','y'], axis=1)
    return dff, price, y 


def regression(x_train, price_train, x_test, price_test):
    input_dim = 3
    model_reg = Sequential()
    model_reg.add(Dense(6, activation='relu', input_shape=(input_dim,)))
    model_reg.add(Dense(3, activation='relu'))
    model_reg.add(Dense(1, activation='relu'))
    model_reg.compile(loss='mean_absolute_error', optimizer = Adam())

    model2= model_reg.fit(x_train, price_train, batch_size=128, epochs=50, 
                      verbose=1, validation_data=(x_test, price_test))

    #model_reg.save(model_path+'regression.h5')
    #saved_model = load_model(model_path+'regression.h5')

    plt.plot(model2.history['loss'])
    plt.plot(model2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Regression')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(plot_path+'regression_loss.png')
    plt.show()



    y_pred = model_reg.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    return pred2


def prediction(y_pred):
    binary_threshold = 0.5
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    pred3= pred2>binary_threshold
    pred4 = pred3*1
    return pred4, pred2


def classification(x_train, y_train,x_test, y_test):
    input_dim = 3
    model_class = Sequential()
    model_class.add(Dense(12, activation='relu', input_shape=(input_dim,)))
    model_class.add(Dense(8, activation='relu'))
    model_class.add(Dense(4, activation='relu'))
    model_class.add(Dense(1, activation='sigmoid'))
    model_class.compile(loss='binary_crossentropy', optimizer = Adam())

    model1= model_class.fit(x_train, y_train, batch_size=64, epochs=20, 
                      verbose=1, validation_data=(x_test, y_test))

    #model_class.save(model_path+'classification.h5')
   #saved_model = load_model(model_path+'classification.h5')


    plt.plot(model1.history['loss'])
    plt.plot(model1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Classification')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(plot_path+'classification_loss.png')
    plt.show()

    y_pred = model_class.predict(x_test)
    y_pred, y_probs = prediction(y_pred)
    auc_score = roc_auc_score(y_test, y_probs)
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
    return y_pred, fpr, tpr, auc_score

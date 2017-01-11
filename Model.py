# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:47:12 2017

@author: Raymond
"""

from Controller import Controller
import pandas as pd
import numpy as np
import json
import requests
import sys
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
from sklearn.svm import SVC

class Web_Crawler(object):
    
    @staticmethod
    def Bloomberg_data(Ticker):        
        Ticker = Ticker.replace(':','%3A')
        
        res = requests.get("http://www.bloomberg.com/markets/api/bulk-time-series/price/"+str(Ticker)+"?timeFrame=1_YEAR")
        data = json.loads(res.text)
    
        PX_Price = data[0]['price']
    
        date_list = []
        value_list = []
    
        for each in PX_Price:
            date_list.append(each['date'])
            value_list.append(each['value'])
    
        df = pd.Series(value_list,index=date_list)
        df.index = df.index.to_datetime()
    
        return df

    @staticmethod
    def Bloomberg_Tickers():
        Regions = ['asiaPacific','emea','americas']
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        Ticker_Name = []            

        for Region in Regions:
            url = 'https://www.bloomberg.com/markets/api/comparison/geographic-indices?name=' + str(Region) + '&type=region&securityType=COMMON_STOCK&locale=en'
            res = requests.get(url, headers = headers)
            data = json.loads(res.text)
            
            Number = len(data)
            
            
            for i in range(Number):
                for j in range(len(data[i]['fieldDataCollection'])):
                    Name = data[i]['fieldDataCollection'][j]['longName']
                    ID = data[i]['fieldDataCollection'][j]['id']
                    #print Name, ID
                    Ticker_Name.append((ID,Name))

        return Ticker_Name
#==============================================================================

class Model(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def Data_to_db(Ticker, Name):
        try:
            data = Web_Crawler.Bloomberg_data(Ticker)
            sys.stderr.write("[info] successfully prepared data list of size " + str(len(data))) + '\n'
        except:
            raise Exception("Error with internet connection to www.bloomberg.com")
            
        try:
            Controller.Data_insert(Ticker = Ticker, Name = Name, Series = data)
            sys.stderr.write("[info] wrote " + str(len(data)) + ' rows to db \n')
            sys.stderr.write("[info] " + str(Ticker) + ' saved successfully \n')
        except:
            raise Exception("Error with I/O to db.")
    
    @staticmethod
    def Update_All(): #Find the ticker and name according to Summary_table
        Ticker_Name_list = Controller.Ticker_query()
        
        for each in Ticker_Name_list:
            try:
                Model.Data_to_db(Ticker = each[0], Name = each[1])
            except:
                pass


#==============================================================================
class Machine_Learning(object):
    
    @staticmethod
    def Decision_Tree(df,train_test_ratio,Predict_label):
        features_df = df[df.columns.drop(Predict_label)]
        
        features_list = []
        for i in range(len(features_df)):
            features_list.append(features_df.ix[i].tolist())
        
        predict_list = df[Predict_label].tolist()
    
        if len(features_list) == len(predict_list):
            Data_length = len(features_list)
        else:
            raise ValueError('Error with number of data.')
        Train_number = int(Data_length * train_test_ratio)
        Test_number = int(Data_length * (1-train_test_ratio))
        X_train = features_list[:Train_number]
        Y_train = predict_list[:Train_number]
        X_test = features_list[Train_number:]
        Y_test = predict_list[Train_number:]
        Date_list = df.index[Train_number:]
        
        clf_svm = SVC()
        clf_svm.fit(X_train,Y_train)    
        print '=== Support Vector Machine ==='
        print "Numbers of In-sample and out-sample = (" + str(Train_number) + ',' + str(Test_number) + ')'
        print "Prediction accuracy = " + str(round(clf_svm.score(X_test,Y_test)*100,2)) + '%'
        
        clf_tree = DecisionTreeClassifier()
        clf_tree.fit(X_train,Y_train)
        #print Ticker, clf_tree.score(Train_Predict[2],Train_Predict[3])
        
        Signal_list = []        
        for i in range(len(Y_test)):
        #for i in np.arange(-11,-1):
            Signal_list.append(clf_tree.predict(X_test[i]))
            
            #print Y_test[i], clf_tree.predict(X_test[i])
            #print str(Date_list[i])[:10] + ' Actual:[' + str(Y_test[i]) + '], Prediction:' + str(clf_tree.predict(X_test[i]))
        
        print '=== Decision Tree ==='
        print "Numbers of In-sample and out-sample = (" + str(Train_number) + ',' + str(Test_number) + ')'
        print "Prediction accuracy = " + str(round(clf_tree.score(X_test,Y_test)*100,2)) + '%'

        Signal_df = pd.DataFrame(Signal_list,index = Date_list,columns = ['Signal'])
        Signal_df.index = Signal_df.index.to_datetime()        
        
        
        return Signal_df
#==============================================================================
def Split_predict_label(df,Predict_label):
    #print df
    features_df = df[df.columns.drop(Predict_label)]
    predict_list = df[Predict_label].tolist()

    features_list = []
    
    for i in range(len(features_df)):
        features_list.append(features_df.ix[i].tolist())
    #print len(features_df),len(predict_list)
    return features_list,predict_list






def Cross_Validation_with_DataFrame(df,Number_split,Predict_label):

    Number = len(df)/Number_split
    
    Order = range(0,len(df),Number)
    Order = Order[:-1]
    print '--------------------------------------------------------'
    print 'Cross validation for SVM'
    for each in Order:
        Train_data = df.drop(df.index[each:each + Number])
        Test_data = df.ix[each:each + Number]
        In_Sample = Split_predict_label(df = Train_data,Predict_label = Predict_label)
        Out_Sample = Split_predict_label(df = Test_data,Predict_label = Predict_label)
        
        
        clf_svm = SVC()
        clf_svm.fit(In_Sample[0],In_Sample[1])
        print clf_svm.score(Out_Sample[0],Out_Sample[1])

    print 'Cross validation for Decision Tree model'
    for each in Order:
        Train_data = df.drop(df.index[each:each + Number])
        Test_data = df.ix[each:each + Number]
        In_Sample = Split_predict_label(df = Train_data,Predict_label = Predict_label)
        Out_Sample = Split_predict_label(df = Test_data,Predict_label = Predict_label)
        
        
        clf_tree = DecisionTreeClassifier()
        clf_tree.fit(In_Sample[0],In_Sample[1])
        print clf_tree.score(Out_Sample[0],Out_Sample[1])
        
        Train_data = np.nan
        Test_data = np.nan


#==============================================================================
Tickers = ['SPX:IND', 'TWSE:IND', 'NKY:IND','NDX:IND','CCMP:IND','SPTSX60:IND',
          'DAX:IND', 'UKX:IND','SET:IND', 'USDJPY:CUR','KOSPI:IND','TRAN:IND',
          'CAC:IND', 'HSI:IND','PCOMP:IND',#'VNINDEX:IND',
          'SHCOMP:IND','CL1:COM','HG1:COM','GC1:COM',
          'C 1:COM','S 1:COM','W 1:COM','LC1:COM','JCI:IND','SENSEX:IND','SMI:IND','NG1:COM']


data = Controller.Multi_data_query(Tickers = Tickers)
# 'VNINDEX:IND',,'RTSI$:IND'
#for each in data.columns:
    #print each + ' ' + str(data[each].dropna().index[0])[:10]



'''
The data above will generate some NaN value in DataFrame, so drop the entire
row when all index have the NaN data.

After drop the NaN where all index have NaN values. At the end, few NaN value 
still exist in DataFrame. So the next process below will adopt forward filling.
'''

#
data = data.fillna(method = 'ffill')
data = data.dropna()
#print data
#Lookback window and Forward return parameter.
Shift_para = 20

Momentum = data.pct_change(Shift_para) #Past data
Forward = data.pct_change(-Shift_para) #Forward data
#print Momentum
#print Forward

freq = 'MS'
how = 'first'

Momentum = Momentum.resample(freq, how = how).dropna()
Forward = Forward.resample(freq, how = how).dropna()


#for Predict_target in Tickers:
Predict_target = 'SPX:IND'
Predict_table = Momentum
Predict_table['Predict_label'] = Forward[Predict_target]
#Predict_table['Volatility'] = pd.rolling_std(Predict_table['Predict_label'].pct_change(),Shift_para).diff()
Predict_table['Predict_label'] = np.where(Predict_table['Predict_label'] > 0., 1, 0)
Predict_table = Predict_table.dropna()



print '==================== ' + str(Predict_target) + ' ===================='
Out_sample_prediction = Machine_Learning.Decision_Tree(df = Predict_table,train_test_ratio=0.6,Predict_label='Predict_label')
#print Out_sample_prediction
#print Out_sample_prediction
#print len(Out_sample_prediction)
Cross_Validation_with_DataFrame(df = Predict_table,Number_split = 4,Predict_label='Predict_label')
data = data.resample('D',how = 'last')
data['Signal'] = Out_sample_prediction['Signal']
#data['Signal'] = data['Signal'].replace(0,-1)
data = data.dropna(how = 'all').fillna(method = 'ffill')
data['Return'] = data[Predict_target].pct_change()
data['Strategic_Return'] = data['Return']*data['Signal'].shift(1) + 1.
data = data.dropna()
data['EquityCurve'] = data['Strategic_Return'].cumprod()
data['BH'] = (data['Return'] + 1.).cumprod()
#print data
data[['EquityCurve','BH']].dropna().plot()
'''
Target_asset = pd.DataFrame(data[Predict_target])
Out_sample_df = Target_asset.resample(freq, how = how)
Out_sample_df['Signal'] = np.nan
Out_sample_df['Signal'][-len(Out_sample_prediction):] = Out_sample_prediction


Target_asset['Signal'] = Out_sample_df['Signal']
Target_asset = Target_asset.fillna(method = 'ffill')
Target_asset['Return'] = Target_asset[Predict_target].pct_change() + 1.
Target_asset['Strategic_Return'] = Target_asset['Return'] *Target_asset['Signal'].shift(1)
Target_asset['EquityCurve'] = Target_asset['Strategic_Return'].cumprod()
Target_asset['Buy&Hold'] = Target_asset['Return'].cumprod()
#Target_asset[['EquityCurve','Buy&Hold']].dropna().plot()
'''











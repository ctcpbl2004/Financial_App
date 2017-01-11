# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 16:47:20 2017

@author: Raymond
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import datetime
class Controller(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def Ticker_query(): #Output Ticker
        connection = sqlite3.connect("Fin_database.db")
        cursor = connection.cursor()
        
        cursor.execute("SELECT Ticker, Name FROM Summary_table")
        
        Ticker_data = cursor.fetchall()
        
        connection.close()
        
        return Ticker_data

    @staticmethod #Output Time Series data
    def Data_query(Ticker,Start = None, End = None):
        connection = sqlite3.connect("Fin_database.db")
        cursor = connection.cursor()
        
        if (Start == None) & (End == None):
            cursor.execute("SELECT Date, Value FROM Time_Series WHERE Ticker = ?",(Ticker,))
        elif End == None:
            cursor.execute("SELECT Date, Value FROM Time_Series WHERE Ticker = ? AND Date > ?",(Ticker, Start))
        elif Start == None:
            cursor.execute("SELECT Date, Value FROM Time_Series WHERE Ticker = ? AND Date < ?",(Ticker, End))
        elif (Start != None) & (End != None):
            cursor.execute("SELECT Date, Value FROM Time_Series WHERE Ticker = ? AND Date BETWEEN ? and ?",(Ticker,Start, End))
        else:
            raise ValueError("Timezone Error")
        
        data = cursor.fetchall()
        connection.close()            
        
        if data == []:
            raise ValueError("No such ticker names " + '"' + str(Ticker) + '"' + " in db.")

        Date_list = []
        Value_list = []
        
        for row in data:
            Date_list.append(row[0])
            Value_list.append(row[1])
            
        df = pd.DataFrame(Value_list, index = Date_list, columns = [str(Ticker)])
        df.index = df.index.to_datetime()
        df.sort_index()
        
        return df

    @staticmethod
    def Multi_data_query(Tickers):
        df = pd.DataFrame(index = pd.date_range(start = '1950-01-01',end = datetime.datetime.today().strftime("%Y-%m-%d")),columns = Tickers)

        for each in Tickers:
            df[each] = Controller.Data_query(each)[each]
        return df.dropna(how = 'all')


    @staticmethod
    def Data_insert(Ticker, Name, Series):
        connection = sqlite3.connect("Fin_database.db")
        cursor = connection.cursor()
        
        sys.stderr.write("[info] executing sql for writing data to db \n")
        for i in range(len(Series)):
            cursor.execute("INSERT OR REPLACE INTO Time_Series VALUES (?,?,?,?)",(str(Ticker),str(Name),str(Series.index[i])[:10],float(Series[i])))
        
        connection.commit()
        connection.close()

    @staticmethod
    def Read_excel(filepath, Ticker, Name):
        df = pd.read_excel(filepath,index_col = 0)
        df.index = df.index.to_datetime()
        df = df.dropna()
        
        Controller.Data_insert(Ticker = Ticker, Name = Name, Series = df['Value'])
        
        sys.stderr.write("[info] wrote " + str(len(df['Value'])) + ' rows to db \n')
        sys.stderr.write("[info] " + str(Ticker) + ' saved successfully \n')

    @staticmethod
    def Summarize_data():
        connection = sqlite3.connect("Fin_database.db")
        
        cursor = connection.cursor()
        cursor.execute("SELECT Ticker, Name, MIN(Date), MAX(Date) FROM Time_Series GROUP BY Ticker")
        
        data = cursor.fetchall()
        #connection.close()
        
        for each in data:
            #print each
            cursor.execute("INSERT OR REPLACE INTO Summary_table VALUES (?,?,?,?)",(str(each[0]),str(each[1]),str(each[2]),str(each[3])))
        
        connection.commit()
        connection.close()
#==============================================================================
'''
Controller.Read_excel('D:\Python_Github\Data from XQ\USDCHF.xlsx',
                      'USDCHF:CUR',
                      "USDCHF Spot Exchange Rate - Price of 1 USD in CHF")


data = Controller.Data_query(Ticker = 'CL1:COM')
print data
''' 
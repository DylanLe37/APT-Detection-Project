import numpy as np
import os
import pandas as pd
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
import dask.dataframe as dd
import utils
from pathlib import Path

class dataExploration:
    def __init__(self,dataPath):
        self.dataPath=Path(dataPath)
        self.dataSample = None

    def loadDataSample(self,fileName,fileCols,sampleSize=85000000):#85 million is about the limit of this pc
        self.dataSample = pd.read_csv(self.dataPath/fileName,sep=',',names=fileCols,nrows=sampleSize)
        print(f'Sample uses: {self.dataSample.memory_usage(deep=True).sum()/1024**2:.2f} MB')
        return self.dataSample.head()

    def convertDataCat(self):
        self.dataSample = utils.objectToCat(self.dataSample)
        return self.dataSample.dtypes

    def dataStructure(self):
        print('Basic Info:')
        print(self.dataSample.info())

        print('Things to clean up in each col:') #probably need to account for more than just nulls, like "?" values
        print(self.dataSample.isnull().sum())

        print('Number of unique values by col (excludes nans):')
        for col in self.dataSample.columns:
            unqCol = self.dataSample[col].nunique()
            print(f'Unique {col}s : {unqCol}')

        catColNames = self.dataSample.select_dtypes('category').columns
        for col in catColNames:
            print(f'\n{col} appearance values:')
            print(self.dataSample[col].value_counts().head())
        return

    def temporalAnalysis(self):
        self.dataSample['timeStamp'] = pd.to_datetime(self.dataSample['Time'],unit='s') #no proper date reference so we're in 1970
        self.dataSample['Hour'] = self.dataSample['timeStamp'].dt.hour #these are misaligned but function as labels I guess
        self.dataSample['Day'] = self.dataSample['timeStamp'].dt.day
        self.dataSample['Date'] = self.dataSample['timeStamp'].dt.date

        print(f'\nSample runs from: {self.dataSample['timeStamp'].min()} to {self.dataSample['timeStamp'].max()}')
        print(f'Length of Sample: {self.dataSample['timeStamp'].max() - self.dataSample['timeStamp'].min()}')

        dateEvents = self.dataSample.groupby('Date').size()
        print(f'\nSample has an average : {dateEvents.mean():.1f} events/day')
        print(f'Fewest daily events: {dateEvents.min()} events/day')
        print(f'Most Daily events: {dateEvents.max()} events/day')

        fig,ax = plt.subplots(2,2,figsize=(10,10))

        hourlyEvents = self.dataSample.groupby('Hour').size()
        dailyEvents = self.dataSample.groupby('Day').size()

        ax[0][0].plot(dateEvents.index,dateEvents.values)
        ax[0][0].set_title('Events by Date')
        ax[0][0].set_xlabel('Date')
        ax[0][0].set_ylabel('Number of Events')
        ax[0][0].tick_params('x',rotation=45)

        ax[0][1].bar(hourlyEvents.index,hourlyEvents.values)
        ax[0][1].set_title('Events by Hour')
        ax[0][1].set_xlabel('Hour')
        ax[0][1].set_ylabel('Number of Events')

        ax[1][0].bar(dailyEvents.index,dailyEvents.values)
        ax[1][0].set_title('Events by Day')
        ax[1][0].set_xlabel('Day')
        ax[1][0].set_ylabel('Number of Events')

        if 'Auth Orientation' in self.dataSample.columns:
            authResults = self.dataSample['Auth Orientation'].value_counts()
            ax[1][1].bar(np.arange(0,len(authResults),1),authResults.values,tick_label=authResults.index)
            ax[1][1].set_title('Authentication Events')
            ax[1][1].set_xlabel('Authentication Type')
            ax[1][1].set_ylabel('Number of Events')
            ax[1][1].tick_params('x',labelrotation=45)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        return

fileSet = ['dns.txt','flows.txt','redteam.txt','proc.txt','auth.txt']
colSet = [['Time','Source Comp','Comp Resolved'],
          ['Time','Duration','Source Comp','Source Port','Destination Comp','Destination Port','Protocol','Packet Count','Byte Count'],
          ['Time','User@Domain','Source Comp','Destination Comp'],
          ['Time','User@Domain','Comp','Process Name','Start/End'],
          ['Time','Source User@Domain','Destination User @Domain','Source Comp','Destination Comp','Auth Type','Logon Type','Auth Orientation','Success/Fail']]

dataFolder = '/home/dylan/Documents/APTDetection/Data/'

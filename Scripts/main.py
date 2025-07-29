import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
import utils
from pathlib import Path

class dataExploration:
    def __init__(self,dataPath):
        self.dataPath=Path(dataPath)
        self.dataSample = None

    def loadDataSample(self,fileName,fileCols,sampleSize=50000000):
        self.dataSample = pd.read_csv(self.dataPath/fileName,sep=',',names=fileCols,nrows=sampleSize)
        print(f'Sample uses: {self.dataSample.memory_usage(deep=True).sum()/1024**2:.2f} MB')
        return self.dataSample.head()

    def convertDataCat(self):
        self.dataSample = objectToCat(self.dataSample)
        return self.dataSample.dtypes

    def dataStructure(self):
        print('Basic Info:')
        print(self.dataSample.info())

        print('Things to clean up in each col:')
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

fileSet = ['dns.txt','flows.txt','redteam.txt','proc.txt','auth.txt']
colSet = [['Time','Source Comp','Comp Resolved'],
          ['Time','Duration','Source Comp','Source Port','Destination Comp','Destination Port','Protocol','Packet Count','Byte Count'],
          ['Time','User@Domain','Source Comp','Destination Comp'],
          ['Time','User@Domain','Comp','Process Name','Start/End'],
          ['Time','Source User@Domain','Destination User @Domain','Source Comp','Destination Comp','Auth Type','Logon Type','Auth Orientation','Success/Fail']]

dataFolder = '/home/dylan/Documents/APTDetection/Data/'



# for f in range(len(fileSet)):
#     # filer = pd.read_csv(fileSet[f],sep=",",chunksize=10000,low_memory=False)
#     # filer.columns = colSet[f]
#     filer = pd.read_csv(fileSet[f],sep=",",usecols=[0])
#     fileList.append(filer)
# filer = pd.read_csv(fileSet[2])
# filer.columns = colSet[2]
#
# plt.plot(filer['Time'])
# plt.show(block=True)
print('done')
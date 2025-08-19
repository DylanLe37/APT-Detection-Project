import pandas as pd
import dataExploration as de

def objectToCat(dataFrame):
    return pd.concat([dataFrame.select_dtypes([],['object']),
                      dataFrame.select_dtypes(['object']).apply(pd.Series.astype,dtype='category')]
                     ,axis=1).reindex(dataFrame.columns,axis=1)

def runInitialExploration(dataFolder,dataInd=4,sampleSize=5000000): #need more ram...
    fileSet = ['dns.txt', 'flows.txt', 'redteam.txt', 'proc.txt', 'auth.txt']
    colSet = [['Time', 'Source Comp', 'Comp Resolved'],
              ['Time', 'Duration', 'Source Comp', 'Source Port', 'Destination Comp', 'Destination Port', 'Protocol',
               'Packet Count', 'Byte Count'],
              ['Time', 'User@Domain', 'Source Comp', 'Destination Comp'],
              ['Time', 'User@Domain', 'Comp', 'Process Name', 'Start/End'],
              ['Time', 'Source User@Domain', 'Destination User @Domain', 'Source Comp', 'Destination Comp', 'Auth Type',
               'Logon Type', 'Auth Orientation', 'Success/Fail']]

    dataExp = de.dataExploration(dataFolder)
    dataExp.loadDataSample(fileSet[dataInd],colSet[dataInd],sampleSize)
    dataExp.convertDataCat()
    dataExp.temporalAnalysis()
    userResults = dataExp.userAnalysis()
    privResults = dataExp.privilegeAccounts()
    topoResults = dataExp.networkTopology()
    authResults = dataExp.authAnalysis()
    baselineResults = dataExp.basicBaselines()

    return[{
        'userResults': userResults,
        'privResults': privResults,
        'topoResults': topoResults,
        'authResults': authResults,
        'baselineResults': baselineResults,
    },dataExp.dataSample]

dataFolder = '/home/dylan/Documents/APTDetection/Data/'
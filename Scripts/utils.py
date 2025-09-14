import pandas as pd
from pathlib import Path
import dataExploration as de
import featureExtraction as fe
import detectionModels as dm

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
              ['Time', 'Source User@Domain', 'Destination User@Domain', 'Source Comp', 'Destination Comp', 'Auth Type',
               'Logon Type', 'Auth Orientation', 'Success/Fail']]

    dataExp = de.dataExploration(dataFolder)
    dataExp.loadDataSample(fileSet[dataInd],colSet[dataInd],Path(dataFolder)/fileSet[2],sampleSize)
    dataExp.convertDataCat()
    dataExp.removeServiceAccs()
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

def runFeatureExtraction(explorationResults):
    featureExtractor = fe.featureExtraction(explorationResults[0],explorationResults[1])
    featureData = featureExtractor.temporalFeatures()
    featureData = featureExtractor.sequenceFeatures(featureData)
    featureData = featureExtractor.behavioralFeatures(featureData)
    featureData = featureExtractor.computerAccessFeatures(featureData)
    # featureData = featureExtractor.networkFeatures(featureData)
    featureData = featureExtractor.graphFeatures(featureData)
    featureData = featureExtractor.authenticationFeatures(featureData)
    featureData = featureExtractor.anomalyFeatures(featureData)
    featureVal = featureExtractor.featureValidate(featureData)
    return featureData,featureVal

def buildDetectionModels(featureResults,dataFolder):
    redTeamPath = Path(dataFolder)/'redteam.txt'
    models =['isolationForest','SVM','randomForest']
    detectionModel = dm.detectionModels(featureResults,redTeamPath=redTeamPath)
    detectionModel.groundTruthLabels()
    detectionModel.modelFeatures()
    detectionModel.timeSeriesFeatures()
    detectionModel.trainIsoForest()
    detectionModel.trainSVM()
    detectionModel.trainRandomForest()
    detectionModel.trainLSTM()
    ensemblePreds,ensembleScores = detectionModel.ensembleModel()

    return {
        'detectionModel': detectionModel,
        'models': detectionModel.models,
        'performanceMetrics': detectionModel.performanceMetrics,
        'ensemblePreds':ensemblePreds,
        'ensembleScores':ensembleScores
    }


dataFolder = '/home/dylan/Documents/APTDetection/Data/'
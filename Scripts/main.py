import pandas as pd
from pathlib import Path
import dataExploration as de
import featureExtraction as fe
import detectionModels as dm

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
    # featureData = featureExtractor.networkFeatures(featureData) #memory issue here, probably need to do more optimization...
    featureData = featureExtractor.graphFeatures(featureData)
    featureData = featureExtractor.authenticationFeatures(featureData)
    featureData = featureExtractor.anomalyFeatures(featureData)
    featureVal = featureExtractor.featureValidate(featureData)
    return featureData,featureVal

def buildDetectionModels(featureData,dataFolder):
    redTeamPath = Path(dataFolder)/'redteam.txt'
    models = ['isolationForest','randomForest']
    detectionModel = dm.detectionModels(featureData,redTeamPath=redTeamPath)
    detectionModel.groundTruthLabels()
    detectionModel.modelFeatures()
    detectionModel.timeSeriesFeatures()
    detectionModel.trainIsoForest()
    #detectionModel.trainSVM() #SVM takes forever with this many data samples, but it does work f desired
    detectionModel.trainRandomForest()
    detectionModel.trainLSTM()
    ensemblePreds,ensembleScores = detectionModel.ensembleModel(models) #don't add LSTM to ensemble model since it's sequence based unlike the others

    return {
        'detectionModel': detectionModel,
        'models': detectionModel.models,
        'performanceMetrics': detectionModel.performanceMetrics,
        'ensemblePreds':ensemblePreds,
        'ensembleScores':ensembleScores
    }


dataFolder = '/path/to/data/here/'
results = runInitialExploration(dataFolder,sampleSize=3500000)
featureData,featureValidation = runFeatureExtraction(results)

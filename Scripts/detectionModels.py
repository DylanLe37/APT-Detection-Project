import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve,precision_score,accuracy_score,recall_score,f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class detectionModels:
    def __init__(self,featureData,redTeamPath=None):
        self.featureData = featureData
        self.redTeamPath = redTeamPath
        self.models = {}
        self.scalers = {}
        self.performanceMetrics = {}

    def groundTruthLabels(self):
        print('Generating ground truth attack labels')
        redTeamData = pd.read_csv(
            self.redTeamPath,
            names = ['Time', 'User@Domain', 'Source Comp', 'Destination Comp']
        )

        self.featureData['isAttack'] = 0
        self.featureData['attackSeq'] = 0
        attackTimes = redTeamData['Time'].values

        for attackTime in attackTimes:
            attackMask = (
                (self.featureData['Time'] >= attackTime - 1800) &
                (self.featureData['Time'] <= attackTime + 1800)
            )
            self.featureData.loc[attackMask,'isAttack'] = 1

            seqMask = (
                (self.featureData['Time'] >= attackTime - 1800) &
                (self.featureData['Time'] <= attackTime + 1800)
            )
            self.featureData.loc[seqMask,'attackSeq'] = 1

        seqCount = self.featureData['attackSeq'].sum()
        print(f'Attack sequences: {seqCount:,} ({seqCount / len(self.featureData):.4f}%)')
        return

    def modelFeatures(self,testSize = 0.3):
        print('Generating model features')

        numCols = self.featureData.select_dtypes(include=[np.number]).columns
        excludeSet = ['isAttack','attackSeq','Time','timeStamp','Source User@Domain','Destination Comp','Source Comp',
                      'Day','timeStamp','Date','Hour','authsPerHour','authsPerDay','authsPerMin','workHours','newUsersPerMin',
                      'userAuthsPerDay','userAuthsPerHour','riskCurrentHour','riskCurrentMin'] #drop temporal stuff mostly
        includeSet = [feature for feature in numCols
                      if not feature in excludeSet]

        featureMat = self.featureData[includeSet]
        featureMat = featureMat.replace([np.inf, -np.inf], np.nan)
        featureMat = featureMat.fillna(featureMat.median())

        for col in featureMat.columns:
            if featureMat[col].dtype == 'float64':
                featureMat[col] = featureMat[col].astype('float32')

        attackLabels = self.featureData['isAttack'].values
        print(f'Class Imbalance (normal:attack) : {(1-attackLabels.mean())/attackLabels.mean():.1f}:1')

        self.modelData = featureMat
        self.modelLabels = attackLabels
        self.modelCols = includeSet

        dataTrain, dataTest, labelTrain, labelTest = train_test_split(
            self.modelData, self.modelLabels, test_size = testSize, stratify=self.modelLabels, random_state = 2025
        )

        self.dataTrain = dataTrain
        self.dataTest = dataTest
        self.labelTrain = labelTrain
        self.labelTest = labelTest

        return

    # def timeSeriesFeatures(self,seqLength = 10, maxUsers = 1000, testSize = 0.3):
    #     print('Generating time series features')
    #
    #     userCount = self.featureData['Source User@Domain'].value_counts()
    #     topUsers = userCount.head(maxUsers).index
    #
    #     filteredData = self.featureData[self.featureData['Source User@Domain'].isin(topUsers)].copy()
    #     sortedData = filteredData.sort_values(['Source User@Domain','Time']).reset_index(drop=True)
    #
    #     seq = []
    #     labels = []
    #     userSeq = 0
    #
    #     for user in topUsers:
    #         userData = sortedData[sortedData['Source User@Domain'] == user]
    #
    #         if len(userData) < seqLength:
    #             continue
    #
    #         userFeatures = userData[self.modelCols].values
    #         userLabels = userData['isAttack'].values
    #
    #         maxSeqPerUser = 100
    #         tau = max(1,len(userData) // maxSeqPerUser)
    #
    #         for i in range(0,len(userData)-seqLength+1,tau):
    #             seq.append(userFeatures[i:i+seqLength])
    #             labels.append(int(userLabels[i:i+seqLength].max()))
    #
    #         userSeq+=1
    #         if userSeq % 100==0:
    #             print(f'Processed {userSeq} users')
    #
    #     if seq:
    #         seqData = np.array(seq,dtype=np.float32)
    #         seqLabels = np.array(labels)
    #
    #         self.seqData = seqData
    #         self.seqLabels = seqLabels
    #
    #         seqTrain, seqTest, seqLabelsTrain, seqLabelsTest = train_test_split(
    #             self.seqData,self.seqLabels,test_size = testSize, stratify=self.seqLabels, random_state = 2025
    #         )
    #
    #         self.seqTrain = seqTrain
    #         self.seqTest = seqTest
    #         self.seqLabelsTrain = seqLabelsTrain
    #         self.seqLabelsTest = seqLabelsTest
    #     return

    def timeSeriesFeatures(self,seqLength=10,targetSequences=50000,attackRatio=0.15,testSize=0.3):
        print('Generating time series features')

        sortedData = self.featureData.sort_values(['Source User@Domain','Time']).reset_index(drop=True)

        targetAttackSeqs = int(targetSequences*attackRatio)
        targetNormalSeqs = targetSequences-targetAttackSeqs

        attackSeqs = []
        attackLabels = []

        attackUserDat = sortedData[sortedData['isAttack']==1].groupby('Source User@Domain')

        for user,userAttacks in attackUserDat:
            userData = sortedData[sortedData['Source User@Domain']==user]
            attackInds = userData[userData['isAttack']==1].index.tolist()
            for attackID in attackInds:
                userPos = userData.index.tolist()
                if attackID in userPos:
                    pos = userPos.index(attackID)
                    startPos = max(0,pos-seqLength+1)
                    endPos = startPos+seqLength
                    if endPos<=len(userData):
                        seqDat = userData.iloc[startPos:endPos]
                        if len(seqDat)==seqLength:
                            seqFeatures = seqDat[self.modelCols].values
                            seqLabel = int(seqDat['isAttack'].max())

                            attackSeqs.append(seqFeatures)
                            attackLabels.append(seqLabel)
            if len(attackSeqs)>=targetAttackSeqs:
                break

        normalSeqs = []
        normalLabels = []

        userAttackCounts = sortedData.groupby('Source User@Domain')['isAttack'].sum()
        normalUsers = userAttackCounts[userAttackCounts==0].index.tolist()

        np.random.seed(2025)
        sampledUsers = np.random.choice(
            normalUsers,
            size=min(5000,len(normalUsers)),
            replace=False
        )

        seqsPerUser = targetNormalSeqs//len(sampledUsers)

        for user in sampledUsers:
            userData = sortedData[sortedData['Source User@Domain']==user]
            # normalUserData = userData[userData['isAttack']==0]

            # maxSeqs = min(seqsPerUser,(len(normalUserData)-seqLength+1))

            # if maxSeqs>0:
            #     startPositions = np.random.choice(
            #         len(normalUserData)-seqLength+1,
            #         size=min(maxSeqs,20),
            #         replace=False
            #     )

                for startPos in startPositions:
                    seqData = normalUserData.iloc[startPos:startPos+seqLength]

                    if len(seqData)==seqLength:
                        seqFeatures = seqData[self.modelCols].values
                        seqLabel = 0
                        normalSeqs.append(seqFeatures)
                        normalLabels.append(seqLabel)

            if len(normalSeqs)>=targetNormalSeqs:
                break

        if len(attackSeqs)>targetAttackSeqs:
            inds = np.random.choice(
                len(attackSeqs),
                targetAttackSeqs,
                replace=False
            )
            attackSeqs = [attackSeqs[i] for i in inds]
            attackLabels = [attackLabels[i] for i in inds]

        if len(normalSeqs)>targetNormalSeqs:
            inds = np.random.choice(
                len(normalSeqs),
                targetNormalSeqs,
                replace=False
            )
            normalSeqs = normalSeqs[inds]
            normalLabels = normalLabels[inds]

        allSeqs = attackSeqs+normalSeqs
        allLabels = attackLabels+normalLabels

        seqData = np.array(allSeqs,dtype='float32')
        seqLabels = np.array(allLabels)

        print(f'Total Sequences:{len(allSeqs):,}')
        print(f'Attack Sequences:{seqLabels.sum():,} ({seqLabels.mean()*100:.2f}%)')
        print(f'Normal Sequences:{(seqLabels==0).sum():,} ({(1-seqLabels.mean())*100:.2f}%)')

        self.seqData = seqData
        self.seqLabels = seqLabels

        seqTrain,seqTest,seqLabelsTrain,seqLabelsTest = train_test_split(
            seqData,seqLabels,
            testSize=testSize,
            stratify=seqLabels,
            random_state=2025
        )

        self.seqTrain = seqTrain
        self.seqTest = seqTest
        self.seqLabelsTrain = seqLabelsTrain
        self.seqLabelsTest = seqLabelsTest

        return

    def modelPerformance(self,modelName,groundTruth,preds,predScores=None,modelType='unsupervised'):

        precision = precision_score(groundTruth,preds,zero_division=0)
        recall = recall_score(groundTruth,preds,zero_division=0)
        f1 = f1_score(groundTruth,preds,zero_division=0)

        metrics = {'precision':precision,'recall':recall,'f1':f1}

        if modelType == 'supervised':
            metrics['accuracy'] = accuracy_score(groundTruth,preds)
            metrics['confMat'] = confusion_matrix(groundTruth,preds)

        if predScores is not None:
            metrics['auc'] = roc_auc_score(groundTruth,predScores)
        else:
            metrics['auc'] = 0.5

        self.performanceMetrics[modelName] = metrics

    def trainIsoForest(self,contamination='auto',estimators = 50):
        print('Training isolation forest')

        scaler = StandardScaler()
        scaledTrainData = scaler.fit_transform(self.dataTrain)
        scaledTestData = scaler.transform(self.dataTest)

        contamination = max(0.001,min(0.5,self.labelTrain.mean()*2))

        startTime = time.time()

        isoForest = IsolationForest(
            contamination = contamination,
            n_estimators = estimators,
            random_state = 2025,
            n_jobs = -1,
            max_samples='auto'
        )
        isoForest.fit(scaledTrainData)
        trainTime = time.time()-startTime

        print(f'Isolation forest training done in {trainTime:.2f} seconds')

        preds = isoForest.predict(scaledTestData)
        preds = (preds==-1).astype(int)
        scores = isoForest.score_samples(scaledTestData)

        self.modelPerformance('isolationForest',self.labelTest,preds,-scores,modelType='unsupervised')
        self.models['isolationForest'] = isoForest
        self.scalers['isolationForest'] = scaler

        #performing kinda bad? maybe data issue
        return

    def trainSVM(self,nu='auto',gamma='scale'):
        print('Training SVM')

        scaler = RobustScaler()

        scaledTrainData = scaler.fit_transform(self.dataTrain)
        scaledTestData = scaler.transform(self.dataTest)

        if nu == 'auto':
            nu = max(0.01,min(0.1,self.labelTrain.mean()*3))

        startTime = time.time()
        svm = OneClassSVM(
            nu = nu,
            gamma = gamma,
            kernel = 'rbf'
        )

        svm.fit(scaledTrainData)
        trainingTime = time.time()-startTime
        print(f'SVM training done in {trainingTime:.2f} seconds')

        preds = svm.predict(scaledTestData)
        scores = svm.decision_function(scaledTestData)

        preds = (preds==-1).astype(int)

        self.modelPerformance('SVM',self.labelTest,preds,-scores,modelType='unsupervised')
        self.models['SVM'] = svm
        self.scalers['SVM'] = scaler

        #run while out, see how long takes, not finishing that fast, maybe need to subsample
        #definitely need to subsample
        return

    def trainRandomForest(self,estimators = 50,depth=10,weights='balanced'):
        print('Training random forest')

        scaler = StandardScaler()
        scaledTrainData = scaler.fit_transform(self.dataTrain)
        scaledTestData = scaler.transform(self.dataTest)

        startTime = time.time()
        randomForest = RandomForestClassifier(
            n_estimators = estimators,
            max_depth = depth,
            class_weight = weights,
            random_state = 2025,
            n_jobs = -1,
            max_features = 'sqrt'
        )

        randomForest.fit(scaledTrainData,self.labelTrain)
        trainingTime = time.time()-startTime
        print(f'Random forest training done in {trainingTime:.2f} seconds')

        preds = randomForest.predict(scaledTestData)
        predProb = randomForest.predict_proba(scaledTestData)[:,1]

        self.modelPerformance('Random Forest',self.labelTest,preds,predProb,modelType='supervised')

        self.models['randomForest'] = randomForest
        self.scalers['randomForest'] = scaler
        return

    def trainLSTM(self,units=32,epochs=20,batchSize=64,dropout=0.3):
        print('Training LSTM')

        scaler = StandardScaler()

        sampleCount,timeStepCount,featureCount = self.seqTrain.shape

        dataTrainReshape = self.seqTrain.reshape(-1,featureCount)
        scaledTrainData = scaler.fit_transform(dataTrainReshape)
        scaledTrainData = scaledTrainData.reshape(sampleCount,timeStepCount,featureCount)

        sampleCountTest = self.seqTest.shape[0]
        dataTestReshape = self.seqTest.reshape(-1,featureCount)
        scaledTestData = scaler.transform(dataTestReshape)
        scaledTestData = scaledTestData.reshape(sampleCountTest,timeStepCount,featureCount)

        lstmModel = Sequential([
            LSTM(units,input_shape=(timeStepCount,featureCount)),
            Dropout(dropout),
            Dense(units//2,activation='relu'),
            Dropout(dropout),
            Dense(1,activation='sigmoid')
        ])
        lstmModel.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        classWeights = {
            0:1,
            1:((len(self.seqLabelsTrain)-self.seqLabelsTrain.sum())/self.seqLabelsTrain.sum() if self.seqLabelsTrain.sum()>0 else 1)
        }

        startTime = time.time()

        earlyStopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = lstmModel.fit(
            scaledTrainData,self.seqLabelsTrain,
            validation_data = (scaledTestData,self.seqLabelsTest),
            epochs = epochs,
            batch_size = batchSize,
            class_weight = classWeights,
            callbacks = [earlyStopping],
            verbose = 0
        )

        trainingTime = time.time()-startTime
        print(f'LSTM training done in {trainingTime:.2f} seconds')

        predProb = lstmModel.predict(scaledTestData).flatten()
        preds = (predProb>0.5).astype(int)

        self.modelPerformance('LSTM',self.seqLabelsTest,preds,predProb,modelType='supervised')

        self.models['LSTM'] = lstmModel
        self.scalers['LSTM'] = scaler

        return

    def ensembleModel(self,models):
        print('Creating ensemble model')

        ensemblePreds = {}
        for modelName in models:
            model = self.models[modelName]
            scaler = self.scalers[modelName]

            scaledTestData = scaler.transform(self.dataTest)

            if modelName == 'isolationForest':
                preds = model.predict(scaledTestData)
                scores = model.score_samples(scaledTestData)
                ensemblePreds[modelName] = {
                    'binary':(preds==-1).astype(int),
                    'scores':-scores
                }

            elif modelName == 'SVM':
                preds = model.predict(scaledTestData)
                scores = model.decision_function(scaledTestData)
                ensemblePreds[modelName] = {
                    'binary': (preds == -1).astype(int),
                    'scores': -scores
                }

            elif modelName == 'randomForest':
                binaryPreds = model.predict(scaledTestData)
                probPreds = model.predict_proba(scaledTestData)[:,1]
                ensemblePreds[modelName] = {
                    'binary': binaryPreds,
                    'scores': probPreds
                }

        binaryPreds = np.column_stack([
            ensemblePreds[m]['binary'] for m in models
        ])
        scorePreds = np.column_stack([
            ensemblePreds[m]['scores'] for m in models
        ])

        ensembleBinary = (binaryPreds.mean(axis=1)>0.5).astype(int)
        ensembleScores = scorePreds.mean(axis=1)

        self.modelPerformance('Ensemble',self.labelTest,ensembleBinary,ensembleScores,'unsupervised')

        return ensembleBinary, ensembleScores



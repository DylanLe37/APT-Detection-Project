import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
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

class detectionModel:
    def __init__(self,featureData,redTeamPath=None):
        self.featureData = featureData.copy()
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
                (self.featureData['Time'] >= attackTime - 300) &
                (self.featureData['Time'] <= attackTime + 300)
            )
            self.featureData.loc[attackMask,'isAttack'] = 1

            seqMask = (
                (self.featureData['Time'] >= attackTime - 900) &
                (self.featureData['Time'] <= attackTime + 900)
            )
            self.featureData.loc[seqMask,'attackSeq'] = 1

        seqCount = self.featureData['attackSeq'].sum()
        print(f'Attack sequences: {seqCount:,} ({seqCount / len(self.featureData):.4f}%)')
        return

    def modelFeatures(self):
        print('Generating model features')

        numCols = self.featureData.select_dtypes(include=[np.number]).columns
        excludeSet = ['isAttack','attackSeq','Time','timeStamp','Source User@Domain','Destination Comp','Source Comp']
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

        return

    def timeSeriesFeatures(self,seqLength = 10, maxUsers = 1000):
        print('Generating time series features')

        userCount = self.featureData['Source User@Domain'].value_counts()
        topUsers = userCount.head(maxUsers).index

        filteredData = self.featureData[self.featureData['Source User@Domain'].isin(topUsers)].copy()
        sortedData = filteredData.sort_values(['Source User@Domain','Time']).reset_index(drop=True)

        seq = []
        labels = []
        userSeq = 0

        for user in topUsers:
            userData = sortedData[sortedData['Source User@Domain'] == user]

            if len(userData) < seqLength:
                continue

            userFeatures = userData[self.modelCols].values
            userLabels = userData['isAttack'].values

            maxSeqPerUser = 100
            tau = max(1,len(userData) // maxSeqPerUser)

            for i in range(0,len(userData)-seqLength+1,tau):
                seq.append(userFeatures[i:i+seqLength])
                labels.append(int(userLabels[i:i+seqLength].max()))

            userSeq+=1
            if userSeq % 100==0:
                print(f'Processed {userSeq} users')

        if seq:
            seqData = np.array(seq,dtype=np.float32)
            seqLabels = np.array(labels)

            self.seqData = seqData
            self.seqLabels = seqLabels

        return

    def isolationForest(self,contamination='auto',estimators = 50):
        return
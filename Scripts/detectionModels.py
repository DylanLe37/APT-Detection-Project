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
        return self.featureData
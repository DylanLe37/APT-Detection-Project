import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class featureExtraction:
    def __init__(self,baseLines,dataSample):
        self.baselines = baseLines
        self.dataSample = dataSample
        self.features={}
        self.scalars={}

    def temporalFeatures(self):
        print('Extracting temporal features...')
        tempFeatures = self.dataSample.copy()

        peakHours = self.baselines['baselineResults']['temporal']['hourlyPattern'].nlargest(8).index
        tempFeatures['workHours'] = tempFeatures['Hour'].isin(peakHours).astype(int)
        tempFeatures = tempFeatures.sort_values(['Source User@Domain','timeStamp'])
        tempFeatures['timeSinceLastAuth'] = tempFeatures.groupby('Source User@Domain')['timeStamp'].diff().dt.total_seconds()

        tempFeatures['sessionID'] = (
            tempFeatures.groupby('Source User@Domain')['timeSinceLastAuth']
            .transform(lambda x: (x>1800).cumsum())
        )

        tempFeatures['minute'] = tempFeatures['timeStamp'].dt.floor('1min')
        timeAuths = tempFeatures.groupby('minute').size().reset_index()
        timeAuths.columns = ['minute','authsPerMin']
        timeAuths = timeAuths.sort_values('minute')

        timeAuths['authsPerHour'] = (
            timeAuths['authsPerMin'].rolling(window=60,min_periods=1)
            .sum()
        )

        timeAuths['authsPerDay'] = (
            timeAuths['authsPerMin'].rolling(window=1440,min_periods=1)
            .sum()
        )

        tempFeatures = tempFeatures.merge(
            timeAuths[['minute','authsPerHour','authsPerDay']],
            on='minute',
            how='left'
        )

        userAuths = tempFeatures.groupby(['Source User@Domain','minute']).size().reset_index()
        userAuths.columns = ['Source User@Domain','minute','userAuthsPerMin']
        userAuths = userAuths.sort_values(['Source User@Domain','minute'])

        userRolling = userAuths.groupby('Source User@Domain').apply(
            lambda group: group.assign(
                userAuthsPerHour = group['userAuthsPerMin'].rolling(60,min_periods=1).sum(),
                userAuthsPerDay = group['userAuthsPerMin'].rolling(1440,min_periods=1).sum()
            )
        ).reset_index(drop=True)

        tempFeatures = tempFeatures.merge(
            userRolling[['Source User@Domain','minute','userAuthsPerHour','userAuthsPerDay']],
            on=['Source User@Domain','minute'],
            how='left'
        )
        tempFeatures = tempFeatures.drop('minute',axis=1)

        userAvgHours = self.baselines['baselineResults']['users']['hourMean'].to_dict()
        tempFeatures['userHourDeviation'] = tempFeatures.apply(
            lambda row: abs(row['Hour']-userAvgHours.get(row['Source User@Domain'], 12))
            ,axis=1
        )
        print('Temporal features extracted')
        return tempFeatures

    def sequenceFeatures(self,featureData):
        print('Extracting sequence features...')
        seqFeatures = featureData.copy()
        seqFeatures = seqFeatures.sort_values(['Source User@Domain','timeStamp'])
        seqFeatures['authSeqNum'] = seqFeatures.groupby('Source User@Domain').cumcount()+1

        seqFeatures['prevDestComp'] = seqFeatures.groupby('Source User@Domain')['Destination Comp'].shift(1)
        seqFeatures['prevAuthType'] = seqFeatures.groupby('Source User@Domain')['Auth Type'].shift(1)

        seqFeatures['destinationChange'] = (
            seqFeatures['Destination Comp'] != seqFeatures['prevDestComp']
        ).astype(int)

        seqFeatures['rapidSeq'] = (
            seqFeatures['timeSinceLastAuth'] < 300
        ).astype(int)

        def calcRollingUniqueDests(userSet,windows=[5,10,20]):
            comps = userSet['Destination Comp'].tolist()
            unqDests = {f'uniqueDestLast{w}':[] for w in windows}

            for i in range(len(comps)):
                for window in windows:
                    startInd = max(0,i-window+1)
                    windowComps = comps[startInd:i+1]
                    unqDests[f'uniqueDestLast{window}'].append(len(set(windowComps)))
            return pd.DataFrame(unqDests,index=userSet.index)

        unqRollResults = seqFeatures.groupby('Source User@Domain').apply(
            calcRollingUniqueDests
        )

        for window in [5,10,20]:
            seqFeatures[f'uniqueDestLast{window}'] = (
                unqRollResults[f'uniqueDestLast{window}']
                .reset_index(level=0,drop=True)
            )
        print('Sequence features extracted')
        return seqFeatures
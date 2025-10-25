import numpy as np
import os
import pandas as pd
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
import dask.dataframe as dd
from pathlib import Path

def objectToCat(dataFrame):
    return pd.concat([dataFrame.select_dtypes([],['object']),
                      dataFrame.select_dtypes(['object']).apply(pd.Series.astype,dtype='category')]
                     ,axis=1).reindex(dataFrame.columns,axis=1)

class dataExploration:
    def __init__(self,dataPath):
        self.dataPath=Path(dataPath)
        self.dataSample = None

    def loadDataSample(self, fileName, fileCols, attackPath, sampleSize=8500000, attackRatio=0.05, dayCount=21):
        print('Loading data...')
        redTeamData = pd.read_csv(attackPath,sep=',',names=['Time', 'User@Domain', 'Source Comp', 'Destination Comp'])

        attackStart = redTeamData['Time'].min()
        attackEnd = redTeamData['Time'].max()
        attackTimes = set(redTeamData['Time'].values)

        attackTarget = int(sampleSize*attackRatio)
        normTarget = sampleSize-attackTarget

        attackEvents =[]
        attacksFound = 0

        for chunkID,chunk in enumerate(pd.read_csv(self.dataPath/fileName,sep=',',names=fileCols,chunksize=500000)):
            chunkAttacks = chunk[chunk['Time'].isin(attackTimes)]
            if len(chunkAttacks)>0:
                attackEvents.append(chunkAttacks)
                attacksFound+=len(chunkAttacks)

            if chunk['Time'].min()>attackEnd:
                break

        sessionStart = 1
        sessionEnd = dayCount*24*3600
        binWidth = 6*3600
        binCount = (sessionEnd-sessionStart)//binWidth
        eventDensity = normTarget//binCount

        normEvents = []
        binEvents ={i:0 for i in range(binCount)}

        for chunk in pd.read_csv(self.dataPath/fileName,sep=',',names=fileCols,chunksize=500000):#takes mad long bc sample many days
            normChunk = chunk[~chunk['Time'].isin(attackTimes)]

            if len(normChunk)>0:
                normChunkCopy = normChunk.copy()
                normChunkCopy['Bin'] = ((normChunkCopy['Time']-sessionStart)//binWidth).astype(int)

                for binID in normChunkCopy['Bin'].unique():
                    if binID in binEvents and binEvents[binID]<eventDensity:
                        binDat = normChunkCopy[normChunkCopy['Bin'] == binID]
                        needed = eventDensity-binEvents[binID]

                        if len(binDat)<= needed:
                            sampledBin = binDat.drop('Bin',axis=1)
                            normEvents.append(sampledBin)
                            binEvents[binID]+=len(sampledBin)
                        else:
                            sampledBin = binDat.sample(n=needed,random_state=2025).drop('Bin',axis=1)
                            normEvents.append(sampledBin)
                            binEvents[binID] = eventDensity

            normTotal = sum(len(chunk) for chunk in normEvents)
            if normTotal>=normTarget:
                break

        allAttacks = pd.concat(attackEvents,ignore_index=True)
        if len(allAttacks)>attackTarget:
            attackSamps = allAttacks.sample(n=attackTarget,random_state=2025)
        else:
            attackSamps = allAttacks

        allNorm = pd.concat(normEvents,ignore_index=True)
        if len(allNorm)>normTarget:
            normSamps = allNorm.sample(n=normTarget,random_state=2025)
        else:
            normSamps = allNorm

        self.dataSample = pd.concat([attackSamps,normSamps],ignore_index=True)
        self.dataSample = self.dataSample.sort_values('Time').reset_index(drop=True)

        fullTimeRange = self.dataSample['Time'].max() - self.dataSample['Time'].min()
        print(f'Total events: {len(self.dataSample):,}')
        print(f'\n Attack Events: {len(attackSamps):,} ({len(attackSamps)/len(self.dataSample):,})')
        print(f'\n Time span: {fullTimeRange/86400:.1f} days')


        return self.dataSample.head()

    def convertDataCat(self):
        self.dataSample = objectToCat(self.dataSample)
        return self.dataSample.dtypes

    def removeServiceAccs(self):
        userFilter = (
            self.dataSample['Source User@Domain'].str.startswith('U', na=False) &
            ~self.dataSample['Source User@Domain'].str.contains(r'\$', na=False) &
            ~self.dataSample['Source User@Domain'].str.contains(
                r'SYSTEM@|ANONYMOUS|LOCAL SERVICE', na=False, case=False
            )
        )

        self.dataSample = self.dataSample[userFilter].reset_index(drop=True)

        return self.dataSample.head()

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

        ax[1][0].bar(dailyEvents.index,dailyEvents.values) #memory constraints mean length of sample is not a full week, this may be useless
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

    def userAnalysis(self): #just for auth data
        userStats = self.dataSample.groupby('Source User@Domain').agg(
            {'timeStamp':['count','min','max'],'Destination Comp':'nunique','Source Comp':'nunique'}).round(2)
        userStats.columns = ['totalAuths','firstAuth','lastAuth','uniqueDests','uniqueSources']

        print(f'\n Total unique users:{self.dataSample['Source User@Domain'].nunique():,}')
        print(f'Total unique computers:{self.dataSample['Destination Comp'].nunique():,}')

        topUsers = userStats.sort_values('totalAuths',ascending=False).head(10)
        print(f'Top 10 users:\n{topUsers[['totalAuths','uniqueSources','uniqueDests']]}')

        plt.figure()
        plt.hist(userStats['totalAuths'],100,range=(0,userStats['totalAuths'].max()))
        plt.title(f'Auth Events by User in Sample, median = {userStats['totalAuths'].median():.0f}')
        plt.xlabel('Total Auth Events')
        plt.ylabel('Counts')

        compStats = self.dataSample.groupby('Destination Comp').agg(
            {'Source User@Domain':'nunique','timeStamp':'count'})
        compStats.columns = ['uniqueUsers','accessCount']

        plt.figure()
        plt.hist(np.log1p(compStats['uniqueUsers']),100)
        plt.title('Computer User Counts (log-scale)')
        plt.xlabel('User Counts (log-scale)')
        plt.ylabel('Count')
        return userStats,compStats

    def privilegeAccounts(self):
        userCompAccess = self.dataSample.groupby('Source User@Domain')['Destination Comp'].nunique()
        highAccessUsers = userCompAccess[userCompAccess > userCompAccess.quantile(0.95)]

        print(f'Users with high access permissions: \n{highAccessUsers.sort_values(ascending=False).head(10)}')

        serviceAccounts = self.dataSample['Source User@Domain'].str.contains(r'\$$',na=False)
        print(f'\nSuspected service accounts: {serviceAccounts.sum():,}')

        hourlyAct = self.dataSample.groupby('Hour').size()
        lowActivityThresh = hourlyAct.quantile(0.25)
        offHours = hourlyAct[hourlyAct<=lowActivityThresh].index
        offHourMask = self.dataSample['Hour'].isin(offHours)
        offHourUsers = self.dataSample[offHourMask]['Source User@Domain'].value_counts()

        plt.figure(figsize=(10,10))
        plt.bar(np.arange(0, 10, 1), offHourUsers.head(10).values, tick_label=offHourUsers.head(10).index)
        plt.title('Top Off Hour Users')
        plt.xlabel('Users')
        plt.ylabel('Number of Events')
        plt.xticks(np.arange(0,10,1),rotation=-45,fontsize='xx-small')

        return highAccessUsers,offHourUsers

    def networkTopology(self):
        compConnections = self.dataSample.groupby(['Source Comp','Destination Comp']).size()
        print(f'\nMost common connections:\n{compConnections.sort_values(ascending=False).head(10)}') #have to remove self connections...

        sourceComps = set(self.dataSample['Source Comp'].unique())
        destinationComps =set(self.dataSample['Destination Comp'].unique())

        print(f'Source Comps: {len(sourceComps-destinationComps):,}')
        print(f'Destination Comps: {len(destinationComps-sourceComps):,}')
        print(f'Comps in Both: {len(sourceComps & destinationComps):,}')

        connectionsIn = self.dataSample.groupby('Destination Comp')['Source Comp'].nunique()
        likelyServers = connectionsIn[connectionsIn >= connectionsIn.quantile(0.95)]

        print(f'Potential Servers:\n{likelyServers.sort_values(ascending=False).head(10)}')

        return compConnections,likelyServers

    def authAnalysis(self):
        if 'Auth Type' in self.dataSample.columns:
            authTypes = self.dataSample['Auth Type'].value_counts()
            plt.figure(figsize=(5,5))
            plt.bar(np.arange(0,len(authTypes),1),authTypes.values,tick_label=authTypes.index)
            plt.xticks(rotation=90,fontsize=8)
            plt.tight_layout()

        typeSuccess = self.dataSample.groupby('Auth Type')['Success/Fail'].apply(
            lambda x: (x=='Success').mean() if 'Success' in x.values else 0
        )

        if 'Logon Type' in self.dataSample.columns:
            logOnTypes = self.dataSample['Logon Type'].value_counts()
            plt.figure(figsize=(5, 5))
            plt.bar(np.arange(0, len(logOnTypes), 1), logOnTypes.values, tick_label=logOnTypes.index)
            plt.xticks(rotation=90, fontsize=8)
            plt.tight_layout()

        if 'Success/Fail' in self.dataSample.columns:
            failedAuths = self.dataSample[self.dataSample['Success/Fail']!='Success']
            print(f'Failed Auth Rate: {len(failedAuths)/len(self.dataSample):.2f}')

            userFails = failedAuths['Source User@Domain'].value_counts()
            print(f'\nUsers with most fails:\n{userFails.head(10)}')

        return authTypes,logOnTypes,failedAuths

    def basicBaselines(self):
        baseLines = {}

        userBaselines = self.dataSample.groupby('Source User@Domain').agg({
            'Hour':['mean','std'],
            'Day':'mean',
            'Destination Comp':'nunique',
            'timeStamp':'count'
        })
        userBaselines.columns = ['hourMean','hourStd','dayMean','uniqueComps','totalAuths']
        print(f'\nBasic user baselines:\n{userBaselines.describe()}')
        baseLines['users'] = userBaselines

        compBaselines = self.dataSample.groupby('Destination Comp').agg({
            'Source User@Domain':'nunique',
            'Hour':['mean','std'],
            'timeStamp':'count'
        })
        compBaselines.columns = ['uniqueUsers','hourMean','hourStd','totalAuths']
        print(f'\nBasic comp baselines:\n{compBaselines.describe()}')
        baseLines['comps'] = compBaselines

        hourlyBaselines = self.dataSample.groupby('Hour').size()
        dailyBaselines = self.dataSample.groupby('Day').size()

        baseLines['temporal'] = {
            'hourlyPattern':hourlyBaselines,
            'dailyPattern':dailyBaselines.mean(),
            'dailyStd':dailyBaselines.std()
        }

        return baseLines
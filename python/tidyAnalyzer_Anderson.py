
# coding: utf-8

# In[2]:

# load tidy dataset

import pandas as pd
import os.path
import numpy as np
from itertools import combinations
import glob

#import numpy as np
#import loadTidyCSV.py

def loadTidy(tidyData): 
    assert os.path.isfile(tidyData), "desired file does not exist" 
    df = pd.read_csv(tidyData, header = None, names = ["date", "animal", "session", "trial", "stimulus", "neuronID", "timePt", "CaSignal"])
    return df 

# tidy csv file and dir (use makeTidy_Anderson.m to convert Ann's structure to csv)
tidyDataDir = '/home/orthogonull/a_MHR/a_research/a_gitResearch/git_ignored/imagingAnalysis/data/2_tidyFormat/'
tidyDataFileTemplate = 'mouse'
tidyDataFileExt = '.csv'

# get all input files you want to add to the same dataset
dataFiles = np.sort(glob.glob(    "/home/orthogonull/a_MHR/a_research/a_gitResearch/git_ignored/imagingAnalysis/data/2_tidyFormat/mouse*.csv"))
print("data files: \n", dataFiles)


print("\n loading and appending to prior pandas data frame")
dataLst = []
for file in dataFiles:
    print(file)
    dataLst.append(loadTidy(file))
df = pd.concat(dataLst)

print('finished loading')

############# ALL DATA STORED HERE IN DF
    


# In[3]:

## survey/search data to prepare for split operation
metaStrs = [['dates','date'],['animals','animal'],['sessions','session'],['maxTrials','trial'],['stimuli','stimulus']]

## this dictionary holds useful info regarding the range of inputs to loop/search over subsequently
metaDct = {}
for a,b in metaStrs:
    print(a,b)
    metaDct[a] = np.unique(df[b].tolist())
print(metaDct)


# In[24]:

######### PARAMETERS #########
threshTPs_stdFromMean = 1



# In[26]:

### get all pairs of stimuli
stimCmbTpl = tuple(combinations(metaDct['stimuli'],2)) 

######### MAIN LOOP ##########
totalNumComparisons = 0;
for (stimA, stimB) in stimCmbTpl:
    
    ## get all data for both trial types
    indsBoth = (df['stimulus']==stimA) | (df['stimulus']==stimB)
    df_bothStimuli = df[indsBoth]

    
    #### select data by animals and sessions
    for animal in metaDct['animals']:
        print('animal: ', animal)
        for session in metaDct['sessions']:
            print('session:', session)
            inds_animalSession = (df_bothStimuli['animal'] == animal) & (df_bothStimuli['session'] == session)
            df_animalSession = df_bothStimuli[inds_animalSession]
            
            ## get both stim
            df_anmlSessStimA = df_animalSession[df_animalSession['stimulus'] == stimA]
            df_anmlSessStimB = df_animalSession[df_animalSession['stimulus'] == stimB]
            print(stimA,stimB)
            
            ## get lists of trial numbers of each stimuli's presentations 
            trials_stimA = np.unique(df_anmlSessStimA['trial'].tolist())
            trials_stimB = np.unique(df_anmlSessStimB['trial'].tolist())
            print(trials_stimA,trials_stimB)
        
            #### get number of timePts in each trial selected above 
            ## (1 to 3 presentations of the same stimuli exist per session in Prabhat's data)
            numTimePtsPerTrial = np.empty((2,max(len(trials_stimA),len(trials_stimB))))
            numTimePtsPerTrial[:] = np.nan
            stimInd = 0;
            for thisStimTypeTrialNums in [trials_stimA, trials_stimB]:
                trialInd = 0
                for trial in thisStimTypeTrialNums:
                    inds_thisTrial = (df_animalSession['trial']==trial)
                    numTimePtsPerTrial[stimInd,trialInd] = np.sum(inds_thisTrial)
                    trialInd += 1
                stimInd += 1
            print(numTimePtsPerTrial) # rows are for stimuli type; cols are presentation of that stimulus            
            
            #### test and sort candidate comparisons based on whether the number of trials per session 
            ##      and approximate number of timePts match
            
            ## no trials of either type --> discard this comparison for this animal/session   
            if np.all(np.isnan(numTimePtsPerTrial)):
                print("discarded: neither stimulus type were found for this animal and session")
                break # skip to next session (WORK: handle this)
                
            ## different numbers of trials per stimuli/session --> discard this comparison for this animal/session 
            elif np.any(np.isnan(numTimePtsPerTrial)): 
                print("discarded: mismatching numbers of trials per stimulus type for this animal/session")
                break # skip to next session (WORK: handle this)
            
            ## FULFILLED here: condition that allows analysis to proceed to attempted data
            elif not np.any(np.isnan(numTimePtsPerTrial)): 
                print("trial numbers match")
            else:
                raise RuntimeError('unexpected trial comparison occurred')
            
            print("checking approx num of time points")
            
            #### discard this comparison for this animal/session if number of time points are too dissimilar
            minTPs = np.min(numTimePtsPerTrial) 
            maxTPs = np.max(numTimePtsPerTrial)
            meanTPs = np.mean(numTimePtsPerTrial)
            stdTPs = np.std(numTimePtsPerTrial)
            print('min', minTPs)
            print('max', maxTPs)
            print('std', stdTPs)
            print('mean',meanTPs)
            if (np.abs(minTPs-meanTPs) > (threshTPs_stdFromMean * np.abs(meanTPs-stdTPs)))                 or (np.abs(maxTPs-meanTPs) > (threshTPs_stdFromMean * np.abs(meanTPs-stdTPs))): 
                print("discarded this comparison because variance in trial length is above the user's threshold")
                break # skip to next session (WORK: handle this)
            totalNumComparisons += 1    
            print('\n')
        print('########\n')

print('total number of comparisons: ', totalNumComparisons)

           


# In[19]:

#testInds = (df['animal']==1) & (df['session']==1) & (df['stimulus']=='USS') & (df['trial']==5)
#print(np.sum(testInds))


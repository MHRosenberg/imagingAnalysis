
# coding: utf-8

# In[18]:

# load tidy dataset

import pandas as pd
import os.path
import numpy as np
from itertools import combinations
import glob
import sys
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
import pdb

## to increase the cell width of the notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# from sklearn.learning_curve import learning_curve
# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import ShuffleSplit

def loadTidyTimings(tidyTimingsCSV): 
    assert os.path.isfile(tidyTimingsCSV), "desired file does not exist" 
    df = pd.read_csv(tidyTimingsCSV, header = None, names = ["animal", "session", "trial", "stimulus", "onsetFrame", "offsetFrame", "onsetTime", "offsetTime"])
    return df 

def loadTidyCalcium(tidyData): 
    assert os.path.isfile(tidyData), "desired file does not exist" 
    df = pd.read_csv(tidyData, header = None, names = ["date", "animal", "session", "trial", "stimulus", "neuronID", "timePt", "CaSignal"])
    return df 

# helper functions to return the number of distinct types in the provided data frame
getNumNeurons = lambda df: len(np.unique(df['neuronID'].tolist()))
getNeurons = lambda df: np.unique(df['neuronID'].tolist())

getNumTrials = lambda df: len(np.unique(df['trial'].tolist()))
getTrials = lambda df: np.unique(df['trial'].tolist())

getOnsetFrameNum = lambda animal, session, trial: df_timings.loc[(df_timings['animal']==animal) & (df_timings['session']==session) & (df_timings['trial']==trial), ['onsetFrame']].values[0][0]
getOffsetFrameNum = lambda animal, session, trial: df_timings.loc[(df_timings['animal']==animal) & (df_timings['session']==session) & (df_timings['trial']==trial), ['offsetFrame']].values[0][0]


# pass in pre-filtered data set containing data for only one animal and the same session (ie SAME NEURONS!)
def getListsOfTrialIDs(df_animalSession):
    ## get trials for both stimuli
    df_anmlSessStimA = df_animalSession[df_animalSession['stimulus'] == stimA]
    df_anmlSessStimB = df_animalSession[df_animalSession['stimulus'] == stimB]
    print(stimA,stimB)

    ## get lists of trial numbers of each stimuli's presentations 
    trials_stimA = np.unique(df_anmlSessStimA['trial'].tolist())
    trials_stimB = np.unique(df_anmlSessStimB['trial'].tolist())
    print("trial IDs for each stimulus type",trials_stimA,trials_stimB)
    return (trials_stimA,trials_stimB)

## pass in a data frame with only a single animal and session    
def getNumTimePtsPerTrial(df_animalSession, trials_stimA, trials_stimB):

    #### get number of timePts in each trial selected above 
    ## (1 to 3 presentations of the same stimuli exist per session in Prabhat's data)
    numTimePtsPerTrial = np.empty((2,max(len(trials_stimA),len(trials_stimB))))
    numTimePtsPerTrial[:] = np.nan
    stimInd = 0;
    for thisStimTypeTrialNums in [trials_stimA, trials_stimB]:
        trialInd = 0
        for trial in thisStimTypeTrialNums:
            inds_thisTrial = (df_animalSession['trial']==trial)
            tmp_df_thisTrial = df_animalSession[inds_thisTrial] # gives all time points for all neurons
            numNeurons = getNumNeurons(tmp_df_thisTrial) 
            numTimePtsPerTrial[stimInd,trialInd] = np.sum(inds_thisTrial)/numNeurons 
            trialInd += 1
        stimInd += 1
    print('stimuliType x presentationNum (row x col)')    
    print(numTimePtsPerTrial) # rows are for stimuli type; cols are presentation of that stimulus
    return numTimePtsPerTrial

## test candidate comparisons based on whether the number of trials per session and approximate number of timePts match
def areNumTrialsPerStimulusEqual(numTimePtsPerTrial):
    
    ## no trials of either type --> discard this comparison for this animal/session   
    if np.all(np.isnan(numTimePtsPerTrial)):
        print("DISCARDED: neither stimulus type were found for this animal and session")
        return False  # skip to next session (WORK: handle this)
        
    ## different numbers of trials per stimuli/session --> discard this comparison for this animal/session 
    elif np.any(np.isnan(numTimePtsPerTrial)): 
        print("DISCARDED: mismatching numbers of trials per stimulus type for this animal/session")
        return False # skip to next session (WORK: handle this)

    ## FULFILLED here: condition that allows analysis to proceed to attempted data
    elif not np.any(np.isnan(numTimePtsPerTrial)): 
        print("trial numbers match")
    else:
        raise RuntimeError('unexpected trial comparison occurred')
        return False
    
    print("checking approx num of time points")

## input argument generated from getNumTimePtsPerTrial
def areNumTimePtsPerTrialSimilar(numTimePtsPerTrial):
    if np.isnan(threshTPs_stdFromMean) == True:
        return True
    minTPs, maxTPs, meanTPs, stdTPs = timePtStats(numTimePtsPerTrial)             
    if (np.abs(minTPs-meanTPs) > (threshTPs_stdFromMean * np.abs(meanTPs-stdTPs)))         or (np.abs(maxTPs-meanTPs) > (threshTPs_stdFromMean * np.abs(meanTPs-stdTPs))): 
        print("DISCARDED: variance in trial length is above the user's threshold")
        return False # skip to next session (WORK: handle this)

    ### passed all criteria if it made it this far
    return True

## input argument created by getNumTimePtsPerTrial function
def timePtStats(numTimePtsPerTrial):
    minTPs = int(np.amin(numTimePtsPerTrial))
    maxTPs = int(np.amax(numTimePtsPerTrial))
    meanTPs = np.mean(numTimePtsPerTrial)
    stdTPs = np.std(numTimePtsPerTrial)

    ## useful for debugging
#     print('min', minTPs)
#     print('max', maxTPs)
#     print('std', stdTPs)
#     print('mean',meanTPs)
#     print('|min-mean|=',np.abs(minTPs-meanTPs))
#     print('|max-mean|=',np.abs(maxTPs-meanTPs))
#     print('|mean-std|=',np.abs(meanTPs-stdTPs))
#     print('thresh * |mean-std|=',(threshTPs_stdFromMean * np.abs(meanTPs-stdTPs)))
    
    return minTPs, maxTPs, meanTPs, stdTPs

def sameNeuronConcat(df_trunc, truncFrameNum):
    neuronArr_anmlSess_stimA = np.full((getNumNeurons(df_trunc),truncFrameNum-1), np.nan) # -1 for 0 indexing
    neuronArr_anmlSess_stimB = np.full((getNumNeurons(df_trunc),truncFrameNum-1), np.nan) # -1 for 0 indexing
    for stimLst in [trials_stimA, trials_stimB]:
        if np.array_equal(stimLst,trials_stimA) == True:
            print('\nstimulus:', stimA)
        elif np.array_equal(stimLst,trials_stimB) == True:
            print('\nstimulus:', stimB)
        else:
            raise RuntimeError('unexpected trial concatenation condition occurred')
        print('truncation frame: ', truncFrameNum)

        ## create temporary sub matrix of concatenated cells for ONE stimulus
        tmp_neuronsArr_sameStim = np.empty((getNumNeurons(df_trunc),truncFrameNum)) # no -1 DESPITE 0 indexing
        for trial in stimLst:
            print("appending same neurons in trial: ", trial)

            ##  create temporary sub matrix of same trial all cells
            tmp_neuronsArr_sameStim_sameTrial = np.empty((getNumNeurons(df_trunc),truncFrameNum)) # no -1 DESPITE 0 indexing
            for neuron in getNeurons(df_trunc):
                tmp_neuronInds = (df_trunc['trial']==trial) & (df_trunc['neuronID']==neuron)
                tmp_neuronSeries = df_trunc.loc[tmp_neuronInds,'CaSignal']
                
                ## pandas to numpy conversion
                tmp_neuronVec = tmp_neuronSeries.as_matrix()
                tmp_neuronsArr_sameStim_sameTrial[neuron-1,:] = tmp_neuronVec  
#                 print('neuronVec',np.shape(tmp_neuronVec))
#                 print('neuronsArr_sameStim_sameTrial', np.shape(tmp_neuronsArr_sameStim_sameTrial))
                 
            ## append trials to right of same stim if not the first entry
            if trial == stimLst[0]:
                tmp_neuronsArr_sameStim = np.copy(tmp_neuronsArr_sameStim_sameTrial)
            else:    
                tmp_neuronsArr_sameStim = np.concatenate((tmp_neuronsArr_sameStim, tmp_neuronsArr_sameStim_sameTrial), axis=1)
            print('same stim:', np.shape(tmp_neuronsArr_sameStim))

        ## save concatenated data to output variables
        if np.array_equal(stimLst,trials_stimA):
            neuronArr_anmlSess_stimA = tmp_neuronsArr_sameStim
        elif np.array_equal(stimLst,trials_stimB):
            neuronArr_anmlSess_stimB = tmp_neuronsArr_sameStim
        else:
            raise RuntimeError('unexpected same neuron concatenation state occured')
        
    return neuronArr_anmlSess_stimA, neuronArr_anmlSess_stimB


# In[2]:

################ concatenate all .csv files exported from matlab into single pandas dataframe df

### tidy csv file and dir (use makeTidy_Anderson.m to convert Ann's structure to csv)

## for Ubuntu laptop
tidyDataDir = '/home/orthogonull/a_MHR/aa_research/aa_gitResearch/git_ignored/imagingAnalysis/data/2_tidyCSVformat/' 
tidyTimingsFileAndPath = '/home/orthogonull/a_MHR/aa_research/aa_gitResearch/git_ignored/imagingAnalysis/data/2_tidyCSVformat/stimulusTimings.csv' # for Ubuntu laptop

## for Anderson desktop
# tidyDataDir = 'C:\\Users\\public.Analysis\\Desktop\\MHR\\git_ignored\data\\2_tidyCSVformat\\' 
# tidyTimingsFileAndPath = 'C:\\Users\\public.Analysis\\Desktop\\MHR\\git_ignored\\data\\2_tidyCSVformat\\stimulusTimings.csv' # for Anderson desktop

tidyDataFileTemplate = 'mouse'
tidyDataFileExt = '.csv'

dataTemplate = tidyDataDir + tidyDataFileTemplate + '*' + tidyDataFileExt 


print("loading stimulus timings into df_timings")
timingsLst = []
print(tidyTimingsFileAndPath)
timingsLst.append(loadTidyTimings(tidyTimingsFileAndPath))
df_timings = pd.concat(timingsLst)

# get all input files you want to add to the same dataset
dataFiles = np.sort(glob.glob(dataTemplate))
print("\n data files: \n", dataFiles)

print("\n loading and appending to prior pandas data frame")
dataLst = []

for file in dataFiles:
    print(file)
    dataLst.append(loadTidyCalcium(file))
df = pd.concat(dataLst)

print('finished loading')

############# ALL DATA STORED HERE IN DF
    


# In[3]:

## survey/search data to prepare for split operation
metaStrs = [['dates','date'],['animals','animal'],['sessions','session'],['maxTrials','trial'],['stimuli','stimulus']]

print("searching over entire data set to get range of various IDs for data (used in subsequent loops)") 

## this dictionary holds useful info regarding the range of inputs to loop/search over subsequently
metaDct = {}
for a,b in metaStrs:
    print(a,b)
    metaDct[a] = np.unique(df[b].tolist())
print(metaDct)


# In[21]:

######### USER PARAMETERS #########
threshTPs_stdFromMean = np.nan ## set to np.isnan if you don't want to filter based on number of TPs WORK: make this std of each type and not all types


# In[22]:

# get all pairs of stimuli
stimCmbTpl = tuple(combinations(metaDct['stimuli'],2)) 


######### MAIN LOOP ##########
df_SVM = pd.DataFrame(columns=('dateOfAnalysis', 'dateOfExperiment', 'animal', 'session', 'stimulusA', 'stimulusB', 'SVM_accuracy'))
ind_comparison = 0;
for (stimA, stimB) in stimCmbTpl:
    print((stimA,stimB))
    
    ## get all data for both trial types
    indsBoth = (df['stimulus']==stimA) | (df['stimulus']==stimB)
    df_bothStimuli = df[indsBoth]

    #### select data by animals and sessions
    for animal in metaDct['animals']:
        print("stimuli comparison num: ", ind_comparison+1)
        print('animal: ', animal)
        for session in metaDct['sessions']:
            print('session:', session)
            
            ## return subselection of data where the same neurons were recorded
            inds_animalSession = (df_bothStimuli['animal'] == animal) & (df_bothStimuli['session'] == session)
            df_animalSession = df_bothStimuli[inds_animalSession]
            try: 
                dateOfExperiment = df_animalSession['date'].values[0]
            except:
                dateOfExperiment = '?'
            print('date of exp:', dateOfExperiment)
            
            # get lists of trial IDs matching stimuli
            trials_stimA, trials_stimB = getListsOfTrialIDs(df_animalSession)
            
            #### skip this comparison <-- if the data don't match in number of trials 
            numTimePtsPerTrial = getNumTimePtsPerTrial(df_animalSession,trials_stimA,trials_stimB)
            if areNumTrialsPerStimulusEqual(numTimePtsPerTrial)==False:
                break     
                
#             #### skip this comparison <-- if the data don't match in approx number of timePts
            if areNumTimePtsPerTrialSimilar(numTimePtsPerTrial) == False:
                break
            
            # animal, session 
            
            #### select time points
            minTPs, maxTPs, meanTPs, stdTPs = timePtStats(numTimePtsPerTrial)
            truncLst = []
            timingsMissing = False
            isFirstTrialExamined = True
            minStimulusDuration = []
            for trial in np.concatenate((trials_stimA,trials_stimB)):
                
                ## get stimulus timings
                onsetFrame = getOnsetFrameNum(animal,session,trial)
                offsetFrame = getOffsetFrameNum(animal,session,trial)
                stimulusDuration = offsetFrame - onsetFrame
#                 print('stimulus duration',stimulusDuration)
#                 print('onset frame', onsetFrame)
#                 print('offset frame', offsetFrame)
                
                ## break out of both loops if timings are missing
                if np.isnan(onsetFrame) or np.isnan(offsetFrame):
                    print('stimulus timing missing --> skipping')
                    timingsMissing = True
                    break
                    
                ## set as min if first or has min duration
#                 print('min stim dur', minStimulusDuration)
#                 print('first trial examined', isFirstTrialExamined)
                if isFirstTrialExamined == True:
                    isFirstTrialExamined = False
                    minStimulusDuration = stimulusDuration[()]
                    print('first trial examined --> minStimDur is just the stimDur', minStimulusDuration)
                elif stimulusDuration[()] < minStimulusDuration and isFirstTrialExamined == False:
                    print('new shortest is: ', stimulusDuration)
                    minStimulusDuration = stimulusDuration[()]
                minStimulusDuration = int(minStimulusDuration)
                
            ## timings are missing --> skip the rest of the analysis for this set of data becausing 
            if timingsMissing == True: 
                break   
            
            #### truncate data, then select time points, and save to new df
            truncLst = []
            print('trial concat', np.concatenate((trials_stimA,trials_stimB)))
            for trial in np.concatenate((trials_stimA,trials_stimB)):
                onsetFrame = getOnsetFrameNum(animal,session,trial)
                offsetFrame_chosen = onsetFrame + minStimulusDuration
                print('min stim dur', minStimulusDuration)
                print('onset frame', getOnsetFrameNum(animal,session,trial))
                print('offsetFrame_chosen: ',offsetFrame_chosen)
                
                tmp_inds_trunc = (df_animalSession['trial']==trial) & (df_animalSession['timePt'] >= onsetFrame.astype(int)) & (df_animalSession['timePt'] < offsetFrame_chosen.astype(int))
                tmp_df_trunc = df_animalSession[tmp_inds_trunc]
                truncLst.append(tmp_df_trunc)
            df_trunc = pd.concat(truncLst)
            
            ### just an optional sanity check
            print('num time pts in df_trunc')
            getNumTimePtsPerTrial(df_trunc, trials_stimA, trials_stimB) ### work
            ###
            
            #### concatenate same cells 
            ### loop over and concatenate neurons into the same row if they're the same neuron and stimuli 
            ##      (ie mouse, session, stimuli)
            print(animal,session)
            neuronArr_anmlSess_stimA, neuronArr_anmlSess_stimB = sameNeuronConcat(df_trunc, minStimulusDuration)
            print('shape', np.shape(neuronArr_anmlSess_stimA), np.shape(neuronArr_anmlSess_stimB))
            
            ######### SVM #########
            
            ## create SVM format input by concatenating both classes (stimuli types); y is the labels
            print("stimA, stimB",np.shape(neuronArr_anmlSess_stimA), np.shape(neuronArr_anmlSess_stimB))
            X = np.concatenate((neuronArr_anmlSess_stimA, neuronArr_anmlSess_stimB), axis = 0)
            y = np.empty((neuronArr_anmlSess_stimA.shape[0]+neuronArr_anmlSess_stimB.shape[0]))
            y[:neuronArr_anmlSess_stimA.shape[0]] = 0
            y[neuronArr_anmlSess_stimB.shape[0]:] = 1
            print("X:", X.shape)
            print("y:", y.shape)
            
            print("k fold partitioning")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            
            print("training SVM")
            clf = svm.SVC(kernel='linear')
            
            ### WORK
#             if ind_comparison == 14:
#                 np.savetxt("y_train_works.csv", y_train, delimiter=",")
#                 np.savetxt("X_train_works.csv", X_train, delimiter=",")
            
            try:
                
                print(ind_comparison) 
                clf.fit(X_train.astype('float128'), y_train)
                print("testing SVM")
                tmp_score = clf.score(X_test.astype('float128'), y_test)
                print(tmp_score)
            except:
                print('X_train finite?: ', np.all(np.isfinite(X_train)))
                print('X_test finite?: ', np.all(np.isfinite(X_test)))
#                 np.savetxt("y_train_broken.csv", y_train, delimiter=",")
#                 np.savetxt("X_train_broken.csv", X_train, delimiter=",")
                pdb.set_trace()
            ### WORK
            
            ## add each SVM result to df
            analysisDate = pd.to_datetime('now')
            df_SVM.loc[ind_comparison] = [analysisDate, dateOfExperiment, animal, session, stimA, stimB, tmp_score]
#             print(df_SVM)

            ind_comparison += 1
            print('\n')
        print('########\n')
print('total number of comparisons: ', ind_comparison+1)


# In[7]:

## buffer cell
pdb.set_trace()

np.savetxt("X_train.csv", X_train, delimiter=",")

            #             tmp_SVMresult = pd.DataFrame({"animal": [animal]})#,{"testAccuracy": tmp_score})
#             tmp_SVMresult = pd.DataFrame([animal], columns = list([1])) #,{"testAccuracy": tmp_score})
#             df_SVM.loc[df_SVM.index.max() + 1] = [animal, session, tmp_score]
#             df_SVM.loc[totalNumComparisons] = [animal, session, tmp_score]
#             df_SVM.loc[totalNumComparisons] = [0,1,2]
#             df_SVM = pd.DataFrame({'mouse': animal, 'session': session, 'SVMaccuracy': tmp_score}, index =totalNumComparisons)
#             df_SVM.iloc[1] = dict(x=9, y=99)
#             df_SVM.append(tmp_SVMresult, ignore_index=True)
#             clf.predict(X_test, y_test)
            
    
    ##########33
#             cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
            
            # WORK: optional gridsearch
#             gammas = np.logspace(-6, -1, 10)
#             classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
#             classifier.fit(X_train, y_train)
            
#             title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
#             estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
#             plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
#             plt.show()




# In[ ]:

## export timings
df_timings.to_csv('df_timings.csv', header=True, index=True)


# In[25]:

## save raw svm (animal/session) results to /gitTracked/python/
timestr = time.strftime("%Y_%m_%d__%H_%M")
print(timestr)

# Write out raw analysis to csv file
OUTPUT_FILENAME = 'SVM_analysis_raw' + timestr + '.csv'
df_SVM.to_csv(OUTPUT_FILENAME, header=True, index=False)


# In[26]:

## save summary svm data

# svmGrouped, df_SVM_summary = df_SVM.groupby(['stimulusA', 'stimulusB'], as_index=False)

# f = {'SVM_accuracy':['sum','mean'], 'B':['prod']}

stimGrouped = df_SVM.groupby(['stimulusA','stimulusB'], as_index=True)
df_SVM_summaryDesc = stimGrouped.describe()

print(stimGrouped.describe())      

# Write out raw analysis to csv file
OUTPUT_FILENAME = 'SVM_analysis_summary' + timestr + '.csv'
df_SVM_summaryDesc.to_csv(OUTPUT_FILENAME, header=True, index=True)


# In[15]:

## junk / notes

ratData = df.loc[df['stimulusType'] == 'rat',:]
ussData = df.loc[df['stimulusType'] == 'USS',:]

print(ussData)



df.corr()


# Rename the impact force column
df = df.rename(columns={'impact force (mN)': 'impf'})



# In[ ]:




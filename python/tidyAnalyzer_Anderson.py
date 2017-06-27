import pandas as pd
import os.path

#import numpy as np
#import loadTidyCSV.py

def loadTidy(tidyDataDir,tidyDataFile): 
    fullpath = tidyDataDir + tidyDataFile
    assert os.path.isfile(fullpath), "desired file does not exist" 
    df = pd.read_csv(tidyDataDir + tidyDataFile, header = None, names = ["date", "animalNum", "sessionNum", "trialNum", "stimulusType", "neuronID", "timePt", "CaSignal"])
    return df 

# tidy csv file and dir (use makeTidy_Anderson.m to convert Ann's structure to csv)
tidyDataDir = '/home/orthogonull/a_MHR/a_research/a_gitResearch/git_ignored/imagingAnalysis/data/2_tidyFormat/'
tidyDataFile = 'mouse1.csv'

# only need to run this line once
df = loadTidy(tidyDataDir, tidyDataFile)



print('before groupby')
gb = df.groupby(['stimulusType']).get_group('rat')
print(gb)

print('reached end of file')

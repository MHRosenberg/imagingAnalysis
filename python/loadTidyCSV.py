import pandas as pd

def load(tidyDataDir,tidyDataFile): 
    df = pd.read_csv(tidyDataDir + tidyDataFile, header = None, names = ["date", "animalNum", "sessionNum", "trialNum", "stimulusType", "neuronID", "timePt", "CaSignal"])
    return df 
    
load(tidyDataDir, tidyDataFile)
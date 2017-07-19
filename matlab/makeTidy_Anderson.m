%%% Matt Rosenberg 2017 spring rotation
%%% sanity checks at end

%% outdated way to load data (prior to 6/13/17)
%%%load data struct from .mat files via Ann's explore_data, then flatten the structure into a cell array where each row corresponds to a Ca recordings (composed of N cells x time x Ca)
% explore_data % Ann's script to unpack data
% [dataTable, dataCell] = flattenMouseDataStructToTable(mouse); % table and cell contain identical info but are structured differently to aid different implementations

%% clear workspace
clear;
close all;

%% new way to load data (after 6/13/17)
tic
disp('loading (sometimes takes a little while)')
load('/home/orthogonull/a_MHR/aa_research/aa_gitResearch/git_ignored/a_dataForCurrentAnalysis/1_structFormat/quickloader.mat')
disp('finished loading')
toc
%%
%%% input the name of the matlab struct containing the data into the function below
[dataTable, dataCell] = flattenMouseDataStructToTable(expt);
A_results = [];

%%
clear vars expt
VERBOSE = 1;
SELECTED_MOUSE = 0;% use 0 to select all mice

%% just save stimulus onsets to csv file
OUTPUT_DIR = '/home/orthogonull/a_MHR/aa_research/aa_gitResearch/git_ignored/a_dataForCurrentAnalysis/2_tidyCSVformat/';

t = dataTable; 
tidyOnsetsCell = cell(height(t), 6);
for trialInd =1:height(t)
    disp(['trial index: ' num2str(trialInd)])
       %%% check if onset/offset info exists in dataCell
    if ~isnan(t.annot(trialInd,1).stim.stim_on)
        disp('stimulus times found')
        stimOnset = t.annot(trialInd,1).stim.stim_on(1);
        stimOffset = t.annot(trialInd,1).stim.stim_on(2);
        
    else %%% WORK: add condition to potentially provide estimated stimulus onset
        disp('no stimulus onset/offset times found')
        stimOnset = nan;
        stimOffset = nan;
    end
    tidyOnsetsCell(trialInd,:) = [t.mouse(trialInd), t.session(trialInd), t.trial(trialInd), t.stim(trialInd), stimOnset, stimOffset];

end
clearvars t 
 
FILE_NAME = 'stimulusTimings.csv';
path_n_name = [OUTPUT_DIR FILE_NAME];
cell2csv(path_n_name,tidyOnsetsCell)


%% make data tidy (possibly with onsets and offsets depending on the version)

mice = unique(dataTable.mouse);
clear vars dataCell 
tidyData = {};
for mouseInd = 1:numel(mice)
    [dataTable_selectedTrials, dataCell_selectedTrials] = selectTrials(dataTable, mice(mouseInd), VERBOSE);
    for trialInd = 1:length(dataCell_selectedTrials)
        disp(['mouse: ' num2str(mouseInd) '; trial num: ' num2str(trialInd)])
        newTidyTrialCell = makeTrialTidy(dataTable_selectedTrials(trialInd,:), dataCell_selectedTrials(trialInd,:));
        tidyData = [tidyData; newTidyTrialCell];
    end
    tic
    disp(['made data tidy for mouse: ' num2str(mouseInd) 'of ' num2str(numel(mice))])
    disp('writing tidy data to csv file (takes a long time)')
    fileName = ['mouse' num2str(mice(mouseInd)) '.csv'];
    cell2csv(fileName,tidyData)
    toc
    disp('wrote csv file to current directory')
end



















% different format of output data: didn't really help with loading into python
%dataFlatStruct = table2struct(dataTable);

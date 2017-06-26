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
load('/home/orthogonull/a_MHR/a_research/a_gitResearch/git_ignored/imagingAnalysis/quickloader.mat')
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


%%
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
    fileName = ['mouse' num2str(mouseInd)];
        cell2csv('mouse.csv',tidyData)
    toc
    disp('wrote csv file to current directory')
end



















% different format of output data: didn't really help with loading into python
%dataFlatStruct = table2struct(dataTable);

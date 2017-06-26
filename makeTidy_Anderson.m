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
disp('loading (sometimes takes a little while)')    
load('/home/orthogonull/a_MHR/a_research/a_gitResearch/git_ignored/imagingAnalysis/quickloader.mat')
disp('finished loading')

%%
%%% input the name of the matlab struct containing the data into the function below
[dataTable, dataCell] = flattenMouseDataStructToTable(expt); 
A_results = [];

%%
% clear vars expt 
makeTrialTidy



















% different format of output data: didn't really help with loading into python
%dataFlatStruct = table2struct(dataTable);

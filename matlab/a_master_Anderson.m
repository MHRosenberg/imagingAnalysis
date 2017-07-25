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
load('~/a_MHR/a_research/a_Anderson/endoscope/Ca imaging/quickloader.mat')
disp('finished loading')

%%
%%% input the name of the matlab struct containing the data into the function below
[dataTable, dataCell] = flattenMouseDataStructToTable06_13_17(expt); % table and cell contain identical info but are structured differently to aid different implementations
A_results = [];

%% data selection parameters
% TRIAL_TYPE_1 = {'USS'}; % use {} if you wish to select all stimuli types
% TRIAL_TYPE_0 = {'male'}; % rat, tone, USS, male, pred odor, peanut odor, toy, mineral oil odor

SELECTED_MOUSE = 0; % use 0 if you wish to pool all mice
VERBOSE = 0; % displays output for selectTrials function if set to 1, not if set to 0
STIM_ONSET_GUESS = 201;
STIM_OFFSET_GUESS = 430;

%% store selected trials
[dataTable_selectedTrials, ~, ~] = selectTrials06_05_17(dataTable, SELECTED_MOUSE, VERBOSE); % syntax: dataTable, dataCell, SELECTED_MOUSE, stimulie eg. 'rat', 'pred odor' (varargins (list of stimuli name))

%% concatenate cells into the same rows in each session
[dataTable_uniqueCellRows] = rowConcatenateCells06_24_17(dataTable_selectedTrials, STIM_ONSET_GUESS, STIM_OFFSET_GUESS);

%% generate 2 lists: 1. all trial types 2. all unique pairs of trial types (RANDOMIZED to avoid biases in looking at the data)
allStimuliNames = unique(dataTable_uniqueCellRows.stim);
allStimuliNames = allStimuliNames(randperm(length(allStimuliNames))); % optional: RANDOMIZED!!!
allPairsOfStimuli_inds = combnk(1:length(allStimuliNames),2);
allPairsOfStimuli_names = cell(size(allPairsOfStimuli_inds));
for rowInd = 1:length(allPairsOfStimuli_inds)
    allPairsOfStimuli_names(rowInd,:) = {allStimuliNames{allPairsOfStimuli_inds(rowInd,1)} allStimuliNames{allPairsOfStimuli_inds(rowInd,2)}};
end

% %% loop over all pairs of stimuli
% close all
% for stimPairInd = 1:length(allPairsOfStimuli_inds)
%     
% %     TRIAL_TYPE_0 = allPairsOfStimuli_names{stimPairInd,2};
%     
%     
%     for stimTypeInd = 1:2
%         TRIAL_TYPE = allPairsOfStimuli_names{stimPairInd,stimTypeInd};
%         [datTbl_uniqueCellRows_sameStim, datCell_uniqueCellRows_sameStim, ~ ] = selectTrials06_17_17(dataTable_uniqueCellRows, SELECTED_MOUSE, VERBOSE, TRIAL_TYPE); % syntax: dataTable, dataCell, SELECTED_MOUSE, stimulie eg. 'rat', 'pred odor' (varargins (list of stimuli name))
%         
%         %%%% select time points of interest from selected trials and create inputs for classifiers
%         X_cellByTime = [];
%         Y = [];
%         for trialInd = 1:size(datTbl_uniqueCellRows_sameStim,1)
%             disp(['processing trial index: ' num2str(trialInd)])
%             thisTrialCellsToAdd = datTbl_uniqueCellRows_sameStim.rast{trialInd};
%             try
%                 X_cellByTime = [X_cellByTime; thisTrialCellsToAdd];
%             catch
%                 %%%%%%%%%%%%%%%% duplicate shorter dataset to match larger data set
%                 priorDim = size(X_cellByTime,2);
%                 newDim = size(thisTrialCellsToAdd,2);
%                 if priorDim < newDim
%                     numCompleteDups = floor(newDim/priorDim);
%                     remainingTrials = mod(newDim,priorDim);
%                     tempSelf = X_cellByTime;
%                     for dupInd = 2:numCompleteDups
%                         X_cellByTime = [X_cellByTime tempSelf];
%                     end
%                     X_cellByTime = [X_cellByTime X_cellByTime(:,1:remainingTrials)];
%                     X_cellByTime = [X_cellByTime; thisTrialCellsToAdd];
%                 elseif priorDim > newDim
%                     numCompleteDups = floor(priorDim/newDim);
%                     remainingTrials = mod(priorDim,newDim);
%                     tempSelf = thisTrialCellsToAdd;
%                     for dupInd = 2:numCompleteDups
%                         thisTrialCellsToAdd = [thisTrialCellsToAdd tempSelf];
%                     end
%                     thisTrialCellsToAdd = [thisTrialCellsToAdd tempSelf(:,1:remainingTrials)];
%                     X_cellByTime = [X_cellByTime; thisTrialCellsToAdd];
%                 else
%                     disp('this should never occur under normal execution --> check data/code (line 38)')
%                 end
%                 
%             end
%             numCellsAdded = size(thisTrialCellsToAdd,1);
%             if strcmp(dataTable_uniqueCellRows.stim{trialInd},TRIAL_TYPE)
%                 Y = [Y ones(numCellsAdded,1)];  % class 1 is 1
%             elseif strcmp(dataTable_uniqueCellRows.stim{trialInd},TRIAL_TYPE_0)
%                 Y = [Y zeros(numCellsAdded,1)]; %%%%% class 2 is 0
%             else
%                 disp('warning: recheck your input data... there is probably some problem (line 48)')
%             end
%         end
%     end
%     
%     X_timeByCell = X_cellByTime'; % make rows for observations and cols for variables to match most matlab functions
%     disp('transposed X to make rows for time points and columns for cells:')
%     disp(size(X_cellByTime))
%     disp('created input Y (labels) matrix of size: ')
%     disp(size(Y))
%     
%     ind_lastOf1stClass = find(Y==1,1,'last'); % all inds prior are from the first class; all subsequent inds are from the second class
%     ind_firstOf2ndClass = ind_lastOf1stClass + 1;
%     
%     %%% train SVM
%     disp(['training svm for stimuli' TRIAL_TYPE_1 TRIAL_TYPE_0])
%     k=10; % num of folds (partitions)
%     
%     cvFolds = crossvalind('Kfold', Y, k);   %# get indices of 10-fold CV
%     cp = classperf(Y);                      %# init performance tracker
%     
%     TIME_WINDOW = 1:size(X_cellByTime,2);
%     
%     for foldInd = 1:k                                  %# for each fold
%         testInds = (cvFolds == foldInd);                %# get indices of test instances
%         trainInds = ~testInds;                     %# get indices training instances
%         
%         disp(['stimulus pair ' num2str(stimPairInd) ' out of ' num2str(size(dataTable_uniqueCellRows,1))])
%         disp(['training fold: ' num2str(foldInd) ' of ' num2str(k) ' total'])
%         
%         %# train an SVM model over training instances
%         %     svmModel = svmtrain(X_cellByTime(trainIdx,:), Y(trainIdx), ...
%         %                  'Autoscale',true, 'Showplot',false, 'Method','QP', ...
%         %                  'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',1);
%         %
%         % Mdl = fitclinear(X_cellByTime(1:end-1,:),Y(1:end-1)); %,'KernelFunction', 'polynomial', 'PolynomialOrder', 2); %,'ObservationsIn', 'columns'); % for high dim data
%         % SVM_Mdl_high = fitclinear(X,Y,'OptimizeHyperparameters', 'all'); %,'ObservationsIn', 'columns'); % for high dim data
%         Mdl = fitcsvm(X_cellByTime(trainInds,TIME_WINDOW),Y(trainInds)); %,'ObservationsIn', 'columns'); % for low dim data
%         
%         %# test using test instances
%         %     pred = svmclassify(Mdl, X_cellByTime(testIdx,:), 'Showplot',false); % same output but decremented version
%         [predictedLabels,scores] = predict(Mdl,X_cellByTime(testInds,TIME_WINDOW));
%         
%         %# evaluate and update performance object
%         cp = classperf(Y, predictedLabels, testInds);
%     end
%     disp('finished training svm')
%     
%     %%% show SVM results and accuracy
%     
%     %# get confusion matrix
%     %# columns:actual, rows:predicted, last-row: unclassified instances
%     Mdl.ClassNames
%     confusionmat(Y(testInds),predictedLabels)'
%     cp.CountingMatrix
%     
%     %# get accuracy
%     cp.CorrectRate
%     
%     allPairsOfStimuli_names{stimPairInd,3} = cp.CorrectRate;
%     
%     
%     %%% broken save code
%     
%     %     d = char(datetime('today'));
%     %     save(['C:\Users\public.Analysis\Desktop\MHRosenberg_2017rotation\SVMresults' d '.mat'])
%     
%     
%     
%     
%     
%     
%     % ScoreSVMModel = fitPosterior(SVMModel,X,Y);  %%% for posterior prop instead of binary
%     
%     %     sv = SVM_Mdl.SupportVectors;
%     %
%     % sv = SVMModel.SupportVectors;
%     % figure
%     % gscatter(X(:,1),X(:,2),y)
%     % hold on
%     % plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
%     
%     % gscatter(X(:,1),X(:,2),Y)
%     % gscatter(X(:,1),X(:,2),Y)
%     % gscatter(X(:,1),X(:,2),Y, 'g')
%     
%     % hold on
%     % plot(sv(:,1),sv(:,2),'g','ko','MarkerSize',10)
%     % legend('versicolor','virginica','Support Vector')
%     % disp('plotted support vectors')
%     
% end


%% buffer



%% correlation heatmaps for pairs of stimuli %% NOTE: NOT CURRENTLY SUPPORTED FOR COMPARISONS BETWEEN ANIMALS IE SELECTED_MOUSE MUST NOT BE 0!
close all
for stimPairInd = 1:length(allPairsOfStimuli_names)
    TRIAL_TYPE_1 = allPairsOfStimuli_names{stimPairInd,1};
    TRIAL_TYPE_0 = allPairsOfStimuli_names{stimPairInd,2};
    
    %get first trial type min and return truncated data
    [dataTable_selectedTrials, ~, ~] = selectTrials06_05_17(dataTable, dataCell, SELECTED_MOUSE, VERBOSE, TRIAL_TYPE_1);
    if isempty(dataTable_selectedTrials)
        continue;
    end
    [X1_cellByTime, minTimePtsPerTrial_1, maxCalciumDataTimePtsPerTrial_1] = truncateTrialsToShortestLength06_09_17(dataTable_selectedTrials, 'columns');
    
    %get second trial type min and return truncated data
    [dataTable_selectedTrials, ~, ~] = selectTrials06_05_17(dataTable, dataCell, SELECTED_MOUSE, VERBOSE, TRIAL_TYPE_0);
    if isempty(dataTable_selectedTrials)
        continue;
    end
    [X0_cellByTime, minTimePtsPerTrial_0, maxCalciumDataTimePtsPerTrial_0] = truncateTrialsToShortestLength06_09_17(dataTable_selectedTrials, 'columns', minTimePtsPerTrial_1);
    
    %truncate first if second is shorter than first
    if minTimePtsPerTrial_1 > minTimePtsPerTrial_0
        [dataTable_selectedTrials, ~, ~] = selectTrials06_05_17(dataTable, dataCell, SELECTED_MOUSE, VERBOSE, TRIAL_TYPE_1);
        [X1_cellByTime, minTimePtsPerTrial_1, maxCalciumDataTimePtsPerTrial_1] = truncateTrialsToShortestLength06_09_17(dataTable_selectedTrials, 'columns', minTimePtsPerTrial_0);
    end
    
    disp(['min & max num of trials for stimulus type: ' num2str(minTimePtsPerTrial_1) '| ' num2str(maxCalciumDataTimePtsPerTrial_1) '| ' TRIAL_TYPE_1])
    disp(['min & max num of trials for stimulus type: ' num2str(minTimePtsPerTrial_0) '| ' num2str(maxCalciumDataTimePtsPerTrial_0) '| ' TRIAL_TYPE_0])
    clear dataTable_selectedTrials % cleared because it's an overwritten variable so results might be misleading
    
    figure
    %     C = corr(X1_cellByTime',X0_cellByTime','type','pearson'); % 2 other types of correlation are supported by matlabs
    C = xcorr2(X1_cellByTime',X0_cellByTime'); % 2 other types of correlation are supported by matlabs
    imagesc(C)
    colorbar
    colormap('jet')
    title(['cell corr for mouse: ' num2str(SELECTED_MOUSE)])
    xlabel(TRIAL_TYPE_1)
    ylabel(TRIAL_TYPE_0)
    
    figure
    C = xcorr2(X1_cellByTime,X0_cellByTime);
    imagesc(C)
    colorbar
    colormap('jet')
    title(['time pt corr for mouse: ' num2str(SELECTED_MOUSE)])
    xlabel(TRIAL_TYPE_1)
    ylabel(TRIAL_TYPE_0)
    
    
    k = waitforbuttonpress;
    spreadfigures;
    %     set(gca, 'xlabel', TRIAL_TYPE_1)
    %     set(gca, 'ylabel', TRIAL_TYPE_0)
    
    %         meanCorr_cells{stimInd,1} = TRIAL_TYPE_1;
    %         meanCorr_cells{stimInd,2} = mean(mean(C));
end
%         meanCorr_cells = sortrows(meanCorr_cells,-2);
%         A_results. = allPairsOfStimuli_names;




%% store selected trials
[dataTable_selectedTrials, dataCell_selectedTrials, varargin] = selectTrials06_05_17(dataTable, dataCell, SELECTED_MOUSE, TRIAL_TYPE_1,TRIAL_TYPE_0);

%% get shortest and longest Ca trial lengths
[X_truncFull] = truncateTrialsToShortestLength(dataTable_selectedTrials);


%% partial least-squares regression ON BOTH TRIAL TYPES
close all;

NUM_COMPONENTS = 20;

X_trFl_cellByTime = X_truncFull;

% Y_cellByTime = Y * ones(1,size(X_trFl_cellByTime,1));
% Y_timeByCell = Y_cellByTime'; %%%%%% results in all columns of XL being the same
% Y_plsReg = zscore(Y_timeByCell);

X_plsReg = X_trFl_cellByTime;
%%% Y_plsReg = Y_timeByCell; % probably wrong
%%% Y_plsReg = Y_cellByTime; % probably wrong
Y_plsReg = Y;
% X_plsReg = zscore(X_plsReg); % normalization (optional)


plsReg = [];
[XL, YL, XS, YS, betas,plsReg.PCTVAR,plsReg.MSE,plsReg.stats] = ...
    plsregress(X_plsReg,Y,NUM_COMPONENTS);

% n = size(X_trFl_cellByTime,1);
% Y_approx = [ones(n,1),X_trFl_cellByTime]*betas;
% figure
% hold on
% scatter(X_trFl_cellByTime,Y_approx);


X0_approx = XS * XL';
Y0_approx = YS * YL';
figure
hold on
numTimePts = size(X0_approx,2);
for cellInd = 1:size(X0_approx,1)
    plot(1:numTimePts,X0_approx(cellInd,:),'r')
    plot(1:numTimePts,X_plsReg(cellInd,:),'b')
end
title('Xapprox (XS XL transpose)')
legend('partial least squares approx', 'raw')

% figure
% hold on
% scatter(XL(1:ind_firstOf2ndClass-1,1),YL(1:ind_firstOf2ndClass-1,1),'r')
% scatter(XL(ind_firstOf2ndClass:end,1),YL(ind_firstOf2ndClass:end,1),'b')
% title('XL YL')

figure
hold on
scatter(XL(1:200,1),XL(1:200,2),'r')
scatter(XL(201:400,1),XL(201:400,2),'b')
scatter(XL(401:600,1),XL(401:600,2),'g')
title('XL')

% figure
% hold on
% scatter(YL(1:ind_firstOf2ndClass-1,1),YL(1:ind_firstOf2ndClass-1,2),'r')
% scatter(YL(ind_firstOf2ndClass:end,1),YL(ind_firstOf2ndClass:end,2),'b')
% title('YL')

figure
hold on
scatter(XS(1:ind_firstOf2ndClass-1,1),XS(1:ind_firstOf2ndClass-1,2),'r')
scatter(XS(ind_firstOf2ndClass:end,1),XS(ind_firstOf2ndClass:end,2),'b')
title('XS')

figure
hold on
scatter(YS(1:ind_firstOf2ndClass-1,1),YS(1:ind_firstOf2ndClass-1,2),'r')
scatter(YS(ind_firstOf2ndClass:end,1),YS(ind_firstOf2ndClass:end,2),'b')
title('SY')

%%% less important plsregress figures
figure;
plot(1:NUM_COMPONENTS,cumsum(100*plsReg.PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

figure
yfit = [ones(size(X_plsReg,1),1) X_plsReg]*betas;
residuals = Y_plsReg- yfit;
stem(residuals)
xlabel('Observation');
ylabel('Residual');

spreadfigures


%% t-SNE (REQUIRES MATLAB 17a)
close all
figure
rng default % for reproducibility
Y = tsne(X_cellByTime,'Algorithm','exact','Distance','mahalanobis');
subplot(2,2,1)
gscatter(Y(:,1),Y(:,2),species)
title('Mahalanobis')

rng default % for fair comparison
Y = tsne(X_cellByTime,'Algorithm','exact','Distance','cosine');
subplot(2,2,2)
gscatter(Y(:,1),Y(:,2),species)
title('Cosine')

rng default % for fair comparison
Y = tsne(X_cellByTime,'Algorithm','exact','Distance','chebychev');
subplot(2,2,3)
gscatter(Y(:,1),Y(:,2),species)
title('Chebychev')

rng default % for fair comparison
Y = tsne(X_cellByTime,'Algorithm','exact','Distance','euclidean');
subplot(2,2,4)
gscatter(Y(:,1),Y(:,2),species)
title('Euclidean')

%% PCA
close all

X_timeByCell_full = X_truncFull';

STARTING_SCORE_DIM = 1;
PCA = [];
[PCA.coeff,PCA.score,PCA.latent,PCA.tsquared,PCA.explainedVar] = pca(X_timeByCell_full,'VariableWeights','variance');

data_reconstructed = PCA.score * PCA.coeff';

figure
hold on
plot(PCA.score(1:ind_firstOf2ndClass-1,STARTING_SCORE_DIM),PCA.score(1:ind_firstOf2ndClass-1,STARTING_SCORE_DIM+1),'r');
plot(PCA.score(ind_firstOf2ndClass:end,STARTING_SCORE_DIM),PCA.score(ind_firstOf2ndClass:end,STARTING_SCORE_DIM+1),'b');
% plot(score(:,1),score(:,3),'+')
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
%plot3 for 3d line plot
% consider using biplot to view individual contributions of variables to components

%% plot all autocorrelations between cells
close all
analysis = [];
meanCorr_cells = cell(length(allStimuliNames),2);
for mouseInd = [1 3] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% select mice here
    for stimInd = 1:length(allStimuliNames)
        TRIAL_TYPE_1 = allStimuliNames{stimInd};
        [dataTable_selectedTrials, dataCell_selectedTrials, varargin] = selectTrials06_05_17(dataTable, dataCell, mouseInd, TRIAL_TYPE_1);
        [X_truncFull] = truncateTrialsToShortestLength(dataTable_selectedTrials);
        X_trFl_cellByTime = X_truncFull;
        
        figure
        C = corr(X_trFl_cellByTime',X_trFl_cellByTime');
        imagesc(C)
        colorbar
        colormap('jet')
        title(['cells corr: ' TRIAL_TYPE_1 ' mouse: ' num2str(mouseInd)])
        
        meanCorr_cells{stimInd,1} = TRIAL_TYPE_1;
        meanCorr_cells{stimInd,2} = mean(mean(C));
    end
    meanCorr_cells = sortrows(meanCorr_cells,-2);
    analysis.mouse{mouseInd}.meanCorrCells = meanCorr_cells;
end
spreadfigures

%% plot all autocorrelations between time points
close all
meanCorr_timePts = cell(length(allStimuliNames),2);
for mouseInd = [1 3]
    for stimInd = 1:length(allStimuliNames)
        TRIAL_TYPE_1 = allStimuliNames{stimInd};
        [dataTable_selectedTrials, dataCell_selectedTrials, varargin] = selectTrials06_05_17(dataTable, dataCell, mouseInd, TRIAL_TYPE_1);
        [X_truncFull] = truncateTrialsToShortestLength(dataTable_selectedTrials);
        X_trFl_cellByTime = X_truncFull;
        
        figure
        C = corr(X_trFl_cellByTime,X_trFl_cellByTime);
        imagesc(C)
        colorbar
        colormap('jet')
        title(['time points corr: ' TRIAL_TYPE_1 ' mouse: ' num2str(mouseInd)])
        
        disp(['mean correlation for ' TRIAL_TYPE_1 ': ' num2str(mean(mean(C)))])
        
        meanCorr_timePts{stimInd,1} = TRIAL_TYPE_1;
        meanCorr_timePts{stimInd,2} = mean(mean(C));
    end
    meanCorr_timePts = sortrows(meanCorr_timePts,-2);
    analysis.mouse{mouseInd}.meanCorrTimePts = meanCorr_timePts;
end
spreadfigures


%% plot raw correlation heat map
close all
figure
C = corr(X_trFl_cellByTime(1:ind_lastOf1stClass,:)',X_trFl_cellByTime(1:ind_lastOf1stClass,:)');
imagesc(C)
colorbar
colormap('jet')
title('corr betw cells class1')
numCells = size(X_trFl_cellByTime,1);
line([ind_firstOf2ndClass,ind_firstOf2ndClass],[-10,numCells+10],'color', 'k','linewidth', 4)

figure
C = corr(X_trFl_cellByTime(ind_firstOf2ndClass:end,:),X_trFl_cellByTime(ind_firstOf2ndClass:end,:));
imagesc(C)
colorbar
colormap('jet')
title('corr betw time points class1')

figure
C = corr(X_trFl_cellByTime',X_trFl_cellByTime');
imagesc(C)
colorbar
colormap('jet')
title('corr betw cells class2')
numCells = size(X_trFl_cellByTime,1);
line([ind_firstOf2ndClass,ind_firstOf2ndClass],[-10,numCells+10],'color', 'k','linewidth', 4)

figure
C = corr(X_trFl_cellByTime,X_trFl_cellByTime);
imagesc(C)
colorbar
colormap('jet')
title('corr betw time points class2')
spreadfigures


%% train Naive Bayes
Mdl = fitcnb(X_cellByTime,Y);
Mdl.ClassNames
Mdl.Prior

%% plot "raw" data in heatmap and separated dF/F format

responseThreshold_inSTDs = 2;
y_offset_inPlot = 6;

close all

for trialInd = 1:2:height(dataTable)
    close all
    
    data = dataTable.rast{trialInd};
    
    response_max = max(data,[],2);
    response_min = min(data,[],2);
    response_std = std(data,[],2);
    
    data_offsetY = nan(size(data));
    for cellInd = 1:size(data,1)
        data_offsetY(cellInd,:) = data(cellInd,:) + y_offset_inPlot * cellInd;
    end
    
    titleStr = ['stim: ' char(dataTable.stim(trialInd)) ' mouse: ' num2str(dataTable.mouse(trialInd)) ' session: ' num2str(dataTable.session(trialInd)) ' trial: ' num2str(dataTable.trial(trialInd))];
    figure('name', titleStr, 'numbertitle', 'off');
    
    plot(data_offsetY')
    for cellInd = 1:size(data,1)
        thisCellSTD = std(data(cellInd,:));
        aboveThresholdInds = find(data(cellInd,:) > responseThreshold_inSTDs * thisCellSTD);
        if ~isempty(aboveThresholdInds)
            %             ShadePlotForEmpahsis([aboveThresholdInds(1) aboveThresholdInds(1)],'r', 0.5)
        end
    end
    
    figure
    colormap('jet')
    imagesc(data)
    colorbar
    spreadfigures
    
    
    waitforbuttonpress
end

%% plot raw data comparing 1st and 2nd presentations of the same data across days.
close all
dataTable_byStim = sortrows(dataTable,'stim','ascend');

% for i = 1:2:height(dataTable_byStim)
for trialInd = 1:2:3
    close all
    if strcmp(dataTable_byStim.stim(trialInd),dataTable_byStim.stim(trialInd+1))
        
        firstTrialData = dataTable_byStim.rast{trialInd};
        titleStr = ['stim: ' char(dataTable_byStim.stim(trialInd)) ' mouse: ' num2str(dataTable_byStim.mouse(trialInd)) ' session: ' num2str(dataTable_byStim.session(trialInd)) ' trial: ' num2str(dataTable_byStim.trial(trialInd))];
        figure('name', titleStr, 'numbertitle', 'off');
        %         title(titleStr,'fontsize', 7)
        subplot(2,1,1)
        %         plot(firstTrialData')
        errobar(firstTrialData')
        subplot(2,1,2)
        colormap('jet')
        imagesc(firstTrialData)
        colorbar
        %         text( 0.5, 0, titleStr, 'FontSize', 14', 'FontWeight', 'Bold', ...
        %       'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
        
        
        secondTrialData = dataTable_byStim.rast{trialInd+1};
        titleStr = ['stim: ' char(dataTable_byStim.stim(trialInd+1)) 'mouse: ' num2str(dataTable_byStim.mouse(trialInd+1)) ' session: ' num2str(dataTable_byStim.session(trialInd+1)) ' trial: ' num2str(dataTable_byStim.trial(trialInd+1))];
        figure('name', titleStr, 'numbertitle', 'off');
        %         title(titleStr,'fontsize', 7)
        hold on
        subplot(2,1,1)
        plot(secondTrialData')
        subplot(2,1,2)
        colormap('jet')
        imagesc(secondTrialData)
        colorbar
    else
        disp('error expected trials to match stimulus type but they did not...')
    end
    spreadfigures
    waitforbuttonpress
end





%% cross correlations



%%%%%%%%%%%%%%%%%%%%%%%%%%% sanity checks

%% partial least-squares regression ON JUST ONE TRIAL TYPE
close all;
subsetOfTrials = 1:ind_firstOf2ndClass-1;

Y_cellByTime = Y * ones(1,numel(subsetOfTrials));
NUM_COMPONENTS = 20;
plsReg = [];
[plsReg.XL,plsReg.YL,plsReg.XS,plsReg.YS,plsReg.betas,plsReg.PCTVAR,plsReg.MSE,plsReg.stats] = plsregress(X_cellByTime(subsetOfTrials,:),Y_cellByTime(subsetOfTrials,:),NUM_COMPONENTS);
figure;
plot(1:NUM_COMPONENTS,cumsum(100*plsReg.PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

figure
yfit = [ones(size(X_cellByTime(subsetOfTrials,:),1),1) X_cellByTime(subsetOfTrials,:)]*plsReg.betas;
residuals = Y_cellByTime(subsetOfTrials,:) - yfit;
stem(residuals)
xlabel('Observation');
ylabel('Residual');
title('regressed on one trial type, fitting one trial type')

figure
yfit = [ones(size(X_cellByTime,1),1) X_cellByTime]*plsReg.betas;
residuals = Y_cellByTime - yfit;
stem(residuals)
xlabel('Observation');
ylabel('Residual');
title('regressed on one trial type, fitting BOTH trial types')

%% boxplot MHR: NOT CONFIDENT IN INTERPRETABILITY / CORRECTNESS!

figure
boxplot(X_cellByTime')

figure
violin(X_cellByTime')

%% FOR REFERENCE ONLY

[allDataFlatCell(:,8)] = {'test'}; %%%%%%%%%%%%%%% this is how to batch assign rows without a for loop


find([allDataFlatCell{:,3}]==3) % same as: find(cell2mat(allDataFlatCell(:,3))==3) %%%%%%%%%%% how to find indices for numeric data

find(ismember(allDataFlatCell(:,5),'tone')) % %%%%%%%%%%% how to search cell for strings


% Y_cellByTime_upsideDown = flipud(Y_cellByTime);
% X_cellByTime_upsideDown = flipud(X_cellByTime);
% X_plsReg = X_cellByTime_upsideDown; % sanity check to make sure results are symmetrical
% Y_plsReg = Y_cellByTime_upsideDown;
% figure
% plot(score(1:226,1),score(1:226,2),'r')
% hold on
% plot(score(226:end,1),score(226:end,2),'b')
%
% figure
% pareto(explainedVar(1:10));
% xlabel('Principal Component')
% ylabel('Variance Explained (%)')
%
% coeff_orth = inv(diag(std(X_timeByCell_full)))*coeff; % transform coeffs to be orthogonal
% figure
% biplot(coeff_orth(:,1:2),'scores',score(:,1:2));
% % axis([-.26 0.6 -.51 .51]);
%
% figure
% biplot(coeff_orth(:,1:3),'scores',score(:,1:3));
% % axis([-.26 0.8 -.51 .51 -.61 .81]);
% view([30 40]);

%         p = plot(cell2mat(sessionAvgPlotMat'));
%         legend(avg_Ca_responses(:,1),'location','best');
%         title(['mouse ' num2str(mouseInd)])
%         xticks(1:1:3)
%         analyzed_mouse.fig = p;




%             for trialInd = 1:numTrials
%                 stimTypes{trialInd} = mouse(mouseInd).(sessionName)(trialInd).stim;
%             end
%             stimTypes = unique(stimTypes);
%
%             % average Ca response by stim/trial type
%             avg_Ca_responses = stimTypes;
%             avg_Ca_responses{length(stimTypes),2} = []; % columns: stim type, avg response
%             for stimTypeInd = 1:length(stimTypes)
%                 desiredStim = stimTypes{stimTypeInd};
%                 disp(desiredStim);
%                 inds_trialsWstim = find(strcmp({mouse(mouseInd).(sessionName).stim}, desiredStim)==1);
%                 temp = nan(length(inds_trialsWstim),1);
%                 for ind = 1:length(inds_trialsWstim)
%                     temp(ind) = mean(mean(mouse(mouseInd).(sessionName)(inds_trialsWstim(ind)).rast));
%                 end
%                 avg_Ca_responses{stimTypeInd, 2} = mean(temp);
%             end
%             analyzed_mouse(mouseInd).(sessionName).avg_Ca_responses = avg_Ca_responses;
%             sessionAvgPlotMat = [sessionAvgPlotMat avg_Ca_responses(:,2)];

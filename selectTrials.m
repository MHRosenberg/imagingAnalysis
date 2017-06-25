function [dataTable_selectedTrials, dataCell_selectedTrials, varargin] = selectTrials(dataTable, SELECTED_MOUSE, verbose, varargin)

dataCell = table2cell(dataTable);

if size(varargin,2) == 2
    TRIAL_TYPE_1 = varargin{1};
    TRIAL_TYPE_2 = varargin{2};
    disp(varargin)
    
    %%% select trial inds of individual mouse or pool all mice
    if SELECTED_MOUSE == 0
        disp('all mice selected')
        trialInds_mouse = 1:size(dataTable,1);
    else
        trialInds_mouse = find(ismember(dataTable.mouse,SELECTED_MOUSE));
    end
    disp(['selected trials for mouse: ' num2str(SELECTED_MOUSE)])
    disp(trialInds_mouse');
    
    %%% select trial inds of stimulus type FOR TRAINING
    if ~isempty(TRIAL_TYPE_1)
        trialInds_selectedStim1 = find(ismember(dataTable.stim,TRIAL_TYPE_1));
    else
        trialInds_selectedStim1 = trialInds_mouse; % choose all trials for selected mouse if no stimulus is specified above
    end
    disp(['selected trials for stimulus type 1: ' TRIAL_TYPE_1])
    disp(trialInds_selectedStim1');
    if ~isempty(TRIAL_TYPE_2)
        trialInds_selectedStim2 = find(ismember(dataTable.stim,TRIAL_TYPE_2));
    else
        trialInds_selectedStim2 = trialInds_mouse; % choose all trials for selected mouse if no stimulus is specified above
    end
    disp(['selected trials for stimulus type 2: ' TRIAL_TYPE_2])
    disp(trialInds_selectedStim2');
    
    %%%
    % add further criteria here with same format
    %%%
    
    %%% select inds that meet all criteria above
    trialInds_class1 = intersect(trialInds_mouse,trialInds_selectedStim1);
    trialInds_class2 = intersect(trialInds_mouse,trialInds_selectedStim2);
    
    %%% store selected trials into new dataTable and dataCell, both ending in _selectedTrials
    class1num = numel(trialInds_class1);
    class2num = numel(trialInds_class2);
    totalNumTrials = class1num + class2num;
    
    dataCell_selectedTrials = cell(totalNumTrials,size(dataTable,2));
    dataCell_selectedTrials(1:class1num,:) = dataCell(trialInds_class1,:);
    dataCell_selectedTrials(class1num+1:totalNumTrials,:) = dataCell(trialInds_class2,:);
    
    dataTable_selectedTrials = cell2table(dataCell_selectedTrials, 'variablenames', dataTable.Properties.VariableNames);
    
    if verbose == 1
        disp(trialInds_selectedStim2');
        disp('class 1 inds')
        disp(trialInds_class1');
        disp('class 2 inds')
        disp(trialInds_class2');
        disp('finished selecting trials. check output above as a sanity check that the correct inputs were chosen')
        disp('created dataTable_selectedTrials and dataCell_selectedTrials')
        disp(dataTable_selectedTrials);
    end
elseif size(varargin,2) == 1
    TRIAL_TYPE_1 = varargin{1};
    disp(varargin)
    
    %%% select trial inds of individual mouse or pool all mice
    if SELECTED_MOUSE == 0
        trialInds_mouse = 1:size(dataTable,1);
    else
        trialInds_mouse = find(ismember(dataTable.mouse,SELECTED_MOUSE));
    end
    
    %%% select trial inds of stimulus type FOR TRAINING
    if ~isempty(TRIAL_TYPE_1)
        trialInds_selectedStim1 = find(ismember(dataTable.stim,TRIAL_TYPE_1));
    else
        trialInds_selectedStim1 = trialInds_mouse; % choose all trials for selected mouse if no stimulus is specified above
    end
    
    %%% select inds that meet all criteria above
    trialInds_class1 = intersect(trialInds_mouse,trialInds_selectedStim1);
    
    disp('finished selecting trials. check output above as a sanity check that the correct inputs were chosen')
    
    %%% store selected trials into new dataTable and dataCell, both ending in _selectedTrials
    class1num = numel(trialInds_class1);
    totalNumTrials = class1num;
    
    if verbose == 1
        disp(['selected trials for mouse: ' num2str(SELECTED_MOUSE)])
        disp(trialInds_mouse);
        
        disp(['selected trials for stimulus type 1: ' TRIAL_TYPE_1])
        disp(trialInds_selectedStim1);
        
        if isempty(trialInds_class1)
            disp('NO TRIALS MEET CRITERIA! --> skipped')
        else
            disp('trials below meet selection criteria')
            disp(trialInds_class1');
        end
    end
    
    dataCell_selectedTrials = cell(totalNumTrials,size(dataTable,2));
    dataCell_selectedTrials(1:class1num,:) = dataCell(trialInds_class1,:);
    dataTable_selectedTrials = cell2table(dataCell_selectedTrials, 'variablenames', dataTable.Properties.VariableNames);
    disp('created dataTable_selectedTrials and dataCell_selectedTrials')
    disp(dataTable_selectedTrials);
    
elseif isempty(varargin)
    
    if SELECTED_MOUSE == 0
        trialInds_mouse = 1:size(dataTable,1);
    else
        trialInds_mouse = find(ismember(dataTable.mouse,SELECTED_MOUSE));
    end
    
    totalNumTrials = length(trialInds_mouse);
    
    if verbose == 1
        disp(['selected trials for mouse: ' num2str(SELECTED_MOUSE)])
        disp(trialInds_mouse');
        disp('no stimuli chosen --> returning all trials matching mouse selection(s)')
    end
    dataCell_selectedTrials = dataCell(trialInds_mouse,:);
    dataTable_selectedTrials = cell2table(dataCell_selectedTrials, 'variablenames', dataTable.Properties.VariableNames);
end

if SELECTED_MOUSE == 0
    evalin('caller',['clear ', 'dataCell dataTable']) 
end

if isempty(dataTable_selectedTrials)
    disp('error: no output'); % for debug only
end

end
function [datTab_selTrl_uniqueCellRows, datCell_selTrl_uniqueCellRows] = rowConcatenateNeurons(dataTable_selectedTrials, STIM_ONSET_GUESS, STIM_OFFSET_GUESS)

datTab_selTrl_uniqueCellRows = {};

priorInds = 0;
numMice = numel(unique(dataTable_selectedTrials.mouse));
for mouseInd = 1:numMice %%% mouse
    numSessions = numel(unique(dataTable_selectedTrials.session));
    for sessionInd = 1:numSessions %%% session
        allStimuliNames = unique(dataTable_selectedTrials.stim);
        numStimTypes = numel(allStimuliNames);
        
        for stimTypeInd = 1:numStimTypes %%% stimulus
            %%% find indices of the cells that are the same across trials 
            trialInds_selected_mouse = find(ismember(dataTable_selectedTrials.mouse,mouseInd));
            trialInds_selectedSession = find(ismember(dataTable_selectedTrials.session,sessionInd));
            trialInds_selectedStim = find(ismember(dataTable_selectedTrials.stim,allStimuliNames{stimTypeInd}));
            trialInds_sameCell_sameStim = intersect(intersect(trialInds_selected_mouse,trialInds_selectedSession),trialInds_selectedStim);

            %%% select trials meeting criteria above
            tempTable = dataTable_selectedTrials(trialInds_sameCell_sameStim,:);
            for trialInd = 1:height(tempTable)
                %%% get stimulus onset and offset
                stimInds_onsetOffset = tempTable.annot(trialInd,1).stim.stim_on;
                timeInd_start = stimInds_onsetOffset(1);
                timeInd_end = stimInds_onsetOffset(2);
                %%% if stim onset/offset don't exist then use user-specified guesses
                if isnan(timeInd_start) || isnan(timeInd_end)
                    timeInd_start = STIM_ONSET_GUESS;
                    timeInd_end = STIM_OFFSET_GUESS;
                end
                
                
                %%% select trials for specified time points (eg stimulus onset/offset)
                if trialInd == 1
                    try tempTable.rast{1} = tempTable.rast{1}(timeInd_start:timeInd_end);
                    catch ME
                        disp(ME.identifier);
                        pause;
                    end
                    
                else
                    tempTable.rast{1} = [tempTable.rast{1} tempTable.rast{trialInd}(timeInd_start:timeInd_end)];
                end
            end
            tempCell = table2cell(tempTable);

            try tempCell(1,:)
                datCell_selTrl_uniqueCellRows(stimTypeInd+priorInds,:) = tempCell(1,:);  
                clear tempCell
            catch
                disp(['no stim type: ' allStimuliNames{stimTypeInd} ' in this session --> skipping'])
            end
            clear tempCell
        end
        priorInds = priorInds + numStimTypes;
    end
end
datCell_selTrl_uniqueCellRows(all(cellfun('isempty',datCell_selTrl_uniqueCellRows),2),:) = [];
datTab_selTrl_uniqueCellRows = cell2table(datCell_selTrl_uniqueCellRows,'VariableNames', {'date','mouse','session','trial','stim', 'CaFR', ...
    'annoFR', 'rast','CaTime', 'rast_matched', 'match', 'units', 'bounds', 'io', 'annot','annoTime'});

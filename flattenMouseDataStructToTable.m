function [allDataFlatTable, allDataFlatCell] = flattenMouseDataStructToTable(dataStruct)

if exist('mouse') == 0
%     explore_data
end


DATE = '2017_05_00';

MOUSE_IND_START = 1;
MOUSE_IND_END = length(dataStruct);

SESSION_IND_START = 1;
SESSION_IND_END = numel(find(contains(fieldnames(dataStruct), 'session')));

analyzed_mouse = [];
allDataFlatCell = cell(100,5);

% mouse
flattenedCellInd = 1;
for mouseInd = MOUSE_IND_START:MOUSE_IND_END
    
    if isempty(dataStruct(mouseInd).session1)
        disp(newline);
        disp(['mouse ' num2str(mouseInd) ' has no data --> skipping it'])
    else
        disp(['extracting data from mouse ' num2str(mouseInd)])
        
        % session
        sessionAvgPlotMat = [];
        for sessionInd = SESSION_IND_START:SESSION_IND_END
            disp(newline);
            disp(['session ' num2str(sessionInd)]);
            
            %%%%% strip all leaves into flattened cell array
            
            %%% get stimulus/trial names
            sessionName = ['session' num2str(sessionInd)];
            numTrials = size(dataStruct(mouseInd).(sessionName)(:),1);
            stimTypes = cell(numTrials,1);
            
            for trialInd =1:numTrials
                allDataFlatCell{flattenedCellInd,1} = DATE;
                allDataFlatCell{flattenedCellInd,2} = mouseInd;
                allDataFlatCell{flattenedCellInd,3} = sessionInd;
                allDataFlatCell{flattenedCellInd,4} = trialInd;
                allDataFlatCell{flattenedCellInd,5} = dataStruct(mouseInd).(sessionName)(trialInd).stim;
                allDataFlatCell{flattenedCellInd,6} = dataStruct(mouseInd).(sessionName)(trialInd).CaFR;
                allDataFlatCell{flattenedCellInd,7} = dataStruct(mouseInd).(sessionName)(trialInd).annoFR;
                allDataFlatCell{flattenedCellInd,8} = dataStruct(mouseInd).(sessionName)(trialInd).rast;
                allDataFlatCell{flattenedCellInd,9} = dataStruct(mouseInd).(sessionName)(trialInd).CaTime;
                allDataFlatCell{flattenedCellInd,10} = dataStruct(mouseInd).(sessionName)(trialInd).rast_matched;
                allDataFlatCell{flattenedCellInd,11} = dataStruct(mouseInd).(sessionName)(trialInd).match;
                allDataFlatCell{flattenedCellInd,12} = dataStruct(mouseInd).(sessionName)(trialInd).units;
                allDataFlatCell{flattenedCellInd,13} = dataStruct(mouseInd).(sessionName)(trialInd).bounds;
                allDataFlatCell{flattenedCellInd,14} = dataStruct(mouseInd).(sessionName)(trialInd).io;
                allDataFlatCell{flattenedCellInd,15} = dataStruct(mouseInd).(sessionName)(trialInd).annot;
                allDataFlatCell{flattenedCellInd,16} = dataStruct(mouseInd).(sessionName)(trialInd).annoTime;
                flattenedCellInd = flattenedCellInd +1;
            end
            
            disp('flattened data from struct to cell array')
            
        end
    end
end

idx=all(cellfun(@isempty,allDataFlatCell(:,:)),2);
allDataFlatCell(idx,:)=[];

allDataFlatTable = cell2table(allDataFlatCell,'VariableNames', {'date','mouse','session','trial','stim', 'CaFR', ...
    'annoFR', 'rast','CaTime', 'rast_matched', 'match', 'units', 'bounds', 'io', 'annot','annoTime'});

end


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
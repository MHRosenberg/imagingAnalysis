function [tidyTrialCell] = makeTrialTidy(dataTable, dataCell)
trialInd = 1;

alreadyTidyPart = dataCell(trialInd,1:5);

notFlatPartCell = dataTable.rast(trialInd); % ISSUE: expand to flatten the rest of the table at some point
notFlatPartMat = notFlatPartCell{1};


%%% get num of neurons and time points
try
    [numRows, numCols] = size(notFlatPartMat);
    if numRows < numCols
        disp('based on dims, data is neuron x timePt')
        numNeurons = numRows;
        numTimePts = numCols;
    else
        disp('based on dims, data is timePt x timePt')
        numNeurons = numCols;
        numTimePts = numRows;
    end
    clear vars numRows numCols
catch ME
    disp('error caused by notFlatPart')
end


newTidyPart = cell(numTimePts * numNeurons, 5);
newInd = [];
for neuronInd = 1:numNeurons
   for timePtInd = 1:numTimePts
       newInd = timePtInd + ((neuronInd - 1) * numTimePts); % keep adding new neurons to the bottom
       newTidyPart(newInd,:) = {neuronInd, timePtInd, notFlatPartMat(neuronInd,timePtInd), nan, nan};
   end
end


%%% check if onset/offset info exists in dataCell
if ~isempty(dataTable.annot(1,1).stim.stim_on)
    stimOnset = dataTable.annot(1,1).stim.stim_on(1);
    stimOffset = dataTable.annot(1,1).stim.stim_on(2);
    newTidyPart(newInd+1,:) = {nan, nan, nan, stimOnset, stimOffset};
else %%% WORK: add condition to potentially provide estimated stimulus onset
    disp('no stimulus onset/offset times found')
end


%%% add stimulus onset/offset info to master data cell array if it exists


%%% store 


[numRows_newTidy, numCols_newTidy] = size(newTidyPart);
[numRows_alreadyTidy, numCols_alreadyTidy] = size(alreadyTidyPart);


tidyTrialCell = cell(numRows_newTidy, numCols_alreadyTidy + numCols_newTidy);


[tidyTrialCell(:, 1:numCols_alreadyTidy)] = repmat(alreadyTidyPart,numRows_newTidy,1);
tidyTrialCell(:, numCols_alreadyTidy+1 : numCols_alreadyTidy+numCols_newTidy) = newTidyPart;

disp('data SHOULD be "tidy" now : ) ')
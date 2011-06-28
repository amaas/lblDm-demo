function [ objVal, objGrad ] = slaveLblDmRcErObj( wVec )
%SLAVELBLDMRCOBJ Slave evaluation of rc objective
% assumes docThetas and bow data in global state

%% setup params
global state;
modelParams = state.modelParams;
dictSize = modelParams.DictSize;
rvDim = modelParams.RepVecDim;
numDocs = size(state.docBow,1);
rcIndex = modelParams.repConIndex();
wbIndex = modelParams.wordBiasIndex();

% get parameters from their optimized or non-optimized vectors
repConMat = reshape(wVec(rcIndex),dictSize, rvDim);
wbVec = wVec(wbIndex);
% weights for each doc stored as a column
docThetaMat = reshape(state.wNonopt,rvDim, []);

assert(size(docThetaMat,2) == size(state.docBow,1));

% accumulators of gradients and objective values
objGrad = zeros(size(wVec));
objVal = 0;
docProbObj = 0;


for batchStart = 1 : modelParams.BatchSize : numDocs
    batchDocInd = batchStart : ...
        min((batchStart+modelParams.BatchSize-1),numDocs);    
    cur_docBOW = state.docBow(batchDocInd,:);
    cur_docThetaMat = docThetaMat(:,batchDocInd);    
    cur_docLen = state.docLen(batchDocInd);
    cur_docBOWC = bsxfun(@times, cur_docBOW, cur_docLen);
    
    %% objective and gradient for topic model 
    % forward prop .. compute word prob distros for all docs
    docProbs = exp(bsxfun(@plus, repConMat * cur_docThetaMat, wbVec)');    
    sumExp = sum(docProbs,2);
    docProbs = bsxfun(@rdivide, docProbs, sumExp);
    % objval is sum of log model probs weighted by actual probs
    docProbObj = -1 * sum(full(sum(cur_docBOWC .* log(docProbs+eps))));
    %objVal = objVal ...
    %    - state.docLen' * full(sum(cur_docBOW .* log(docProbs+eps), 2));
    objVal = objVal + docProbObj;
    % derivatives are difference of expectations
    % weight docThetas by docLen to handle dot multiply of len for each doc    
    cur_docThetaMat = bsxfun(@times, cur_docThetaMat, cur_docLen');
    objGrad(rcIndex) = objGrad(rcIndex) - ...
        reshape((cur_docThetaMat * cur_docBOW - cur_docThetaMat * docProbs)', [], 1);
    % bias derivative is just the word probabilities
    objGrad(wbIndex) = ...
        objGrad(wbIndex) - (cur_docLen'*(cur_docBOW - docProbs))';
end;
    
% update derivatives for regularization terms
objGrad(rcIndex) = ...
    objGrad(rcIndex) + ...
    (sum(state.docLen) * modelParams.LambdaRc) .* ...
    wVec(rcIndex);

% update for regularization
rcRegObj = sum(state.docLen) * modelParams.LambdaRc *.5 * ...
    sum(sum(repConMat.^2));

objVal = objVal + rcRegObj;
fprintf(1,'docObj: %f   wnObj: %f\n', docProbObj, rcRegObj);
end


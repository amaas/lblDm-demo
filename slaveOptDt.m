function [ thetaVecOut ] = slaveOptDt( thetaVec )
%SLAVEOPTDT Use minfunc to fit each doc theta 
% thetaVec are many docTheta vectors concatenated
% assumes thetaVec ordering matches docProbs in global state

% open local worker pool if necessary
% if matlabpool('size') == 0
%     matlabpool open local 4;
% end;

global state;

modelParams = state.modelParams;
rvDim = modelParams.RepVecDim;
repConMat = reshape(state.wNonopt(modelParams.repConIndex()),...
    modelParams.DictSize, rvDim);
wbVec = state.wNonopt(modelParams.wordBiasIndex());
thetaVecOut = [];

% setup minfunc options
options.Method = 'lbfgs';
options.Display = 'off';
options.DerivativeCheck = 'off';
% limit the max fun evals so not much time is spent optimizing each doc
options.MaxFunEvals = 100;
options.TolFun = 1e-7;
options.TolX = 1e-10;
tic;
parfor d = 1 : size(state.docBow,1)
    curInd = (rvDim*(d-1)+1 : rvDim*d);
    [coVec, coF, coFlag, coInfo] = minFunc(@lblDmObjDt, thetaVec(curInd), ...
        options, state.modelParams, state.docBow(d,:), repConMat, wbVec, state.docLen(d));
    thetaVecOut = [thetaVecOut; coVec];
end
toc;
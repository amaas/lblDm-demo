function [ objVal, objGrad ] = lblDmObjDt( docThetaMat, modelParams, docBOW, repConMat, wbVec, docLen )
%LBLDMOBJALTMF lblDm objective function for single doc theta only
% This function is of the form passed to minFunc
% it assumes wVec is a single docTheta vector
% and as a result docLen is a scalar
% NOTE: must pass docInd to minFunc to pass along


% pull batch data form global context
dictSize = modelParams.DictSize;
rvDim = modelParams.RepVecDim;

% make sure we have a param vector we expect
%assert( length(wVec) == rvDim );
cur_docBOW = full(docBOW);
    
% forward prop .. compute word prob distros for all docs
docProbs = exp(bsxfun(@plus, repConMat * docThetaMat, wbVec)');
docProbs = bsxfun(@rdivide, docProbs, sum(docProbs,2));
% objval is sum of log model probs weighted by actual probs
objVal = -docLen .* full(sum(cur_docBOW .* log(docProbs+eps)));

% derivatives are difference of expectations
% no option for repCon derivatives since we never need them in this obj
objGrad = -docLen .* reshape((cur_docBOW * repConMat - docProbs*repConMat)',[], 1);
        
    
% update derivatives for regularization terms
%numThetaParams = length(modelParams.thetaMatIndex());
numThetaParams = length(docThetaMat);
L1Reg = modelParams.L1Reg;

% update for regularization

dtRegObj = sum(docLen) * ...
    (modelParams.LambdaDt / (2*numThetaParams)) * ...
    sum(sum(docThetaMat.^2));
objGrad = objGrad + ...
    (sum(docLen) * modelParams.LambdaDt / numThetaParams) .* ...
    docThetaMat;

% plot(cur_docBOW,'kx'); hold on; plot(docProbs,'ro'); hold off; 
 %fprintf(1,'error obj: %f  dt obj: %f\n', ...
 %    objVal, dtRegObj);

 objVal = objVal + dtRegObj;

end


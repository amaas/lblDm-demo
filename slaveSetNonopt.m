function [ ] = slaveSetNonopt( modelParamVec, wNonopt)
%SLAVESETNONOPT Takes data from server to store as local data
% put document data and non-optimized model params into global state
% not all data needs be passed
% if a matrix == -1 it doesn't overwrite the previous data


global state;

if wNonopt ~= -1
    state.wNonopt = wNonopt;
end;

if modelParamVec ~= -1
    state.modelParams = LblDmParam;
    state.modelParams.fromVector(modelParamVec)
end;

end


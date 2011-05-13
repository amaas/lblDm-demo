% this is the main code to run to learn vectors
% it sets up model parameters and what training data to use
% the computing environment (i.e. minfunc path) is also se here

% set flags
clusterRun = 1;

% path setup
addpath minFunc/;

% declare global params used later
global modelParams;
global data_docBOW;


% setup model parameters
modelParams = LblDmParam;
% 20-D vectors seem to capture the data well
modelParams.RepVecDim = 20;
% only the top 1k words in the sample flickr data are seen much
modelParams.DictSize = 1000;
% number of docs to use. run time scales somewhat poorly in this dimension
modelParams.NumDocs = 100000;
% TODO use batchsize?
%modelParams.BatchSize = 5000;
% weight given to gaussian prior on word vectors R and doc coefs Theta
modelParams.LambdaRc = 1e-4;
modelParams.LambdaDt = 1e-4;
% labels stored in BoW file for this dataset
modelParams.LabelFname = '/dev/null';
modelParams.BowFname = 'data/tagVectors.mat';

disp(modelParams)

% initialize parameters. played around with a few initialization 
wInit = .1 * (rand(modelParams.totalNumParams(),1) - .5);
% set random norm 1 vectors to start
%repConMat = rand(modelParams.DictSize,modelParams.RepVecDim)-.5;
%repConMat = bsxfun(@rdivide, repConMat, sqrt(sum(repConMat.^2,2)));
%wInit(modelParams.repConIndex()) = .1 * repConMat(:);
% special care for biases
%corpusBOW = sum(data_docBOW,1);
%corpusBOW = corpusBOW ./ sum(corpusBOW);
%wInit(modelParams.wordBiasIndex()) = 10*corpusBOW;
wInit(modelParams.wordBiasIndex()) = 0;

lblDmAltMF_rpc(wInit);

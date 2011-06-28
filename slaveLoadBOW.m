function [ numLoaded ] = slaveLoadBOW( docInd, bowFname, vocabSize )
%SLAVELOADBOW loads BOW data insto slave state
% docInd is the index into the full data matrix
% loads data and stores it in global used by worker/slave

% assumes word count matrix is stored in a var called data_docBOW
% word count matrix is num documents by num vocab
S = load(bowFname);
data_docBOW = S.data_docBOW;
[numDocs, origVocabSize] = size(data_docBOW);

% trim number of docs, or select specific indices
% ignore top 50 words as they're much more frequent (power/zipf law)
data_docBOW = data_docBOW(:,51:(vocabSize+50));
% remove docs with few tags
data_docBOW = data_docBOW(sum(data_docBOW,2)>2,:);
data_docBOW = data_docBOW(docInd,:);

numDocs = size(data_docBOW,1);
% count total number of words in doc avoiding 0s
data_docLen = full(max(sum(data_docBOW,2), ones(numDocs,1)));
% normalize BOW Distros
data_docBOW = bsxfun(@rdivide, data_docBOW, data_docLen);


% load data into global state
global state;
state.docBow = data_docBOW;
state.docLen = data_docLen;
numLoaded = size(state.docBow,1);
end


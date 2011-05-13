classdef LblDmParam < handle
    %LBLDMPARAM Parameters for a lbl document model optimization
    
    
    properties
        % dimensionality of word vector representations
        RepVecDim = 100;
        % size of input dictionary (and target dictionary)
        DictSize = 400;
        % number of documents in the training set
        NumDocs = 1000;
        % weighting factors on the regularizers
        LambdaRc = 0.00001;
        LambdaDt = 0.0001;
        % store filenames used in training
        BowFname = '';
        LabelFname = '';
        VocabFname = '';
        % number of documents to process at one time in objective
        % batching for computational purposes only, 
        % full batch used always when computing objective
        BatchSize = 100000;
        % if flag is true uses words predicting ratings to train vectors
        % WARNING ratings option not supported by this code
        UseRatings = 0;
        % number of dimensions in rating data. 2 = binary polarity
        NumRatingDims = 2;
        % 0 indicates L2 regularization of doc thetas.
        % otherwise L1 is used, and this sets the logCosh scale
        % WARNING only L2 supported by this code. L1 doesn't work well
        L1Reg = 0;
    end
    
    methods
        % accessors to get indices of params in single vector
        
        % return all params as vector
        function pv = toVector(AP)
            pv = [AP.RepVecDim, AP.DictSize, AP.NumDocs, ...
                AP.LambdaRc, AP.LambdaDt, AP.BatchSize, ...
                AP.UseRatings, AP.NumRatingDims];
        end
        % set params based on vector
        function [] = fromVector(AP, pv)
            AP.RepVecDim = pv(1);
            AP.DictSize = pv(2);
            AP.NumDocs  = pv(3);
            AP.LambdaRc  = pv(4);
            AP.LambdaDt  = pv(5);
            AP.BatchSize = pv(6);
            AP.UseRatings = pv(7);
            AP.NumRatingDims = pv(8);
        end;
        % word representation matrix
        % HACK lblDmObjAltMF assumes repConMat and wb are first 2 params
        function repConInd = repConIndex(AP)
            repConInd = 1:(AP.RepVecDim * AP.DictSize);
        end
        
        % word bias vector
        function wbInd = wordBiasIndex(AP)            
            startInd = (AP.RepVecDim * AP.DictSize)+1;
            wbInd = startInd : (startInd+AP.DictSize-1);
        end
        % weights for softmax layer with biases at the end
        % we use k-1 definition for softmax
        function rlrInd = ratingLrIndex(AP)
            startInd = ((AP.RepVecDim+1) * AP.DictSize)+1;
            rlrInd = startInd : (startInd-1+((AP.NumRatingDims-1)*(AP.RepVecDim+1)));
        end
        % all document transform weights
        function thetaMatInd = thetaMatIndex(AP)     
            matSize = AP.NumDocs * AP.RepVecDim;
            prevInd = NaN;
            if AP.UseRatings
                prevInd = AP.ratingLrIndex();
            else
                prevInd = AP.wordBiasIndex();
            end;
            startInd = prevInd(end)+1;
            thetaMatInd = startInd : (startInd+matSize-1);
        end
        
        % document weights for a given document(s)
        % can also gather many documents in sequence
        function thetaVecInd = thetaVecIndex(AP, docIndStart, numDocs)
            if nargin < 3
                numDocs = 1;
            end;
            vecSize = AP.RepVecDim * numDocs;
            prevInd = NaN;
            if AP.UseRatings
                prevInd = AP.ratingLrIndex();
            else
                prevInd = AP.wordBiasIndex();
            end;            
            startInd = prevInd(end)+(vecSize*(docIndStart-1)) + 1;
            thetaVecInd = startInd : (startInd+vecSize-1);
        end;
        
        % total number of parameters         
        function numParams = totalNumParams(AP)
            thetaMatInd = AP.thetaMatIndex();
            numParams = thetaMatInd(end);
        end
    end
    
end


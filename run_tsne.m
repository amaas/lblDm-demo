% runs t-sne on word vectors to visualize representation similarity
addpath tsne/;
% Load data
load tmp/curParamCache.mat;
repConMat = reshape(wFull(modelParams.repConIndex()),modelParams.DictSize,[]);
vocabString = textread('data/flickrVocab.txt','%s');
% the first 50 words are not used by the model. trim them
vocabString = vocabString(51:end);
% Set parameters
no_dims = 2;
init_dims = size(repConMat,2);
perplexity = 15;

% Run t−SNE
mappedX = 5*tsne(repConMat, [], no_dims, init_dims, perplexity);

% Plot results
%gscatter(mappedX(:,1), mappedX(:,2), train_labels, ’o’);

if 1
figure
scatter(mappedX(:,1),mappedX(:,2),'w.');
%scatter3(mappedX(:,1),mappedX(:,2),mappedX(:,3),'w.');
%axis off;
for i=1:size(mappedX,1)
    text(mappedX(i,1),mappedX(i,2),vocabString{i});%,...
    %text(mappedX(i,1),mappedX(i,2),mappedX(i,3),vocabString{i});%,...
         %'FontName','Andale Mono', 'FontSize',20,'Interpreter','none');
end
end

if 0
    for i = 1:size(docThetaMat,1)
        plot(abs(docThetaMat(i,:)),'o')
        pause
    end

end

% f = fopen('data/flickrVocab.txt','w');
% for i = 1:length(Final_Tag_List)
%     fprintf(f,'%s\n',Final_Tag_List{i});
% end;
% fclose(f);
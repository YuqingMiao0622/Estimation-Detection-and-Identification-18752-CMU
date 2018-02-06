function [ quadTerm, termName ] = getQuadTerm( data, orgF )
% This function takes a data matrix of column features, and computes the
% quadratic terms associated with the features specified by orgF. Quadratic
% terms are normalized to zero mean and unit variance. 

[numSamp,~] = size(data); % number of samples
numOF = length(orgF); % number of original features
if numOF<2
    numQuadF = 2*numOF;
else
    numQuadF = 2*numOF+nchoosek(numOF,2); % number of quadratic terms
end
quadTerm = zeros(numSamp,numQuadF); % quadratic terms

%% add original terms
count = 1;
for i = orgF
    quadTerm(:,count) = data(:,i); % at first it contains the original features
    quadTerm(:,count) = (quadTerm(:,count)-mean(quadTerm(:,count)))/std(quadTerm(:,count)); % normalize
    count = count+1; % count indicated the current column idx
end

%% add square terms
for i = 1:numOF
    quadTerm(:,count) = quadTerm(:,i).^2; % add the square of this feature
    quadTerm(:,count) = (quadTerm(:,count)-mean(quadTerm(:,count)))/std(quadTerm(:,count)); % normalize
    count = count+1; % increment count so that the next term is added after the last column
end % after this point, count should be = 2*F+1

%% add product terms
if numOF>=2
    for i = 1:numOF-1
        for j = i+1:numOF
            quadTerm(:,count) = quadTerm(:,i).*quadTerm(:,j); % add the product between these 2 features
            quadTerm(:,count) = (quadTerm(:,count)-mean(quadTerm(:,count)))/std(quadTerm(:,count)); % normalize
            count = count+1;
        end
    end
end % after this point, count should be 2*F+nchoosek(F,2)+1

%% create string array of feature names
termName = cell(1,numQuadF-1);
for k = 1:numOF
    termName{k} = ['x_{' num2str(orgF(k)) '}']; % linear term
    termName{numOF+k} = ['x_{' num2str(orgF(k)) '}^2']; % quadratic term
end
count = numOF*2;
for k = 1:numOF-1
    for j = k+1:numOF
        termName{count+1} = ['x_{' num2str(orgF(k)) '}x_{' num2str(orgF(j)) '}']; % product term
        count = count+1;
    end
end
end

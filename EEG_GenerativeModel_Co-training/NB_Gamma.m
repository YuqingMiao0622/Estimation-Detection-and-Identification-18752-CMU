function [ Ac, p, k, theta ] = NB_Gamma( dataTrain, dataTest, orgF )
% This function trains a Naive Bayes with Gamma models using dataTrain and
% test the trained model using dataTest. 
% Input:  1/ dataTrain is a Lx(m+1) matrix of features X + 1 column for Y
%         2/ dataTest is a numTx(m+1) matrix of features X and label Y
%         3/ orgF is the set of features being used
% Output: 1/ Ac = testing accuracy
%         2/ p = Bernoulli model parameter
%         3/ (k,theta) = Gamma model parameters
dataTrain = dataTrain(:,[orgF end]);
[L,m] = size(dataTrain); m = m-1;
% minX = repmat(min(dataTrain(:,1:m)),size(dataTrain,1),1);
% dataTrain(:,1:m) = dataTrain(:,1:m)-minX+1e-6;
dataTest = dataTest(:,[orgF end]);
% minX = repmat(min(dataTest(:,1:m)),size(dataTest,1),1);
% dataTest(:,1:m) = dataTest(:,1:m)-minX+1e-6;
[numT,~] = size(dataTest);

%% start training, model parameters = {p,k(y,j),theta(y,j)}
p = sum(dataTrain(:,m+1))/L; % learn p
k = zeros(m,2); % since we are doing Naive Bayes, there are 2*m Gamma(k,theta) models to learn
theta = zeros(m,2);
for y = 0:1
   for j = 1:m
       Xj_y = dataTrain(find(dataTrain(:,m+1)==y),j);
       [k(j,y+1),theta(j,y+1)] = gamma_mle(Xj_y,100); % learn the parameters
   end
end

%% detection using the trained model
PDF_y0 = zeros(size(dataTest));
for j = 1:m
    PDF_y0(:,j) = log(gampdf(dataTest(:,j),k(j,1),theta(j,1))); % take log of likelihood probabilities
end
PDF_y0(:,m+1) = log(1-p); % log of prior 
Py0 = sum(PDF_y0,2); % log of posterior probability

PDF_y1 = zeros(size(dataTest));
for j = 1:m
    PDF_y1(:,j) = log(gampdf(dataTest(:,j),k(j,2),theta(j,2))); % take log of likelihood probabilities
end
PDF_y1(:,m+1) = log(p); % log of prior 
Py1 = sum(PDF_y1,2); % log of posterior probability

y_hat = (Py1>Py0); % predicted label
y_true = dataTest(:,m+1); % true label
Err = sum(abs(y_hat-y_true))/numT; % testing error
Ac = 1-Err; % accuracy = 1-error
end


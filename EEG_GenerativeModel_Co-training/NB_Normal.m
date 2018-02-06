function [ Ac, p, mu, sigma ] = NB_Normal( dataTrain, dataTest )
% This function trains a Naive Bayes with Gamma models using dataTrain and
% test the trained model using dataTest. 
% Input:  1/ dataTrain is a Lx(m+1) matrix of features X + 1 column for Y
%         2/ dataTest is a numTx(m+1) matrix of features X and label Y
% Output: 1/ Ac = testing accuracy
%         2/ p = Bernoulli model parameter
%         3/ (mu,sigma) = Normal model parameters

[L,m] = size(dataTrain); m = m-1;
[numT,~] = size(dataTest);

%% start training
p = sum(dataTrain(:,m+1))/L;
mu = zeros(m,2);
sigma = zeros(m,2);
for y = 0:1
   for j = 1:m 
       Xj_y = dataTrain(find(dataTrain(:,m+1)==y),j);
       mu(j,y+1) = mean(Xj_y); % mean of Normal(mu,sigma^2)
       sigma(j,y+1) = std(Xj_y); % std of Normal(mu,sigma^2)
   end
end

%% detection using trained model
PDF_y0 = zeros(size(dataTest));
for j = 1:m
    PDF_y0(:,j) = normpdf(dataTest(:,j),mu(j,1),sigma(j,1));
end
PDF_y0(:,m+1) = 1-p;
Py0 = prod(PDF_y0,2);

PDF_y1 = zeros(size(dataTest));
for j = 1:m
    PDF_y1(:,j) = normpdf(dataTest(:,j),mu(j,2),sigma(j,2));
end
PDF_y1(:,m+1) = p;
Py1 = prod(PDF_y1,2);

y_hat = (Py1>Py0);
y_true = dataTest(:,m+1);
Err = sum(abs(y_hat-y_true))/numT;
Ac = 1-Err;
end
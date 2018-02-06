function [ y_hat ] = NB_Gamma_detection( dataTest, orgF, p, k, theta )
% Given a NB_Gamma model and some test samples, this function predicts the 
% label Y
% Input:  1/ dataTest: Nxm matrix, N = number of test samples, m = number
% of features. We assume that true label is unknown. 
%         2/ orgF: which features to use among all m features. 
%         3/ p, k, theta: NB_Gamma model parameters. 
% Output: 1/ y_hat: a real-valued predicted label using the given NB_Gamma 
% model. 

dataTest = dataTest(:,[orgF end]);
[~,m] = size(dataTest); m = m-1;

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

y_hat = (Py1-Py0); % predicted label, either positive (y=1) or negative (y=0), but we return the real value here
end


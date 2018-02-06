function [ Ac, optW, optC ] = SVM( dataTrain, dataTest, orgF )
% Given the data set, the function run SVM and report accuracy and model
% parameters
% Input:  1/ dataTrain/dataTest is a Nx(M+1) matrix where the first M columns 
% contains raw features X and the last column is label Y
%         2/ orgF indicate which features are used for training/testing
% Output: 1/ Ac is test accuracy
%         2/ optW and optC are trained model parameters, they are used for 
%            detection: y_hat = optW'*testX+optC

%% SVM initialization
global M N X V Y t lambda;

X = getQuadTerm(dataTrain,orgF)'; % obtain quadratic terms associated with orgF
Y = dataTrain(:,end)*2-1; % change label value: 0 -> -1 and 1 -> 1
[M,N] = size(X); % M = no. features, N = no. training samples
setPara = struct('t',1,'beta',15,'Tmax',1e6,'tol',1e-6,'W',rand(M,1),'C',rand());
lambda = 0.005; % regularization parameter

testX = getQuadTerm(dataTest,orgF)'; % obtain quadratic terms associated with orgF
testY = dataTest(:,end)*2-1; % change label value: 0 -> -1 and 1 -> 1

%% Train SVM using interior point method
V = X.*repmat(Y,1,M)';
t = setPara.t;
beta = setPara.beta;
Tmax = setPara.Tmax;
tol = setPara.tol;
W = setPara.W;
C = setPara.C;
R = ones(N,1) - V'*W - C*Y;
zeta = max(R,0)+0.001;
init_Z = [W; C; zeta];
while (t <= Tmax)
    [optSolution, ~] = solveOptProb_NM(@costFcn, init_Z,tol);
    init_Z = optSolution;
    t = t*beta;
end

%% Compute accuracy using test set
optW = optSolution(1:M);
optC = optSolution(M+1);
Yhat = sign(testX'*optW + optC);
Ac = sum(testY.*Yhat == 1)/length(testY);
end

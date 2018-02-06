function optLambda = getOptLambda(inX, inY, setPara)
% Get the optimal lambda
%
% INPUTS:
%   inX(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   inY(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.beta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiLambda: Optimal lambda value 
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

global cvIdx testFold testIdx lambdaList X Y M N V t lambda;
nLambda = length(lambdaList);
errLambda = zeros(1,nLambda);

for i = 1:nLambda % iteratively pick a value for lambda
    lambda = lambdaList(i);
    for k = 1:6 % iteratively pick a validation set
        if testFold==k
            continue;
        end
        valIdx = (cvIdx == k); % validation set
        valX = inX(:,valIdx);
        valY = inY(valIdx);
        trIdx = ~(testIdx|valIdx); % training set
        X = inX(:,trIdx);
        Y = inY(trIdx);

        [M,N] = size(X); % M = no. features, N = no. training samples
        V = X.*repmat(Y,1,M)';
        t = setPara.t;
        beta = setPara.beta;
        Tmax = setPara.Tmax;
        tol = setPara.tol;
        W = setPara.W;
        C = setPara.C;
        
        %% Train SVM
        S = ones(N,1) - V'*W - C*Y;
        zeta = max(S,0)+0.001;
        init_Z = [W; C; zeta];
        
%         % check if init_Z is feasible
%         R = V'*W + C*Y + zeta - ones(N,1);
%         checkZeta = sum(zeta<=0); % count number of negative or zero elements in zeta
%         checkR = sum(R<=0); % count number of negative or zero elements in R
%         if (checkZeta > 0) || (checkR > 0)
%             disp('init_Z is not feasible'); % init_Z is not feasible
%         end
        
%         checkF = zeros(1,ceil(log(Tmax)/log(beta))); % check if the value of F is decresing after each run
%         runs = 1;
        while (t <= Tmax)
            [optSolution, ~] = solveOptProb_NM(@costFcn, init_Z,tol);
            init_Z = optSolution;
%             [checkF(runs), ~, ~] = feval(@costFcn,optSolution);
            t = t*beta;
%             runs = runs+1;
        end
        
        %% Compute classification error using validation set
        optW = optSolution(1:M);
        optC = optSolution(M+1);
        Yhat = sign(valX'*optW + optC);
        err = sum(valY.*Yhat ~= 1)/length(valY);
        errLambda(i) = errLambda(i) + err;
    end
    errLambda(i) = errLambda(i)/5; % average error for this lambda after 5 cv runs
end
[~,minIdx] = min(errLambda);
optLambda = lambdaList(minIdx); % pick the lambda with smallest cv error
end
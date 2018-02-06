function [optSolution, err] = solveOptProb_NM(costFcn,init_Z,tol)
% Compute the optimal solution using Newton method
%
% INPUTS:
%   costFcn: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%
% OUTPUTS:
%   optSolution: Optimal soultion
%   err: Error
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

global V Y M N;
Z = init_Z;
err = 1;
% errList = [];

while (err/2) > tol
    % Execute the cost function at the current iteration
    % F : function value, G : gradient, H, hessian
    [~, G, H] = feval(costFcn,Z);
    delZ = -H\G;
    s = 1;
    newZ = Z + s*delZ;
    while true
        W = newZ(1:M);
        C = newZ(M+1);
        zeta = newZ(M+2:end);
        R = V'*W + C*Y + zeta - ones(N,1);
        checkZeta = sum(zeta<=0); % count number of negative or zero elements in zeta
        checkR = sum(R<=0); % count number of negative or zero elements in R
        if (checkZeta == 0) && (checkR == 0)
            break; % newZ is feasible
        end
        % if newZ is not feasible, decrease s and try again
        s = 0.5*s;
        newZ = Z + s*delZ;
    end
    Z = newZ;
    err = -G'*delZ; % err should be positive
%     errList = [errList err];
end
optSolution = Z;
end
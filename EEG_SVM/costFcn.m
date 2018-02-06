function [F, G, H] = costFcn(Z)
% Compute the cost function F(Z)
%
% INPUTS: 
%   Z: Parameter values
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

% Assume that X(MxN) and Y(Nx1) are pre-loaded
% Also, hyper-parameters t and lambda must be already set

global X Y V M N t lambda;

%% Preparation
W = Z(1:M);
C = Z(M+1);
zeta = Z(M+2:end);
R = V'*W + C*Y + zeta - ones(N,1);
D = R.^(-1);
E = zeta.^(-1);

%% Compute F
F = sum(zeta) + lambda*(W'*W) - (1/t)*sum(log(R)) - (1/t)*sum(log(zeta));

%% Compute G
Gw = 2*lambda*W - (1/t)*V*D; % gradient wrt W
Gc = -(1/t)*Y'*D; % gradient wrt C
Gz = ones(N,1) - (1/t)*D - (1/t)*E; % gradient wrt zeta
G = [Gw; Gc; Gz];

%% Compute H
D2 = diag(D.^2);
Hw = 2*lambda*eye(M) + (1/t)*V*D2*V'; % M x M
Hwc = (1/t)*(D.^2)'*X'; % 1 x M
Hc = (1/t)*sum(D.^2); % 1 x 1
Hwz = (1/t)*D2*V'; % N x M
Hcz = (1/t)*D2*Y; % N x 1
Hz = (1/t)*(D2+diag(E.^2)); % N x N
H = [Hw Hwc' Hwz'; Hwc Hc Hcz'; Hwz Hcz Hz];
end
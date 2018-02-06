function [ k_hat, theta_hat ] = gamma_mle( X, numIter )
%% this function computes MLE for gamma(k,theta) model by iteratively updating the k parameter 
% X is a vector

avg_log_X = mean(log(X));
log_avg_X = log(mean(X));
k_old = 0.5/(log_avg_X - avg_log_X); % initialize k
for i = 1:numIter
    k_hat = 1/( 1/k_old + (avg_log_X-log_avg_X+log(k_old)-psi(k_old))/(k_old^2*(1/k_old-psi(1,k_old))) );
    k_old = k_hat;
end
theta_hat = mean(X)/k_hat;
end
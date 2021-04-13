function [L, S] = inexact_alm_pcp(D, lambda, display)
%this code solves the following PCP problem by ADMM
% min_{L, S} |L|_* + lambda |S|_1, s.t., D = L + S
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse term 

[m,n] = size(D);

if nargin < 2 || isempty(lambda)
    lambda = 1 / sqrt(max(m,n));
end

if nargin < 3 || isempty(display)
   display = false; 
end
tol = 1e-7;
maxIter = 1000;
% initialize
Y = D;
norm_two = norm(Y, 2);
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

L = zeros( m, n);
mu = 1.25/norm_two;% this one can be tuned
mu_bar = mu * 1e7;
rho = 1.4 + 0.1*tanh(40 - min(m,n)); % this one can be tuned
d_norm = norm(D, 'fro');

iter = 0;
while iter < maxIter
    iter = iter + 1;
    
    temp = D - L + (1/mu)*Y;
    S = max(temp - lambda/mu, 0) + min(temp + lambda/mu, 0);

    temp = D - S + (1/mu)*Y;
    [U, sig, V] = svd(temp, 'econ');
    diagS = diag(sig);
    svp = length(find(diagS > 1/mu));
    diagS = max(0,diagS - 1/mu);
    svp = max(1, svp);
    L = U(:, 1:svp) * bsxfun(@times,V(:, 1:svp)',diagS(1:svp));

    Z = D - L - S;    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if display && (mod(iter, 100) == 0 || iter == 1 || stopCriterion < tol)
        disp(['#iter ' num2str(iter) ', mu= ' num2str(mu) ', stopALM (PCP) ' num2str(stopCriterion)]);
    end 
    if stopCriterion < tol
        break;
    end       
end

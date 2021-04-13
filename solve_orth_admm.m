function [R] = solve_orth_admm(X, Y, display)
%LEARN_U 此处显示有关此函数的摘要
%   此处显示详细说明
% min |RX-Y|_1 s.t. R^TR = I
% input: X --- m x n matrix, with each column being a sequence
%        Y --- p x n sparse matrix, p>=m
% output: R --- p x m column-wisely orthogonal matrix
if nargin < 3 || isempty(display)
    display = false;
end

n = size(X, 2);
p = size(Y, 1);

maxIter = 2000;
tol = 1e-7;

fnorm = norm(Y,'fro');
two_norm = norm(X, 2);

% initialize
W = zeros(p,n); %Lagrange multiplier 
L = zeros(p,n);

%parameters
mu = 1/two_norm;  % this one can be tuned
rho = 1.1;   % this one can be tuned

iter = 0;
%% loop
while iter < maxIter       
    iter = iter + 1;
    %update R
    temp = (Y + L - W/mu)*X';
    [U,~,V] = svd(temp,'econ');
    R = U*V';
    
    %update L
    temp = R*X - Y + W/mu;
    L = max(0,temp - 1/mu)+ min(0, temp + 1/mu);
        
    H = R*X - Y - L;
    %% stop Criterion    
    stopCriterion = norm(H, 'fro') / fnorm;
    if display && (iter == 1 || mod(iter,100)==0 || stopCriterion < tol)
        disp(['Iteration #' num2str(iter) ', mu ' num2str(mu) ', stopALM (L1Min) ' num2str(stopCriterion)]);
    end
    
    if stopCriterion < tol
        break;
    else
        W = W + mu*H;
        mu = min(mu*rho,10^10);
    end    
end

end


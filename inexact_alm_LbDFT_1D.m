function [x] = inexact_alm_LbDFT_1D(y, g, U, lambda)
%%% This matlab code implements the (exact) augmented Lagrange multiplier method for
%%% min_{x}  |F(U*x)|_1 + 0.5*lambda |P_g(x - y)|_2^2
%%% F -- the DFT operator
%% inputs:
% y -- m x 1 vector of observations (required input),with the missing entries being filled with zero
% g -- m x 1 vector with 1 and 0 denoting the observed and missing entries respecitvely (required input) 
% U -- p x m orthogonal matrix
% lambda -- weight parameter
[p, m ] = size(U); 
maxIter = 2000;
tol = 1e-7;
fnorm = norm(y);

% initialize
W = zeros(p,1); %Lagrange multiplier 
x = y;
ftux = fft(U*x);
%parameters
obsrate = sum(g>0.5)/m;
mu = obsrate/fnorm; % this one can be tuned
rho = 1.2; %this one needs be tuned 

iter = 0;
%% loop
while iter < maxIter       
    iter = iter + 1;
    
    %update L
    temp = ftux + W./mu;
    L = shrink_l2(temp,1/mu);
        
    %update x
    temp = lambda*y + U'*ifft(mu*L - W);
    diagA = lambda*g + mu;
    x = temp./diagA;
        
    ftux = fft(U*x);
    
    H = ftux - L;
    %% stop Criterion    
    stopCriterion = norm(H, 'fro') / fnorm;
    if stopCriterion < tol
        break;
    else
        W = W + mu*H;
        mu = min(mu*rho,10^10);
    end    
end
x = real(x);
end


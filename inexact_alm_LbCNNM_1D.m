function [x] = inexact_alm_LbCNNM_1D(y, g,  Q, n, lambda)
%%% This matlab code implements the (inexact) augmented Lagrange multiplier method for
%%% min_{x}  |A(Q*x)|_* + 0.5*n*lambda |P_Omega(x - y)|_2^2
%%% A() -- linear operator that returns the convolutional matrix of a vector
%%% | |_* -- the nuclear norm of a matrix
%% parameters:
% y -- m x 1 vector of observations (required input),with the missing entries being filled with zero
% g -- m x 1 vector with 1 and 0 denoting the observed and missing entries respecitvely (required input) 
% Q --- p x m column-wisely orthogonal matrix (required input) 
% n -- kernel size (required input) 
% lambda -- weight parameter
%      - DEFAULT 1000

p = size(Q, 1);
maxIter = 2000;
tol = 1e-7;

if nargin < 5 || isempty(lambda)
    lambda = 1000;
end

fnorm = sqrt(n)*norm(y);

% initialize
W = zeros(p, n); %Lagrange multiplier 
x = y;

%parameters
obsrate = sum(g > 0.5)/n;
mu = obsrate/fnorm; % this one can be tuned
rho = 1.05; % this one can be tuned
iter = 0;
%% loop
Ax = cconv1mtx(Q*x,n);
while iter < maxIter       
   iter = iter + 1;   
    %update L
    temp = Ax + W./mu;
    [U,S,V] = svd(temp, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    diagS = max(0,diagS - 1/mu);
    if svp < 0.5 %svp = 0
        svp = 1;
    end
    diagS = diagS(1:svp);
    U = U(:,1:svp);
    V = V(:,1:svp);
    L =U*bsxfun(@times,V',diagS);  
        
    %update x
    temp = Q'*adj_1D(mu*L - W)/n;
    temp = lambda*y+temp;
    diagA = lambda*g + mu;
    x = temp./diagA;
    Ax = cconv1mtx(Q*x,n);
     
    H = Ax - L;
    %% stop Criterion    
    stopCriterion = norm(H, 'fro') / fnorm; 
    if stopCriterion < tol
        break;
    else
        W = W + mu*H;
        mu = min(mu*rho,10^10);
    end    
end
end


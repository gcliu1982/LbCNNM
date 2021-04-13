function [pred] = LbCNNM_pred(y, g, A, k, lambda)
%%% reovering a sequence from a subset from its entries by using either LbCNNM or LbDFT
%%% inputs:
% y -- m x 1 vector of observations (required input),with the missing entries being filled with zero
% g -- m x 1 vector with 1 and 0 denoting the observed and missing entries respecitvely (required input) 
% A --- p x m column-wisely orthogonal matrix (required input) 
% k -- kernel size  (required input) 
% lambda --- regularization parameter (optional), default lambda = 100
%%% output: pred --- h forecasts
if nargin < 5 || isempty(lambda)
    lambda = 1000;
end

p = size(A, 1);
if k > p
    error('kernel size should not be greater than dictionary size!');
elseif k == p
    x_hat = inexact_alm_LbDFT_1D(y, g, A, lambda);
else
    x_hat = inexact_alm_LbCNNM_1D(y, g, A, k, lambda);
end
pred = x_hat(g < 0.5);

end



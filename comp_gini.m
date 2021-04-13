function [g] = comp_gini(x)
%COMP_GINI 此处显示有关此函数的摘要
%   此处显示详细说明
x = abs(x(:))';
n = length(x);
k = 1:n;
x_norm = sum(x);
x = sort(x,'ascend');
y = x.*(n - k + 0.5)/(x_norm*n);

g = 1 - 2*sum(y);
end


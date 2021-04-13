function [A] = LbCLM_train(Y)
%LEARN_U 此处显示有关此函数的摘要
%   此处显示详细说明
% input: Y --- m x n matrix, with each column being a training sample 
% output: A --- 2*m x m column-wisely orthogonal matrix
[m, n] = size(Y);
if rank(Y) < 3
    if n > m
        [U,~,~] = svd(Y, 'econ');
    else
        [U,~,~] = svd(Y);
    end
    B = [U'; zeros(m)];
else
    try
        [L, S] = inexact_alm_pcp(Y);
    catch
        L = Y;
        S = zeros(m, n);
    end
    if n > m
        [U,~,~] = svd(L,'econ');
    else
        [U,~,~] = svd(L);
    end
    Y = L + S;
    E = [U'*L;S];
    B = solve_orth_admm(Y, E);
end

[U,V] =comp_UV(2*m); 
A = V*U'*B;
end

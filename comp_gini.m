function [g] = comp_gini(x)
%COMP_GINI �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
x = abs(x(:))';
n = length(x);
k = 1:n;
x_norm = sum(x);
x = sort(x,'ascend');
y = x.*(n - k + 0.5)/(x_norm*n);

g = 1 - 2*sum(y);
end


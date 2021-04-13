function [G] = get_gematrix(x, m)
%compute the generation matrix of a series x, m -- model size
len = length(x);
n_max = len - m + 1;
G = zeros(m, n_max);
for i = 1 : n_max
    G(:,i) = x(i : i + m - 1);
end

end

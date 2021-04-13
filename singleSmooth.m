function [s] = singleSmooth(data, alpha)
% Calculates single exponentially smoothed data with weight parameter
% alpha.

len = length(data);
s = zeros(len+1, 1);
s(2) = data(1);
for i = 3 : len+1
    s(i) = alpha*data(i-1) + (1-alpha)*s(i-1);
end

end
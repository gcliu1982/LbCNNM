function [pred] = ExpSmooth_pred(x, s, h, alpha)
y0 = x(end);
pred = zeros(h, 1);
pred(1) = s(end); % smoothed value at origin
for i = 1 : h - 1
    pred(i+1) = alpha*y0 + (1-alpha)*pred(i);
end
end


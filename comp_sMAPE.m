function [s] = comp_sMAPE(pred, gt)
h = length(gt);
temp = abs(pred - gt)./(abs(pred) + abs(gt));
s = 200*sum(temp)/h;
end


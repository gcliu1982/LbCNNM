function [s] = comp_nrmse(pred, gt)
y = mean(abs(gt));
s = 100*sqrt(mean((pred-gt).^2))/y;
end


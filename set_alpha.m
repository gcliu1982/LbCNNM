function [alphas] = set_alpha(f_hat)
%set the parameters for ExpSmooth
if f_hat > 10
    alphas = 0.05;
elseif f_hat > 5
    alphas = 0.05 : 0.05 : 0.1;
elseif f_hat > 2.5
    alphas = 0.5 : 0.05 : 1;
elseif f_hat > 1.25
    alphas = 0.7 : 0.05 : 1;
else
    alphas = 0.9 : 0.05 : 1;
end
end

